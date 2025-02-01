import torch
import torch.nn.functional as F

import numpy as np

from lib.metrics import pdist, corr_dist
from util.misc import _hash


def contrastive_hardest_negative_loss(
    F0, F1, positive_pairs, pos_thresh, neg_thresh, num_pos=5192, num_hn_samples=2048
):
    """
    Generate negative pairs
    """
    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)
    hash_seed = max(N0, N1)
    sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
    sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

    if N_pos_pairs > num_pos:
        pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
        sample_pos_pairs = positive_pairs[pos_sel]
    else:
        sample_pos_pairs = positive_pairs

    # Find negatives for all F1[positive_pairs[:, 1]]
    subF0, subF1 = F0[sel0], F1[sel1]

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()
    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    D01 = pdist(posF0, subF1, dist_type="L2")
    D10 = pdist(posF1, subF0, dist_type="L2")

    D01min, D01ind = D01.min(1)
    D10min, D10ind = D10.min(1)

    if not isinstance(positive_pairs, np.ndarray):
        positive_pairs = np.array(positive_pairs, dtype=np.int64)

    pos_keys = _hash(positive_pairs, hash_seed)

    D01ind = sel1[D01ind.cpu().numpy()]
    D10ind = sel0[D10ind.cpu().numpy()]
    neg_keys0 = _hash([pos_ind0.numpy(), D01ind], hash_seed)
    neg_keys1 = _hash([D10ind, pos_ind1.numpy()], hash_seed)

    mask0 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False))
    )
    mask1 = torch.from_numpy(
        np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False))
    )
    pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - pos_thresh)
    neg_loss0 = F.relu(neg_thresh - D01min[mask0]).pow(2)
    neg_loss1 = F.relu(neg_thresh - D10min[mask1]).pow(2)
    return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2


# positive anchors do not need such special treatment,
# so we only consider negative anchors here
def contrastive_hardest_negative_loss_anc(
    soutput0,
    soutput1,
    estimated_results,
    positive_pairs,
    pos_thresh,
    neg_thresh,
    num_pos=5192,
):
    """
    Generate negative pairs
    """
    F0 = soutput0.F
    F1 = soutput1.F

    N0, N1 = len(F0), len(F1)
    N_pos_pairs = len(positive_pairs)

    if N_pos_pairs > num_pos:
        pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
        sample_pos_pairs = positive_pairs[pos_sel]
    else:
        sample_pos_pairs = positive_pairs

    pos_ind0 = sample_pos_pairs[:, 0].long()
    pos_ind1 = sample_pos_pairs[:, 1].long()

    posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

    posC0, posC1 = (
        soutput0.C[pos_ind0].detach().cpu().numpy(),
        soutput1.C[pos_ind1].detach().cpu().numpy(),
    )

    pos_loss = 0
    neg_loss = 0

    # repeat these anchors with corr. lengths
    # do not detach here: for teacher, we do not detach tensors.
    # so this job is done by the trainer
    for i, estimated_result in enumerate(estimated_results):
        pos_loss += (
            F.relu(
                (
                    posF0[posC0[:, 0] == i]
                    - estimated_result["meta_tgt_positive"] # .detach()
                )
                .pow(2)
                .sum(1)
                - pos_thresh
            ).mean()
            + F.relu(
                (
                    estimated_result["meta_src_positive"] # .detach()
                    - posF1[posC1[:, 0] == i]
                )
                .pow(2)
                .sum(1)
                - pos_thresh
            ).mean()
        ).mean() / 2.0

        neg_loss += (
            F.relu(
                neg_thresh
                - (
                    posF0[posC0[:, 0] == i]
                    - estimated_result["meta_tgt_negative"] # .detach()
                )
                .pow(2)
                .sum(1)
            )
            .mean()
            + F.relu(
                neg_thresh
                - (
                    estimated_result["meta_src_negative"] # .detach()
                    - posF1[posC1[:, 0] == i]
                )
                .pow(2)
                .sum(1)
            )
            .mean()
        ).mean() / 2.0

    if not isinstance(pos_loss, torch.Tensor):
        pos_loss = torch.tensor(pos_loss).to(posF0.device)

    if not isinstance(neg_loss, torch.Tensor):
        neg_loss = torch.tensor(neg_loss).to(posF0.device)

    pos_loss /= float(len(estimated_results))
    neg_loss /= float(len(estimated_results))


    pos_result = pos_loss.mean()
    neg_result = neg_loss.mean()
    return pos_result, neg_result
