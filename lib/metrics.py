import numpy as np

import torch
import torch.nn.functional as F


def compound_similarity_metric(feats: torch.Tensor, anchor: torch.Tensor):
    """
    feats: [N,d]
    anchor: [d]
    """
    cosine_dists = (1 + F.cosine_similarity(feats, anchor[None, :], dim=-1)) / 2.0
    euclidean_dists = 1.0 - torch.min(
        torch.norm(feats - anchor[None, :], dim=-1),
        torch.tensor(1.0).to(device=feats.device),
    )
    compound_feat_similarity = torch.min(cosine_dists, euclidean_dists)
    return compound_feat_similarity


def eval_metrics(output, target):
    output = (F.sigmoid(output) > 0.5).cpu().data.numpy()
    target = target.cpu().data.numpy()
    return np.linalg.norm(output - target)


def corr_dist(est, gth, xyz0, xyz1, weight=None, max_dist=1):
    xyz0_est = xyz0 @ est[:3, :3].t() + est[:3, 3]
    xyz0_gth = xyz0 @ gth[:3, :3].t() + gth[:3, 3]
    dists = torch.clamp(torch.sqrt(((xyz0_est - xyz0_gth).pow(2)).sum(1)), max=max_dist)
    if weight is not None:
        dists = weight * dists
    return dists.mean()


def pdist(A, B, dist_type="L2"):
    if dist_type == "L2":
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == "SquareL2":
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    else:
        raise NotImplementedError("Not implemented")


def pairwise_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    normalized: bool = False,
    channel_first: bool = False,
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(
            x.transpose(-1, -2), y
        )  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(
            x, y.transpose(-1, -2)
        )  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x**2, dim=channel_dim).unsqueeze(
            -1
        )  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y**2, dim=channel_dim).unsqueeze(
            -2
        )  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


def get_loss_fn(loss):
    if loss == "corr_dist":
        return corr_dist
    else:
        raise ValueError(f"Loss {loss}, not defined")


