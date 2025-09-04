import os
import os.path as osp
import gc
import logging
import numpy as np
import json
from easydict import EasyDict as edict
import copy
import traceback
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from tensorboardX import SummaryWriter

from model import load_model
import util.transform_estimation as te
from lib.metrics import pdist, corr_dist, pairwise_distance
from lib.ops import apply_transform
from lib.timer import Timer, AverageMeter
from lib.eval import find_nn_gpu
from lib.registration import compute_registration_error
from lib.utils import release_cuda, optional_config

from util.file import ensure_dir
from util.misc import _hash

from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import ball_query, knn_gather
import pytorch3d

from scripts.SC2_PCR.SC2_PCR import Matcher

import MinkowskiEngine as ME

from lib.trainer.base import *


from lib.metrics import compound_similarity_metric
from lib.loss import (
    circle_loss,
    weighted_circle_loss,
    contrastive_hardest_negative_loss,
    contrastive_hardest_negative_loss_anc,
    supervised_infonce_loss,
)
from lib.tta import TTAModelParameterProtector


def corr_feats_from_point_feats(feats1, feats2):
    corr_feats = torch.cat([feats1, feats2], dim=-1)
    corr_feats = F.normalize(corr_feats, p=2, dim=-1)
    return corr_feats


class ContinuousCorrExtensionIntegerTrainer(CorrespondenceExtensionTrainer):
    def __init__(
        self,
        config,
        data_loader,
        val_data_loader=None,
    ):
        super().__init__(config, data_loader, val_data_loader)

        if "manual_seed" in config:
            self.manual_seed = int(config.manual_seed)
            print(f"Trainer is using manual seed {self.manual_seed}")
        else:
            self.manual_seed = 0

        self.clean_cache = config.clean_cache

        self.labeler_bn_used = optional_config(config, "labeler_bn_used", False)

        self.pos_radius = config.circle_pos_radius
        self.neg_radius = config.circle_neg_radius
        self.pos_optimal = config.circle_pos_optimal
        self.neg_optimal = config.circle_neg_optimal
        self.log_scale = config.circle_log_scale
        self.circle_max_points = config.circle_max_points

        self.rediscovery_radius = config.rediscovery_radius
        self.adaptive_rediscovery = config.adaptive_rediscovery

        self.iterative_max_iter = config.iterative_max_iter
        self.iterative_rre_thres = config.iterative_rre_thres
        self.iterative_rte_thres = config.iterative_rte_thres
        self.iterative_suppl_ratio = config.iterative_suppl_ratio
        self.iterative_suppl_max_corr = config.iterative_suppl_max_corr

        self.dstl_anchor_weight = config.dstl_anchor_weight
        self.neg_weight = config.neg_weight

        self.enable_labeler_tta = config.enable_labeler_tta
        self.tta_rediscovery_radius = config.tta_rediscovery_radius

        self.enable_mixed_density = config.enable_mixed_density
        self.mixed_density_loss = config.mixed_density_loss
        assert self.mixed_density_loss in ["circle", "contrastive"]

        self.mixed_density_loss_weight = config.mixed_density_loss_weight
        self.mixed_density_cross_weight = optional_config(
            config, "mixed_density_cross_weight", 0.1
        )
        if self.mixed_density_loss == "circle":
            self.mixed_density_circle_max_points = optional_config(
                config, "mixed_density_circle_max_points", 32
            )
        elif self.mixed_density_loss == "contrastive":
            self.base_matching_search_voxel_size = (
                config.voxel_size * config.positive_pair_search_voxel_size_multiplier
            )

        self.loss_type = config.loss_type
        assert self.loss_type in ["circle", "contrastive"]

        self.tta_corr_num_pos = config.tta_corr_num_pos
        self.tta_corr_weight = config.tta_corr_weight
        self.tta_corr_temperature = config.tta_corr_temperature

        self.enable_distillation = config.enable_distillation
        self.dstl_num_pos = config.dstl_num_pos
        self.dstl_weight = config.dstl_weight
        self.dstl_temperature = config.dstl_temperature

        self.EPS = 1e-6

    def contrastive_hardest_negative_loss(self, F0, F1, positive_pairs):
        num_pos: int = self.config.num_pos_per_batch * self.config.batch_size
        num_hn_samples: int = (
            self.config.num_hn_samples_per_batch * self.config.batch_size
        )

        pos_loss, neg_loss = contrastive_hardest_negative_loss(
            F0,
            F1,
            positive_pairs,
            pos_thresh=self.pos_thresh,
            neg_thresh=self.neg_thresh,
            num_pos=num_pos,
            num_hn_samples=num_hn_samples,
        )
        return pos_loss, neg_loss

    def circle_loss(
        self,
        xyz0,
        xyz1,
        feats0,
        feats1,
        T_est,
        input_is_corr=True,
        pos_radius_override=None,
        neg_radius_override=None,
    ):
        T_est = torch.tensor(T_est, device=xyz0.device)
        transformed_P0 = apply_transform(xyz0, T_est)

        dists = pairwise_distance(transformed_P0, xyz1)

        if pos_radius_override is None:
            pos_mask = dists < self.pos_radius
        else:
            pos_mask = dists < pos_radius_override

        if neg_radius_override is None:
            neg_mask = dists >= self.neg_radius
        else:
            neg_mask = dists >= neg_radius_override

        if input_is_corr:
            # Ensure the given positive pairs are marked as positive and not negative
            corr_mask = torch.eye(xyz0.shape[0], device=pos_mask.device)
            pos_mask = torch.logical_or(pos_mask, corr_mask)
            neg_mask = torch.logical_and(neg_mask, torch.logical_not(corr_mask))

        feat_dists = pairwise_distance(feats0, feats1)

        c_loss = circle_loss(
            pos_mask,
            neg_mask,
            feat_dists,
            self.pos_thresh,
            self.neg_thresh,
            self.pos_optimal,
            self.neg_optimal,
            self.log_scale,
        )

        return c_loss

    def compose_main_registration_loss(
        self,
        input_dict,
        soutput0,
        soutput1,
        T_est,
        pos_pairs,
        uncollated_pairs,
        estimated_results,
    ):

        iter_size = self.iter_size

        if self.loss_type == "contrastive":
            pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                soutput0.F, soutput1.F, pos_pairs
            )

            pos_loss /= iter_size
            neg_loss /= iter_size
            loss = pos_loss + self.neg_weight * neg_loss

            result = {
                "loss": loss,
                "pos_loss": pos_loss,
                "neg_loss": neg_loss,
            }

            return result
        elif self.loss_type == "circle":
            _, batch_student_feats_0 = soutput0.decomposed_coordinates_and_features
            _, batch_student_feats_1 = soutput1.decomposed_coordinates_and_features

            c_loss = 0
            for batch_idx, T in enumerate(T_est):
                correspondence = uncollated_pairs[batch_idx]

                if correspondence.size(0) > self.circle_max_points:
                    choice = np.random.permutation(correspondence.size(0))[
                        : self.circle_max_points
                    ]
                    correspondence = correspondence[choice]

                xyz0_indice = correspondence[:, 0]
                xyz1_indice = correspondence[:, 1]

                xyz0_corr = input_dict["pcd0"][batch_idx][xyz0_indice]
                xyz1_corr = input_dict["pcd1"][batch_idx][xyz1_indice]

                xyz0_feats = batch_student_feats_0[batch_idx][xyz0_indice]
                xyz1_feats = batch_student_feats_1[batch_idx][xyz1_indice]

                loss = self.circle_loss(
                    xyz0_corr,
                    xyz1_corr,
                    xyz0_feats,
                    xyz1_feats,
                    T,
                    input_is_corr=True,
                    pos_radius_override=None,
                    neg_radius_override=None,
                )

                c_loss += loss

            c_loss /= len(T_est)
            result = {
                "loss": c_loss,
                "pos_loss": torch.ones(1) * float(c_loss.detach().cpu().item()),
                "neg_loss": torch.zeros(1),
            }
            return result

        # Should not reach here
        raise ValueError(f"Unknown loss type {self.loss_type}")

    def nce_dstl_loss(
        self, feats_corr_0, feats_corr_1, corr_labels, pos_anchor, neg_anchor
    ):
        corr_feats = corr_feats_from_point_feats(feats_corr_0, feats_corr_1)

        has_good_ancs = (
            torch.norm(pos_anchor - neg_anchor)
            >= np.abs(self.neg_thresh - self.pos_thresh)
        )

        if has_good_ancs:
            aug_raw_corr_feats = torch.cat(
                [corr_feats, pos_anchor[None, :], neg_anchor[None, :]], dim=0
            )
            aug_raw_labels = torch.cat(
                [
                    corr_labels,
                    torch.ones(1).to(corr_labels.device),
                    torch.zeros(1).to(corr_labels.device),
                ],
                dim=0,
            ).bool()[None, :]
        else:
            aug_raw_corr_feats = torch.cat([corr_feats], dim=0)
            aug_raw_labels = torch.cat(
                [corr_labels],
                dim=0,
            ).bool()[None, :]


        cont_loss = supervised_infonce_loss(
            aug_raw_corr_feats[None, :],
            aug_raw_labels,
            temperature=self.dstl_temperature,
            reduction="mean",
        )


        cont_loss = torch.nan_to_num(cont_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return cont_loss

    def compose_mixed_density_circle_loss(
        self,
        xyz0,
        xyz1,
        xyz0_feats,
        xyz1_feats,
        T,
        supplement_pcd0,
        supplement_pcd1,
        supplement0_feats,
        supplement1_feats,
        positive_pairs_c0s1,
        positive_pairs_s0c1,
        positive_pairs_s0s1,
    ):
        md_pos_radius = self.pos_radius
        md_neg_radius = self.rediscovery_radius

        s0_ind = positive_pairs_s0s1[:, 0].long()
        s1_ind = positive_pairs_s0s1[:, 1].long()
        loss3 = self.circle_loss(
            supplement_pcd0[s0_ind].to(device=xyz1.device),
            supplement_pcd1[s1_ind].to(device=xyz0.device),
            supplement0_feats[s0_ind].to(device=xyz1_feats.device),
            supplement1_feats[s1_ind].to(device=xyz0_feats.device),
            T,
            input_is_corr=True,
            pos_radius_override=md_pos_radius,
            neg_radius_override=md_neg_radius,
        )

        loss = loss3
        loss = torch.nan_to_num(loss, nan=0, posinf=0, neginf=0)

        return loss

    def compose_mixed_density_contrastive_loss(
        self,
        F0,
        F1,
        supF0,
        supF1,
        positive_pairs_c0s1,
        positive_pairs_s0c1,
        positive_pairs_s0s1,
    ):
        num_pos: int = self.config.num_pos_per_batch * self.config.batch_size // 4
        num_hn_samples: int = (
            self.config.num_hn_samples_per_batch * self.config.batch_size
        ) // 4

        pos_loss_3, neg_loss_3 = contrastive_hardest_negative_loss(
            supF0,
            supF1,
            positive_pairs_s0s1,
            pos_thresh=self.pos_thresh,
            neg_thresh=self.neg_thresh,
            num_pos=num_pos,
            num_hn_samples=num_hn_samples,
        )

        pos_loss = pos_loss_3
        neg_loss = neg_loss_3

        return pos_loss, neg_loss

    def compose_mixed_density_loss(
        self,
        input_dict,
        soutput0,
        soutput1,
        T_est,
        uncollated_pairs,
        suppl_soutput0,
        suppl_soutput1,
        mixed_density_results,
    ):
        iter_size = self.iter_size

        assert mixed_density_results is not None, "Mixed density results are not given"

        # Mixed density training
        m_loss = 0

        if self.mixed_density_loss == "circle":
            for batch_idx, T in enumerate(T_est):
                _, batch_student_feats_0 = soutput0.decomposed_coordinates_and_features
                _, batch_student_feats_1 = soutput1.decomposed_coordinates_and_features
                _, batch_student_suppl_feats_0 = (
                    suppl_soutput0.decomposed_coordinates_and_features
                )
                _, batch_student_suppl_feats_1 = (
                    suppl_soutput1.decomposed_coordinates_and_features
                )

                loss = self.compose_mixed_density_circle_loss(
                    input_dict["pcd0"][batch_idx],
                    input_dict["pcd1"][batch_idx],
                    batch_student_feats_0[batch_idx],
                    batch_student_feats_1[batch_idx],
                    T,
                    input_dict["supplement_pcd0"][batch_idx],
                    input_dict["supplement_pcd1"][batch_idx],
                    batch_student_suppl_feats_0[batch_idx],
                    batch_student_suppl_feats_1[batch_idx],
                    mixed_density_results["c0s1"][batch_idx].cpu(),
                    mixed_density_results["s0c1"][batch_idx].cpu(),
                    mixed_density_results["s0s1"][batch_idx].cpu(),
                )

                loss = torch.nan_to_num(loss, nan=0, posinf=0, neginf=0)
                loss /= len(T_est)
                m_loss += loss
        elif self.mixed_density_loss == "contrastive":
            m_pos, m_neg = self.compose_mixed_density_contrastive_loss(
                soutput0.F,
                soutput1.F,
                suppl_soutput0.F,
                suppl_soutput1.F,
                mixed_density_results["c0s1"].cpu(),
                mixed_density_results["s0c1"].cpu(),
                mixed_density_results["s0s1"].cpu(),
            )

            m_pos /= iter_size
            m_neg /= iter_size

            m_loss = m_pos + self.neg_weight * m_neg
            m_loss = torch.nan_to_num(m_loss, nan=0, posinf=0, neginf=0)
        else:
            raise ValueError(
                f"Unknown mixed density loss type {self.mixed_density_loss}"
            )

        return m_loss

    def compose_loss_simple(
        self,
        input_dict,
        soutput0,
        soutput1,
        T_est,
        pos_pairs,
        uncollated_pairs,
        estimated_results,
        suppl_soutput0=None,
        suppl_soutput1=None,
        mixed_density_results=None,
    ):
        loss_dict = self.compose_main_registration_loss(
            input_dict,
            soutput0,
            soutput1,
            T_est,
            pos_pairs,
            uncollated_pairs,
            estimated_results,
        )

        loss = loss_dict["loss"]

        # Apply distillation loss
        if self.enable_distillation:
            _, batch_student_feats_0 = soutput0.decomposed_coordinates_and_features
            _, batch_student_feats_1 = soutput1.decomposed_coordinates_and_features

            d_loss = 0
            for b, results in enumerate(estimated_results):
                feats_0 = batch_student_feats_0[b]
                feats_1 = batch_student_feats_1[b]

                src_corr_feats = feats_0[results["src_keypts_corr_indices"]]
                tgt_corr_feats = feats_1[results["tgt_keypts_corr_indices"]]
                src_neg_corr_feats = feats_0[results["neg_src_keypts_corr_indices"]]
                tgt_neg_corr_feats = feats_1[results["neg_tgt_keypts_corr_indices"]]

                corr_labels = torch.cat(
                    [
                        torch.ones(src_corr_feats.shape[0]),
                        torch.zeros(src_neg_corr_feats.shape[0]),
                    ]
                ).to(src_corr_feats.device)

                src_feats = torch.cat(
                    [src_corr_feats, src_neg_corr_feats], dim=0
                ).detach()

                tgt_feats = torch.cat(
                    [tgt_corr_feats, tgt_neg_corr_feats], dim=0
                ).detach()

                if src_feats.shape[0] >= self.dstl_num_pos:
                    choice = np.random.permutation(src_feats.shape[0])[
                        : self.dstl_num_pos
                    ]
                    src_feats = src_feats[choice]
                    tgt_feats = tgt_feats[choice]
                    corr_labels = corr_labels[choice]

                dstl_loss = self.nce_dstl_loss(
                    src_feats,
                    tgt_feats,
                    corr_labels,
                    results[
                        "meta_positive"
                    ].detach(),  # do not backpropagate to the teacher
                    results["meta_negative"].detach(),
                )

                dstl_loss = torch.nan_to_num(dstl_loss, nan=0.0, posinf=0.0, neginf=0.0)

                d_loss = d_loss + dstl_loss

            d_loss /= len(estimated_results)
            loss_dict["loss_dstl"] = d_loss
            loss += self.dstl_weight * d_loss

        if self.enable_mixed_density:
            m_loss = self.compose_mixed_density_loss(
                input_dict,
                soutput0,
                soutput1,
                T_est,
                uncollated_pairs,
                suppl_soutput0,
                suppl_soutput1,
                mixed_density_results,
            )
            loss_dict["loss_m"] = m_loss
            loss += float(self.mixed_density_loss_weight) * m_loss

        loss_dict["loss"] = loss

        return loss_dict

    def compose_loss(
        self,
        input_dict,
        soutput0,
        soutput1,
        T_est,
        pos_pairs,
        uncollated_pairs,
        estimated_results,
        suppl_soutput0=None,
        suppl_soutput1=None,
        mixed_density_results=None,
    ):
        result = self.compose_loss_simple(
            input_dict,
            soutput0,
            soutput1,
            T_est,
            pos_pairs,
            uncollated_pairs,
            estimated_results,
            suppl_soutput0,
            suppl_soutput1,
            mixed_density_results,
        )
        return result

    def match_and_filter_corr(
        self,
        C_batch_0,
        F_batch_0,
        C_batch_1,
        F_batch_1,
        radius=20,
        feature_filter="Lowe",
        spatial_filter="Spherical",
        frame_distance=None,
    ):
        """
            Match correspondences through k-nearest-neighbor, filter them using lowe's ratio, then apply spherical filtering
            which eliminates correspondences whose either end is too close to LiDAR
        Input:
            - C_batch_0:    [bs, ni, 3]
            - F_batch_0:    [bs, ni, d]
            - C_batch_1:    [bs, nj, 3]
            - F_batch_1:    [bs, nj, d]
                - note that ni and nj might change, while d is the fixed feature dimension
            - radius:       threshold to cut nearby correspondences around LiDARs if filter=='Spherical'
            - filter:       Specifies the filtering method, must be one in ["Spherical", "Similarity"]
            - frame_distance:   [bs], bears the frame interval of each input point cloud pair if filter=='Similarity'

        Output:
            - matches:      [N, 2], the collated correspondence indexes where N=sum(ni)
            - uncollated_matches: [bs, ni, d], the uncollated correspondence indexes
        """
        for i in range(len(C_batch_0)):
            C_batch_0[i] = C_batch_0[i].to(F_batch_0[0].device)
            C_batch_1[i] = C_batch_1[i].to(F_batch_0[0].device)

        num_corres = 5000
        P1 = pytorch3d.structures.Pointclouds(C_batch_0, features=F_batch_0)
        P2 = pytorch3d.structures.Pointclouds(C_batch_1, features=F_batch_1)

        P1_F = P1.features_padded()
        P2_F = P2.features_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()

        assert feature_filter in ["None", "Lowe"]
        assert spatial_filter in ["Spherical", "Similarity", "None"]

        K = 1 if feature_filter == "None" else 2

        dists_1, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=K)
        dists_2, idx_2, _ = knn_points(P2_F, P1_F, P2_N, P1_N, K=K)

        idx_1 = idx_1[:, :, 0:1]
        idx_2 = idx_2[:, :, 0:1]

        if feature_filter == "Lowe":
            cosine_1 = 1 - 0.5 * dists_1
            cosine_2 = 1 - 0.5 * dists_2

            weights_1 = self.calculate_ratio_test(cosine_1)
            weights_2 = self.calculate_ratio_test(cosine_2)
        else:
            weights_1 = dists_1[:, :, 0:1]
            weights_2 = dists_2[:, :, 0:1]

        # Get topK matches in both directions
        n_corres_1 = min(num_corres, P1_N.min())
        n_corres_2 = min(num_corres, P2_N.min())

        if n_corres_1 < num_corres or n_corres_2 < num_corres:
            pass
            # print(f"Min corresponds is {n_corres_1} and {n_corres_2}")

        m12_idx1, m12_idx2, m12_dist = self.get_topk_matches(
            weights_1, idx_1, n_corres_1
        )
        m21_idx2, m21_idx1, m21_dist = self.get_topk_matches(
            weights_2, idx_2, n_corres_2
        )
        # cosine_1 = cosine_1[:, :, 0:1].gather(1, m12_idx1)
        # cosine_2 = cosine_2[:, :, 0:1].gather(1, m21_idx2)

        # concatenate into correspondances. Shape: (B, ni, 1), (B, ni, 1)
        matches_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        matches_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)

        # recover the index biases for each batched point cloud pair
        length_1_padded = [0] + [len(feats) for feats in F_batch_0]
        length_2_padded = [0] + [len(feats) for feats in F_batch_1]
        bias_1 = torch.cumsum(torch.Tensor(length_1_padded[:-1]), 0)
        bias_2 = torch.cumsum(torch.Tensor(length_2_padded[:-1]), 0)

        # re-collate the dispatched matching-indices to shape (N, 2)
        matches_biased_1 = [match + bias for (match, bias) in zip(matches_idx1, bias_1)]
        matches_biased_2 = [match + bias for (match, bias) in zip(matches_idx2, bias_2)]
        match_1 = torch.cat(matches_biased_1, dim=0)
        match_2 = torch.cat(matches_biased_2, dim=0)

        matches = torch.cat([match_1, match_2], dim=1)
        # Spherical filtering that eliminates correspondences whose either end is too close to LiDAR.
        # Emperically, Lowe filter must precede spherical filter in order to get the best result.
        # Note that 'matches' is not spherically filtered, since it is not used in the current version.
        uncollated_matches = []
        for i in range(len(C_batch_0)):
            if spatial_filter == "None":
                mask = torch.ones_like(
                    torch.norm(C_batch_0[i][matches_idx1[i].squeeze(1)], dim=1)
                ).bool()
            if spatial_filter == "Spherical":
                mask_1 = (
                    torch.norm(C_batch_0[i][matches_idx1[i].squeeze(1)], dim=1) > radius
                )
                mask_2 = (
                    torch.norm(C_batch_1[i][matches_idx2[i].squeeze(1)], dim=1) > radius
                )
                mask = mask_1 & mask_2
            elif spatial_filter == "Similarity":
                d0 = torch.norm(C_batch_0[i][matches_idx1[i].squeeze(1)], dim=1)
                d1 = torch.norm(C_batch_1[i][matches_idx2[i].squeeze(1)], dim=1)

                # convert central distance to (min(d0,d1), |d1-d0|) coordinate for table lookup
                d1_tmp = torch.abs(d0 - d1)
                d0 = torch.min(torch.vstack([d0, d1]), dim=0).values
                d1 = d1_tmp

                # get the mask based on similarity filtering of filed similarity records
                if not hasattr(self, "dist_sim_map"):
                    dataset = self.config.pretraining_dataset
                    maps = np.load(
                        f"config/dist_sim_plot/{dataset}_distSimPlot.npz",
                        allow_pickle=True,
                    )["res"].tolist()
                    self.dist_sim_map = {}
                    for indice in range(6):
                        self.dist_sim_map[indice] = torch.tensor(maps[indice]).to(
                            d0.device
                        )

                # determine the coordinate with rounding in similarity lookup tables
                frame_index = min(max(0, frame_distance[i] // 5), 5)
                xlim, ylim = self.dist_sim_map[frame_index].shape
                frame_to_ygrid_size = {0: 1, 1: 1.5, 2: 2, 3: 2.5, 4: 2.5, 5: 2.5}
                gridsize = [5, frame_to_ygrid_size[frame_index]]
                d0 = (d0 / gridsize[0]).long()
                d1 = (d1 / gridsize[1]).long()

                d0[torch.where(d0 < 0)] = (
                    0  # round out-of-bound input to the limited map
                )
                d1[torch.where(d1 < 0)] = 0
                d0[torch.where(d0 >= ylim)] = ylim - 1
                d1[torch.where(d1 >= xlim)] = xlim - 1
                mask = (
                    self.dist_sim_map[frame_index][d1, d0]
                    > self.config.similarity_thresh
                )

            src_indices = matches_idx1[i][mask]
            tgt_indices = matches_idx2[i][mask]
            uncollated_matches.append(torch.cat((src_indices, tgt_indices), dim=1))

        return matches.detach().cpu(), uncollated_matches

    def match_corr(self, P1_F, P2_F, num_corres):
        dists_1, idx_1, _ = knn_points(P1_F, P2_F, K=1)
        dists_2, idx_2, _ = knn_points(P2_F, P1_F, K=1)

        idx_1 = idx_1[:, :, 0:1]
        idx_2 = idx_2[:, :, 0:1]

        weights_1 = dists_1[:, :, 0:1]
        weights_2 = dists_2[:, :, 0:1]

        # Get topK matches in both directions
        n_corres_1 = min(num_corres, P1_F.shape[1])
        n_corres_2 = min(num_corres, P2_F.shape[1])

        if n_corres_1 < num_corres or n_corres_2 < num_corres:
            pass
            # print(f"Min corresponds is {n_corres_1} and {n_corres_2}")

        m12_idx1, m12_idx2, m12_dist = self.get_topk_matches(
            weights_1, idx_1, n_corres_1
        )
        m21_idx2, m21_idx1, m21_dist = self.get_topk_matches(
            weights_2, idx_2, n_corres_2
        )
        # cosine_1 = cosine_1[:, :, 0:1].gather(1, m12_idx1)
        # cosine_2 = cosine_2[:, :, 0:1].gather(1, m21_idx2)

        # concatenate into correspondances. Shape: (B, ni, 1), (B, ni, 1)
        matches_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1)
        matches_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1)

        return matches_idx1, matches_idx2

    def iterative_corr_expand_registration(
        self,
        src_points,
        tgt_points,
        batch_teacher_feats_0,
        batch_teacher_feats_1,
        src_keypts_corr_indices,
        tgt_keypts_corr_indices,
        max_iters=100,
        iter_rre_thres=1e-2,
        iter_rte_thres=1e-2,
        suppl_sel_ratio=0.4,
        suppl_max_corr=5000,
        detach_meta_corr=True,
    ):
        """
        Perform iterative matching and filtering to obtain the final correspondences.
        src_points: torch.Tensor[N_s, 3]
        tgt_points: torch.Tensor[N_t, 3]
        batch_teacher_feats_0: torch.Tensor[N_s, 32]
        batch_teacher_feats_0: torch.Tensor[N_t, 32]
        src_keypts_corr_indices: torch.Tensor[C]
        tgt_keypts_corr_indices: torch.Tensor[C]
        """

        N_src = src_points.shape[0]
        N_tgt = tgt_points.shape[0]

        src_keypts_corr = src_points[src_keypts_corr_indices]
        tgt_keypts_corr = tgt_points[tgt_keypts_corr_indices]

        output_dict = {}
        result, fitness = self.matcher.SC2_PCR(
            src_keypts_corr[None, :, :], tgt_keypts_corr[None, :, :]
        )

        frag1_warp = apply_transform(src_keypts_corr, result[0].float())
        distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
        pred_labels = (distance < self.matcher.inlier_threshold).bool()

        src_keypts_corr = src_keypts_corr[pred_labels]
        tgt_keypts_corr = tgt_keypts_corr[pred_labels]

        src_keypts_feats = batch_teacher_feats_0[src_keypts_corr_indices]
        tgt_keypts_feats = batch_teacher_feats_1[tgt_keypts_corr_indices]

        pos_src_keypts_feats = src_keypts_feats[pred_labels]  # N,32
        pos_tgt_keypts_feats = tgt_keypts_feats[pred_labels]  # N,32

        pos_corr_feats = corr_feats_from_point_feats(
            pos_src_keypts_feats, pos_tgt_keypts_feats
        )

        meta_positive = F.normalize(pos_corr_feats.sum(dim=0), dim=0)  # 32

        neg_src_keypts_feats = src_keypts_feats[~pred_labels]
        neg_tgt_keypts_feats = tgt_keypts_feats[~pred_labels]

        neg_corr_feats = corr_feats_from_point_feats(
            neg_src_keypts_feats, neg_tgt_keypts_feats
        )

        meta_negative = F.normalize(neg_corr_feats.sum(dim=0), dim=0)  # 32

        if detach_meta_corr:
            meta_positive = meta_positive.detach()
            meta_negative = meta_negative.detach()

        neg_src_keypts_corr_indices = (
            src_keypts_corr_indices[~pred_labels].detach().cpu().numpy()
        )
        neg_tgt_keypts_corr_indices = (
            tgt_keypts_corr_indices[~pred_labels].detach().cpu().numpy()
        )

        src_keypts_corr_indices = (
            src_keypts_corr_indices[pred_labels].detach().cpu().numpy()
        )
        tgt_keypts_corr_indices = (
            tgt_keypts_corr_indices[pred_labels].detach().cpu().numpy()
        )

        max_corr_len = 0
        for it in range(max_iters):
            rest_src_indices = np.setdiff1d(
                np.arange(N_src),
                np.concatenate(
                    [
                        src_keypts_corr_indices,
                        neg_src_keypts_corr_indices,
                    ]
                ),
            )
            rest_tgt_indices = np.setdiff1d(
                np.arange(N_tgt),
                np.concatenate(
                    [
                        tgt_keypts_corr_indices,
                        neg_tgt_keypts_corr_indices,
                    ]
                ),
            )

            if rest_src_indices.shape[0] == 0 or rest_tgt_indices.shape[0] == 0:
                # We do not have anything left.
                break

            rest_src_pts = src_points[rest_src_indices]
            rest_tgt_pts = tgt_points[rest_tgt_indices]
            rest_src_feats = batch_teacher_feats_0[rest_src_indices]
            rest_tgt_feats = batch_teacher_feats_1[rest_tgt_indices]

            if rest_src_pts.shape[0] == 0 or rest_tgt_pts.shape[0] == 0:
                # We do not have anything left.
                break

            src_rest_keypts_corr_indices, tgt_rest_keypts_corr_indices = (
                self.match_corr(
                    rest_src_feats[None, :], rest_tgt_feats[None, :], suppl_max_corr
                )
            )

            # We hate batch dimensions
            src_rest_keypts_corr_indices = src_rest_keypts_corr_indices.squeeze(
                0
            ).squeeze(-1)
            tgt_rest_keypts_corr_indices = tgt_rest_keypts_corr_indices.squeeze(
                0
            ).squeeze(-1)

            rest_src_keypts_corr = rest_src_pts[src_rest_keypts_corr_indices]
            rest_tgt_keypts_corr = rest_tgt_pts[tgt_rest_keypts_corr_indices]
            rest_src_keypts_feats = rest_src_feats[src_rest_keypts_corr_indices]
            rest_tgt_keypts_feats = rest_tgt_feats[tgt_rest_keypts_corr_indices]

            rest_corr_feats = corr_feats_from_point_feats(
                rest_src_keypts_feats, rest_tgt_keypts_feats
            )

            rest_corr_scores = compound_similarity_metric(
                rest_corr_feats, meta_positive
            )

            sorted_indices = torch.argsort(rest_corr_scores, descending=True)

            sorted_indices = sorted_indices[
                : int(suppl_sel_ratio * len(sorted_indices))
            ]

            rest_src_keypts_corr_sel = rest_src_keypts_corr[sorted_indices]
            rest_tgt_keypts_corr_sel = rest_tgt_keypts_corr[sorted_indices]

            rest_src_keypts_corr_indices_sel = src_rest_keypts_corr_indices[
                sorted_indices
            ]
            rest_tgt_keypts_corr_indices_sel = tgt_rest_keypts_corr_indices[
                sorted_indices
            ]

            all_src_keypts_corr = torch.cat(
                [src_keypts_corr, rest_src_keypts_corr_sel], dim=0
            )
            all_tgt_keypts_corr = torch.cat(
                [tgt_keypts_corr, rest_tgt_keypts_corr_sel], dim=0
            )
            all_src_keypts_corr_indices = np.concatenate(
                [
                    src_keypts_corr_indices,
                    rest_src_indices[
                        rest_src_keypts_corr_indices_sel.detach().cpu().numpy()
                    ],
                ]
            )
            all_tgt_keypts_corr_indices = np.concatenate(
                [
                    tgt_keypts_corr_indices,
                    rest_tgt_indices[
                        rest_tgt_keypts_corr_indices_sel.detach().cpu().numpy()
                    ],
                ]
            )

            new_result, new_fitness = self.matcher.SC2_PCR(
                all_src_keypts_corr[None, :], all_tgt_keypts_corr[None, :]
            )

            frag1_warp = apply_transform(all_src_keypts_corr, new_result[0].float())
            distance = torch.sum((frag1_warp - all_tgt_keypts_corr) ** 2, dim=-1) ** 0.5
            pred_labels = (distance < self.matcher.inlier_threshold).bool()

            src_keypts_corr = all_src_keypts_corr[pred_labels]
            tgt_keypts_corr = all_tgt_keypts_corr[pred_labels]

            src_keypts_corr_indices = all_src_keypts_corr_indices[
                pred_labels.detach().cpu().numpy()
            ]
            tgt_keypts_corr_indices = all_tgt_keypts_corr_indices[
                pred_labels.detach().cpu().numpy()
            ]

            all_src_keypts_neg_corr_indices = all_src_keypts_corr_indices[
                ~pred_labels.detach().cpu().numpy()
            ]
            all_tgt_keypts_neg_corr_indices = all_tgt_keypts_corr_indices[
                ~pred_labels.detach().cpu().numpy()
            ]

            neg_src_keypts_corr_indices = np.concatenate(
                [
                    neg_src_keypts_corr_indices,
                    all_src_keypts_neg_corr_indices,
                ]
            )
            neg_tgt_keypts_corr_indices = np.concatenate(
                [
                    neg_tgt_keypts_corr_indices,
                    all_tgt_keypts_neg_corr_indices,
                ]
            )

            pos_src_keypts_feats = batch_teacher_feats_0[src_keypts_corr_indices]
            pos_tgt_keypts_feats = batch_teacher_feats_1[tgt_keypts_corr_indices]

            pos_corr_feats = corr_feats_from_point_feats(
                pos_src_keypts_feats, pos_tgt_keypts_feats
            )

            neg_src_keypts_feats = batch_teacher_feats_0[neg_src_keypts_corr_indices]
            neg_tgt_keypts_feats = batch_teacher_feats_0[neg_src_keypts_corr_indices]

            neg_corr_feats = corr_feats_from_point_feats(
                neg_src_keypts_feats, neg_tgt_keypts_feats
            )

            meta_positive = F.normalize(pos_corr_feats.sum(dim=0), dim=0)  # 32
            meta_negative = F.normalize(neg_corr_feats.sum(dim=0), dim=0)  # 32
            if detach_meta_corr:
                meta_positive = meta_positive.detach()
                meta_negative = meta_negative.detach()

            cur_rre, cur_rte = compute_registration_error(
                result.squeeze(0).detach().cpu().numpy(),
                new_result.squeeze(0).detach().cpu().numpy(),
            )
            result = new_result
            fitness = new_fitness

            if cur_rre < iter_rre_thres and cur_rte < iter_rte_thres:
                # print(f"Exit at {it} for it has rre {cur_rre} and rte {cur_rte}")
                break

            if max_corr_len >= src_keypts_corr_indices.shape[0]:
                # print(f"Exit at {it} for it reaches max len {max_corr_len}")
                break

            max_corr_len = src_keypts_corr_indices.shape[0]

        output_dict["T_est"] = result[0].float()
        output_dict["fitness"] = fitness
        output_dict["src_keypts_corr_indices"] = src_keypts_corr_indices
        output_dict["tgt_keypts_corr_indices"] = tgt_keypts_corr_indices
        output_dict["neg_src_keypts_corr_indices"] = neg_src_keypts_corr_indices
        output_dict["neg_tgt_keypts_corr_indices"] = neg_tgt_keypts_corr_indices

        if detach_meta_corr:
            output_dict["meta_positive"] = meta_positive.detach()
            output_dict["meta_negative"] = meta_negative.detach()
        else:
            output_dict["meta_positive"] = meta_positive
            output_dict["meta_negative"] = meta_negative

        return output_dict

    def mixed_density_corr(self, C_batch_0, C_batch_1, T_est, rediscovery_radius):
        P1 = pytorch3d.structures.Pointclouds(C_batch_0)
        P2 = pytorch3d.structures.Pointclouds(C_batch_1)

        P1_F = P1.points_padded()
        P2_F = P2.points_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()

        _, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=1)
        idx_1 = idx_1[:, :, 0]
        bias_1 = torch.cumsum(torch.Tensor([0] + P1_N.tolist()), 0).long()
        bias_2 = torch.cumsum(torch.Tensor([0] + P2_N.tolist()), 0).long()

        correspondences = []

        if self.mixed_density_loss == "circle":
            uncollated_corr = []

        for i, (p1_length, p2_length) in enumerate(zip(P1_N, P2_N)):
            pos_sel_1 = torch.randperm(p1_length)[: min(p1_length, 5000)].to(
                self.device
            )

            src_keypts_corr = C_batch_0[i][pos_sel_1.long()]
            tgt_keypts_corr = C_batch_1[i][idx_1[i][pos_sel_1].long()]
            within_range_mask = (
                torch.norm(src_keypts_corr - tgt_keypts_corr, dim=1)
                < rediscovery_radius
            )  # keep all correspondences within 2m error
            # the bound here is intentionally set loose due to tolerate possible pose errors

            # mask the out-of-range corrs
            pos_sel_1 = pos_sel_1[within_range_mask]
            pos_sel_2 = idx_1[i][pos_sel_1]

            if self.mixed_density_loss == "circle":
                uncollated_corr.append(
                    torch.cat([pos_sel_1.unsqueeze(1), pos_sel_2.unsqueeze(1)], dim=1)
                    .detach()
                    .cpu()
                )

            pos_sel_1 += bias_1[i]
            pos_sel_2 += bias_2[i]

            correspondences.append(
                torch.cat(
                    [
                        pos_sel_1.unsqueeze(1),
                        pos_sel_2.unsqueeze(1),
                    ],
                    dim=1,
                )
            )

        if self.mixed_density_loss == "circle":
            return uncollated_corr
        else:
            correspondences = torch.cat(correspondences, dim=0)
            correspondences = correspondences.detach().cpu()
            return correspondences

    def corr_through_registration(
        self,
        input_dict,
        batch_teacher_feats_0,
        batch_teacher_feats_1,
        uncollated_pairs,
        return_all_results=False,
        rediscovery_radius=2.0,
        include_mixed_density=False,
        force_rediscovery=True,
    ):
        C_batch_0, C_batch_1, fitnesses = [], [], []

        if self.enable_mixed_density and include_mixed_density:
            C_batch_suppl_0 = []
            C_batch_suppl_1 = []

        resgitration_results = []
        T_ransac = []
        for i in range(len(uncollated_pairs)):
            src_keypts_corr_indices = uncollated_pairs[i][:, 0]
            tgt_keypts_corr_indices = uncollated_pairs[i][:, 1]

            registration_dict = self.iterative_corr_expand_registration(
                input_dict["pcd0"][i],
                input_dict["pcd1"][i],
                batch_teacher_feats_0[i],
                batch_teacher_feats_1[i],
                src_keypts_corr_indices,
                tgt_keypts_corr_indices,
                max_iters=self.iterative_max_iter,
                iter_rre_thres=self.iterative_rre_thres,
                iter_rte_thres=self.iterative_rte_thres,
                suppl_sel_ratio=self.iterative_suppl_ratio,
                suppl_max_corr=self.iterative_suppl_max_corr,
                detach_meta_corr=True,
            )

            resgitration_results.append(registration_dict)
            fitnesses.append(registration_dict["fitness"])

            result = registration_dict["T_est"]
            T_ransac.append(result.float().detach().cpu().numpy())
            C_batch_0.append(
                (input_dict["pcd0"][i] @ result[:3, :3].T + result[:3, 3].T).to(
                    self.device
                )
            )
            C_batch_1.append(input_dict["pcd1"][i].to(self.device))

            if self.enable_mixed_density and include_mixed_density:
                C_batch_suppl_0.append(
                    (
                        input_dict["supplement_pcd0"][i].to(device=result.device)
                        @ result[:3, :3].T
                        + result[:3, 3].T
                    ).to(self.device)
                )
                C_batch_suppl_1.append(input_dict["supplement_pcd1"][i].to(self.device))

        P1 = pytorch3d.structures.Pointclouds(C_batch_0)
        P2 = pytorch3d.structures.Pointclouds(C_batch_1)

        P1_F = P1.points_padded()
        P2_F = P2.points_padded()
        P1_N = P1.num_points_per_cloud()
        P2_N = P2.num_points_per_cloud()

        _, idx_1, _ = knn_points(P1_F, P2_F, P1_N, P2_N, K=1)
        idx_1 = idx_1[:, :, 0]
        bias_1 = torch.cumsum(torch.Tensor([0] + P1_N.tolist()), 0).long()
        bias_2 = torch.cumsum(torch.Tensor([0] + P2_N.tolist()), 0).long()

        correspondences = []
        uncollated_corr = []
        for i, (p1_length, p2_length) in enumerate(zip(P1_N, P2_N)):
            reg_res = resgitration_results[i]
            if (
                not force_rediscovery
                and self.adaptive_rediscovery
                and "src_keypts_corr_indices" in reg_res
                and reg_res["src_keypts_corr_indices"].shape[0] > self.circle_max_points
            ):
                src_keypts_corr_indices = reg_res["src_keypts_corr_indices"]
                tgt_keypts_corr_indices = reg_res["tgt_keypts_corr_indices"]

                pos_sel_1 = torch.from_numpy(src_keypts_corr_indices).to(self.device)
                pos_sel_2 = torch.from_numpy(tgt_keypts_corr_indices).to(self.device)

                uncollated_corr.append(
                    torch.cat([pos_sel_1.unsqueeze(1), pos_sel_2.unsqueeze(1)], dim=1)
                )

                pos_sel_1 += bias_1[i]
                pos_sel_2 += bias_2[i]

                correspondences.append(
                    torch.cat(
                        [
                            pos_sel_1.unsqueeze(1),
                            pos_sel_2.unsqueeze(1),
                        ],
                        dim=1,
                    )
                )
            else:
                pos_sel_1 = torch.randperm(p1_length)[: min(p1_length, 5000)].to(
                    self.device
                )

                pose = torch.Tensor(T_ransac[i]).to(self.device)
                src_keypts_corr = (
                    input_dict["pcd0"][i][pos_sel_1.long()] @ pose[:3, :3].T
                    + pose[:3, 3].T
                )
                tgt_keypts_corr = input_dict["pcd1"][i][idx_1[i][pos_sel_1].long()]
                within_range_mask = (
                    torch.norm(src_keypts_corr - tgt_keypts_corr, dim=1)
                    < rediscovery_radius
                )  # keep all correspondences within 2m error
                # the bound here is intentionally set loose due to tolerate possible pose errors

                # mask the out-of-range corrs
                pos_sel_1 = pos_sel_1[within_range_mask]
                pos_sel_2 = idx_1[i][pos_sel_1]

                if (
                    not force_rediscovery
                    and self.adaptive_rediscovery
                    and "src_keypts_corr_indices" in reg_res
                ):
                    src_keypts_corr_indices = reg_res["src_keypts_corr_indices"]
                    tgt_keypts_corr_indices = reg_res["tgt_keypts_corr_indices"]

                    rediscover_append_count = min(
                        2 * (self.circle_max_points - src_keypts_corr_indices.shape[0]),
                        pos_sel_1.shape[0],
                    )

                    # randomly select rediscovered correspondences
                    choice = np.random.permutation(
                        int(src_keypts_corr_indices.shape[0])
                    )[:rediscover_append_count]
                    pos_sel_1 = pos_sel_1[choice]
                    pos_sel_2 = pos_sel_2[choice]

                    pos_sel_1 = torch.cat(
                        [
                            pos_sel_1,
                            torch.from_numpy(src_keypts_corr_indices).to(self.device),
                        ],
                        dim=0,
                    )
                    pos_sel_2 = torch.cat(
                        [
                            pos_sel_2,
                            torch.from_numpy(tgt_keypts_corr_indices).to(self.device),
                        ],
                        dim=0,
                    )

                    # randomly permute the correspondences
                    perm = np.random.permutation(pos_sel_1.shape[0])
                    pos_sel_1 = pos_sel_1[perm]
                    pos_sel_2 = pos_sel_2[perm]

                uncollated_corr.append(
                    torch.cat([pos_sel_1.unsqueeze(1), pos_sel_2.unsqueeze(1)], dim=1)
                )

                pos_sel_1 += bias_1[i]
                pos_sel_2 += bias_2[i]

                correspondences.append(
                    torch.cat(
                        [
                            pos_sel_1.unsqueeze(1),
                            pos_sel_2.unsqueeze(1),
                        ],
                        dim=1,
                    )
                )

        correspondences = torch.cat(correspondences, dim=0)

        mixed_density_result = {}

        if self.enable_mixed_density and include_mixed_density:
            search_radius = rediscovery_radius  # * (i + 2) ** 2

            mixed_density_result["s0c1"] = torch.zeros(1)
            mixed_density_result["c0s1"] = torch.zeros(1)

            mixed_density_result["s0s1"] = self.mixed_density_corr(
                C_batch_suppl_0,
                C_batch_suppl_1,
                T_ransac,
                search_radius,
            )

        return (
            T_ransac,
            correspondences,
            fitnesses,
            uncollated_corr,
            resgitration_results if return_all_results else None,
            mixed_density_result if include_mixed_density else None,
        )

    def labeler_tta(self, input_dict, sinput0, sinput1, encoded_0, encoded_1):
        assert self.use_sc2pcr, "SC2-PCR is required for labeler TTA."
        assert self.labeler.training, "Labeler must be in training mode."

        self.labeler_optimizer.zero_grad()

        _, batch_enc_feats_0 = encoded_0.decomposed_coordinates_and_features
        _, batch_enc_feats_1 = encoded_1.decomposed_coordinates_and_features

        # We use filtering in both metric space and euclidean space to create cleaner "correspondence labels".
        # This is the key to the labeler self-improving upon longer-distance point cloud pairs.
        pos_pair_tmp, uncollated_pairs = self.match_and_filter_corr(
            input_dict["pcd0"],
            batch_enc_feats_0,
            input_dict["pcd1"],
            batch_enc_feats_1,
            radius=self.config.filter_radius,
            feature_filter=self.config.feature_filter,
            spatial_filter=self.config.spatial_filter,
            frame_distance=input_dict["frame_distance"],
        )

        # The correspondences above are still noisy and not suitable for direct training of the student.
        # Instead, we apply SC2-PCR to select the inliers and perform a registration.
        # The estimated pose is highly likely correct and can be used to re-calculate nearest-neighbor correspondences.
        try:
            (
                T_est,
                pos_pairs,
                fitnesses,
                uncollated_pairs,
                estimated_results,
                _,  # we don't need the mixed density correspondences
            ) = self.corr_through_registration(
                input_dict,
                batch_enc_feats_0,
                batch_enc_feats_1,
                uncollated_pairs,
                return_all_results=True,
                rediscovery_radius=self.tta_rediscovery_radius,
                include_mixed_density=False,
                force_rediscovery=True,
            )
        except Exception as e:
            # raise e
            pos_pairs = pos_pair_tmp

        # point-level loss
        pos_pairs_pts = pos_pairs.cpu()
        pos_loss_pts, neg_loss_pts = self.contrastive_hardest_negative_loss(
            encoded_0.F, encoded_1.F, pos_pairs_pts
        )
        loss_pts = pos_loss_pts + self.neg_weight * neg_loss_pts

        # corr-level loss
        n_samples = int(self.tta_corr_weight // 4)
        temperature = float(self.tta_corr_temperature)

        loss_corr = 0.0
        for i in range(len(estimated_results)):
            src_keypts_corr_indices = estimated_results[i]["src_keypts_corr_indices"]
            tgt_keypts_corr_indices = estimated_results[i]["tgt_keypts_corr_indices"]

            if len(src_keypts_corr_indices) > n_samples:
                choice = np.random.permutation(len(src_keypts_corr_indices))[:n_samples]
                src_keypts_corr_indices = src_keypts_corr_indices[choice]
                tgt_keypts_corr_indices = tgt_keypts_corr_indices[choice]
            else:
                # resample to make the number of correspondences consistent
                choice = np.random.choice(
                    len(src_keypts_corr_indices),
                    n_samples - len(src_keypts_corr_indices),
                    replace=True,
                )
                src_keypts_corr_indices = np.concatenate(
                    [src_keypts_corr_indices, src_keypts_corr_indices[choice]]
                )
                tgt_keypts_corr_indices = np.concatenate(
                    [tgt_keypts_corr_indices, tgt_keypts_corr_indices[choice]]
                )

            neg_src_keypts_corr_indices = estimated_results[i][
                "neg_src_keypts_corr_indices"
            ]
            neg_tgt_keypts_corr_indices = estimated_results[i][
                "neg_tgt_keypts_corr_indices"
            ]

            if len(neg_src_keypts_corr_indices) > n_samples:
                choice = np.random.permutation(len(neg_src_keypts_corr_indices))[
                    :n_samples
                ]
                neg_src_keypts_corr_indices = neg_src_keypts_corr_indices[choice]
                neg_tgt_keypts_corr_indices = neg_tgt_keypts_corr_indices[choice]
            else:
                # resample to make the number of correspondences consistent
                choice = np.random.choice(
                    len(neg_src_keypts_corr_indices),
                    n_samples - len(neg_src_keypts_corr_indices),
                    replace=True,
                )
                neg_src_keypts_corr_indices = np.concatenate(
                    [neg_src_keypts_corr_indices, neg_src_keypts_corr_indices[choice]]
                )
                neg_tgt_keypts_corr_indices = np.concatenate(
                    [neg_tgt_keypts_corr_indices, neg_tgt_keypts_corr_indices[choice]]
                )

            pos_corr_feats = corr_feats_from_point_feats(
                batch_enc_feats_0[i][src_keypts_corr_indices],
                batch_enc_feats_1[i][tgt_keypts_corr_indices],
            )
            pos_corr_feats = F.normalize(pos_corr_feats, dim=1)

            neg_corr_feats = corr_feats_from_point_feats(
                batch_enc_feats_0[i][neg_src_keypts_corr_indices],
                batch_enc_feats_1[i][neg_tgt_keypts_corr_indices],
            )
            neg_corr_feats = F.normalize(neg_corr_feats, dim=1)

            Q = torch.stack([pos_corr_feats, neg_corr_feats])  # 2,N,d
            rand_perm = torch.randperm(Q.shape[1])
            K = Q[:, rand_perm, :]  # 2,N,d (shuffled)

            # logits: (2,2), every row is the similarity between pos and neg
            logits = torch.einsum("aij,bij->ab", Q, K) / n_samples
            labels = torch.arange(2).cuda().long()  # 2: pos and neg
            out = torch.div(logits, temperature)
            out = out.squeeze().contiguous()

            criterion = nn.CrossEntropyLoss().cuda()
            loss_corr = loss_corr + criterion(out, labels)

        loss_corr = loss_corr / len(estimated_results)

        loss = loss_pts + float(self.tta_corr_weight) * loss_corr

        loss.backward()

        self.labeler_optimizer.step()
        self.labeler_optimizer.zero_grad()

    def forward_student(
        self, input_dict, sinput0, sinput1, suppl_sinput0, suppl_sinput1
    ):
        soutput0 = self.model(sinput0)
        soutput1 = self.model(sinput1)

        if self.enable_mixed_density:
            suppl_soutput0 = self.model(suppl_sinput0)
            suppl_soutput1 = self.model(suppl_sinput1)
        else:
            suppl_soutput0 = None
            suppl_soutput1 = None

        return soutput0, soutput1, suppl_soutput0, suppl_soutput1

    def forward_labeler(self, input_dict, sinput0, sinput1):
        if self.enable_labeler_tta:
            with TTAModelParameterProtector(self.labeler, self.labeler_optimizer):
                self.labeler.train()
                encoded_0_pre = self.labeler(sinput0)
                encoded_1_pre = self.labeler(sinput1)

                self.labeler_tta(
                    input_dict, sinput0, sinput1, encoded_0_pre, encoded_1_pre
                )

                self.labeler.eval()
                self._prepare_labeler_bn()
                with torch.no_grad():
                    encoded_0 = self.labeler(sinput0)
                    encoded_1 = self.labeler(sinput1)

                    teacher_results = {
                        "encoded_0": encoded_0,
                        "encoded_1": encoded_1,
                    }

            with torch.no_grad():
                self.labeler.eval()
        else:
            self.labeler.eval()
            self._prepare_labeler_bn()
            with torch.no_grad():
                encoded_0 = self.labeler(sinput0)
                encoded_1 = self.labeler(sinput1)

                teacher_results = {
                    "encoded_0": encoded_0,
                    "encoded_1": encoded_1,
                }

        return teacher_results

    def _prepare_labeler_bn(self):
        return

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0

        # If the epoch reaches a bunch of predefined thresholds,
        # the dataset is extended to contain longer-distance pairs.
        curr_distance = self.data_loader.dataset.update_extension_distance(epoch)

        # save the intermediate models when we are at some important intermediate positions, for plotting only
        if curr_distance and curr_distance in [2, 6, 11, 16, 21, 26]:
            self._save_checkpoint(epoch - 1, filename=f"{curr_distance-1}_checkpoint")

        self.epoch = epoch
        base_mode_flag = self.data_loader.dataset.is_base_dataset()

        if not base_mode_flag or self.config.skip_initialization:
            if not self.labeler:
                # initialize the labeler to the current model (not that specified in LABELER_DIR config!)
                Model = load_model(self.config.model)
                self.labeler = Model(
                    1,
                    self.config.model_n_out,
                    bn_momentum=self.config.bn_momentum,
                    normalize_feature=self.config.normalize_feature,
                    conv1_kernel_size=self.config.conv1_kernel_size,
                    D=3,
                )
                self.labeler = self.labeler.to(self.device)
                self.labeler.load_state_dict(self.model.state_dict())

                labeler_weights = optional_config(self.config, "labeler_weight", None)
                if labeler_weights and osp.exists(labeler_weights):
                    print(f"Load labeler weights from {labeler_weights}")
                    self.labeler.load_state_dict(
                        torch.load(labeler_weights)["state_dict"]
                    )

                # labeler_checkpoint = torch.load('/home/xkz/EYOC/outputs/Experiments/KittiContinuousFramePairDataset-v0.3/ContinuousCorrExtensionTrainer/ResUNetBN2C/SGD-lr3e-1-e200-b8i1-modelnout32/2024-04-04_21-25-10/best_val_checkpoint.pth')
                # self.labeler.load_state_dict(labeler_checkpoint["state_dict"])
                self.num_updates = 1

                if self.enable_labeler_tta:
                    # create labeler's optimizer
                    if optional_config(self.config, "labeler_tta_nesterov", False):
                        assert self.config.labeler_tta_optimizer == "SGD"
                        self.labeler_optimizer = optim.SGD(
                            self.labeler.parameters(),
                            lr=self.config.labeler_tta_lr,
                            momentum=self.config.labeler_tta_momentum,
                            weight_decay=self.config.labeler_tta_weight_decay,
                            nesterov=True,
                        )
                    else:
                        self.labeler_optimizer = getattr(
                            optim, self.config.labeler_tta_optimizer
                        )(
                            self.labeler.parameters(),
                            lr=self.config.labeler_tta_lr,
                            momentum=self.config.labeler_tta_momentum,
                            weight_decay=self.config.labeler_tta_weight_decay,
                        )
                    print("Create Labeler TTA optimizer:", self.labeler_optimizer)

                self.model.load_state_dict(self.labeler.state_dict())
                self.labeler.eval()
            elif self.config.sync_strategy == "Sync":
                # fix the labeler at the beginning of an epoch (fully synchronized)
                with torch.no_grad():
                    self.labeler.load_state_dict(self.model.state_dict())
            elif self.config.sync_strategy == "EMA":
                decay = self.config.ema_decay

                if not hasattr(self, "num_updates"):
                    self.num_updates = 1

                debias = 1 - decay**self.num_updates
                with torch.no_grad():
                    for labeler_param, model_param in zip(
                        self.labeler.state_dict().values(),
                        self.model.state_dict().values(),
                    ):
                        if hasattr(model_param, "detach"):
                            labeler_param.copy_(
                                (
                                    decay * labeler_param
                                    + (1 - decay) * model_param.detach()
                                )
                                / debias
                            )
                        else:
                            labeler_param.copy_(
                                (decay * labeler_param + (1 - decay) * model_param)
                                / debias
                            )
                self.num_updates += 1
                print("Labeler updated with EMA.")
            elif self.config.sync_strategy == "None":
                pass
            else:
                raise NotImplementedError
        else:
            logging.info("Labeler is disabled: using base mode training.")

        self._prepare_labeler_bn()

        data_loader = self.data_loader
        data_loader.dataset.reset_seed(self.manual_seed + epoch)

        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size

        data_meter, data_timer, total_timer = AverageMeter(), Timer(), Timer()
        labeler_hit_meter = AverageMeter()
        labeler_rre_meter = AverageMeter()
        labeler_rte_meter = AverageMeter()

        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            try:
                self.optimizer.zero_grad()
                batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0
                batch_dstl_loss, batch_mixed_density_loss = 0, 0

                data_time = 0
                total_timer.tic()
                for iter_idx in range(iter_size):
                    torch.cuda.synchronize()
                    data_timer.tic()
                    input_dict = next(data_loader_iter)

                    torch.cuda.synchronize()

                    data_time += data_timer.toc(average=False)

                    F_in_0, C_in_0 = input_dict["sinput0_F"].to(
                        self.device
                    ), input_dict["sinput0_C"].to(self.device)
                    sinput0 = ME.SparseTensor(F_in_0, coordinates=C_in_0)

                    F_in_1, C_in_1 = input_dict["sinput1_F"].to(
                        self.device
                    ), input_dict["sinput1_C"].to(self.device)
                    sinput1 = ME.SparseTensor(F_in_1, coordinates=C_in_1)

                    if self.enable_mixed_density:
                        supplF_in_0, supplC_in_0 = input_dict["supplement0_F"].to(
                            self.device
                        ), input_dict["supplement0_C"].to(self.device)
                        suppl_sinput0 = ME.SparseTensor(
                            supplF_in_0, coordinates=supplC_in_0
                        )

                        supplF_in_1, supplC_in_1 = input_dict["supplement1_F"].to(
                            self.device
                        ), input_dict["supplement1_C"].to(self.device)
                        suppl_sinput1 = ME.SparseTensor(
                            supplF_in_1, coordinates=supplC_in_1
                        )
                    else:
                        suppl_sinput0 = None
                        suppl_sinput1 = None

                    soutput0, soutput1, suppl_soutput0, suppl_soutput1 = (
                        self.forward_student(
                            input_dict, sinput0, sinput1, suppl_sinput0, suppl_sinput1
                        )
                    )

                    labeler_results = self.forward_labeler(input_dict, sinput0, sinput1)
                    with torch.no_grad():
                        _, batch_enc_feats_0 = labeler_results[
                            "encoded_0"
                        ].decomposed_coordinates_and_features
                        _, batch_enc_feats_1 = labeler_results[
                            "encoded_1"
                        ].decomposed_coordinates_and_features

                        # We use filtering in both metric space and euclidean space to create cleaner "correspondence labels".
                        # This is the key to the labeler self-improving upon longer-distance point cloud pairs.
                        pos_pair_tmp, uncollated_pairs = self.match_and_filter_corr(
                            input_dict["pcd0"],
                            batch_enc_feats_0,
                            input_dict["pcd1"],
                            batch_enc_feats_1,
                            radius=self.config.filter_radius,
                            feature_filter=self.config.feature_filter,
                            spatial_filter=self.config.spatial_filter,
                            frame_distance=input_dict["frame_distance"],
                        )

                        # The correspondences above are still noisy and not suitable for direct training of the student.
                        # Instead, we apply SC2-PCR to select the inliers and perform a registration.
                        # The estimated pose is highly likely correct and can be used to re-calculate nearest-neighbor correspondences.
                        if self.config.use_sc2_filtering:
                            try:
                                (
                                    T_ransac,
                                    pos_pairs,
                                    fitnesses,
                                    uncollated_pairs,
                                    estimated_results,
                                    mixed_density_results,
                                ) = self.corr_through_registration(
                                    input_dict,
                                    batch_enc_feats_0,
                                    batch_enc_feats_1,
                                    uncollated_pairs,
                                    return_all_results=True,
                                    rediscovery_radius=self.rediscovery_radius,
                                    include_mixed_density=True,
                                    force_rediscovery=False,
                                )
                            except Exception as e:
                                print(
                                    e
                                )  # sometimes SC2-PCR will fail in various ways due to empty correspondence. We just catch and ignore them.
                                traceback.print_exc()

                                continue

                            pos_pairs = pos_pairs.cpu()
                        else:
                            pos_pairs = pos_pair_tmp

                        # obtain the inlier ratio for labeler during training
                        for batch_idx in range(len(input_dict["pcd0"])):
                            xyz0_indice = uncollated_pairs[batch_idx][:, 0]
                            xyz1_indice = uncollated_pairs[batch_idx][:, 1]
                            xyz0_corr = input_dict["pcd0"][batch_idx][xyz0_indice]
                            xyz1_corr = input_dict["pcd1"][batch_idx][xyz1_indice]
                            hit_ratio = self.evaluate_hit_ratio(
                                xyz0_corr,
                                xyz1_corr,
                                input_dict["T_gt"][batch_idx].to(self.device),
                                thresh=self.config.hit_ratio_thresh,
                            )
                            rre, rte = compute_registration_error(
                                input_dict["T_gt"][batch_idx].detach().cpu().numpy(),
                                T_ransac[batch_idx],
                            )

                            labeler_hit_meter.update(hit_ratio)
                            labeler_rre_meter.update(rre)
                            labeler_rte_meter.update(rte)

                    torch.cuda.synchronize()

                    loss_dict = self.compose_loss(
                        input_dict,
                        soutput0,
                        soutput1,
                        T_est=T_ransac,
                        pos_pairs=pos_pairs,
                        uncollated_pairs=uncollated_pairs,
                        estimated_results=estimated_results,
                        suppl_soutput0=suppl_soutput0,
                        suppl_soutput1=suppl_soutput1,
                        mixed_density_results=mixed_density_results,
                    )

                    loss = loss_dict["loss"]
                    loss.backward()

                    torch.cuda.synchronize()

                    batch_loss += loss_dict["loss"].item()
                    batch_pos_loss += loss_dict["pos_loss"].item()
                    batch_neg_loss += loss_dict["neg_loss"].item()

                    if "loss_dstl" in loss_dict:
                        batch_dstl_loss += loss_dict["loss_dstl"].item()

                    if "loss_m" in loss_dict:
                        batch_mixed_density_loss += loss_dict["loss_m"].item()

            except Exception as e:
                continue
                # raise e

            self.optimizer.step()
            gc.collect()

            if self.clean_cache:
                torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_freq == 0:
                self.writer.add_scalar("train/loss", batch_loss, start_iter + curr_iter)
                self.writer.add_scalar(
                    "train/pos_loss", batch_pos_loss, start_iter + curr_iter
                )
                self.writer.add_scalar(
                    "train/neg_loss", batch_neg_loss, start_iter + curr_iter
                )
                logging.info(
                    f"Train Epoch: {epoch} [{curr_iter}/{len(self.data_loader) // iter_size}], Current Loss: {batch_loss:.6f} Pos: {batch_pos_loss:.3f} Neg: {batch_neg_loss:.3f} Dstl: {batch_dstl_loss:.3f} Md: {batch_mixed_density_loss:.3f}"
                    + "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(
                        data_meter.avg,
                        total_timer.avg - data_meter.avg,
                        total_timer.avg,
                    )
                    + "\tLabeler IR: {:.3f}, Labeler RRE {:.3f}, Labeler RTE {:.3f}".format(
                        labeler_hit_meter.avg,
                        labeler_rre_meter.avg,
                        labeler_rte_meter.avg,
                    )
                )

                data_meter.reset()
                total_timer.reset()

    @torch.no_grad()
    def _valid_epoch(self):
        # Change the network to evaluation mode
        self.model.eval()

        # # If the epoch reaches a bunch of predefined thresholds,
        # # the dataset is extended to contain longer-distance pairs.
        # change_flag = self.val_data_loader.dataset.update_extension_distance(self.epoch)
        # # reset the evaluation metrics upon val dataloader update.
        # if change_flag:
        #     self.best_val_epoch = -np.inf
        #     self.best_val = -np.inf
        self.val_data_loader.dataset.reset_seed(self.manual_seed)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        data_timer, feat_timer, matching_timer = Timer(), Timer(), Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        for batch_idx in range(tot_num_data):
            try:
                data_timer.tic()
                # input_dict = data_loader_iter.next()
                input_dict = next(data_loader_iter)

                data_timer.toc()

                # pairs consist of (xyz1 index, xyz0 index)
                feat_timer.tic()
                sinput0 = ME.SparseTensor(
                    input_dict["sinput0_F"].to(self.device),
                    coordinates=input_dict["sinput0_C"].to(self.device),
                )
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(
                    input_dict["sinput1_F"].to(self.device),
                    coordinates=input_dict["sinput1_C"].to(self.device),
                )
                F1 = self.model(sinput1).F
                feat_timer.toc()

                matching_timer.tic()
                xyz0, xyz1, T_gt = (
                    input_dict["pcd0"][0],
                    input_dict["pcd1"][0],
                    input_dict["T_gt"][0],
                )
                xyz0_corr, xyz1_corr = self.find_corr(
                    xyz0, xyz1, F0, F1, subsample_size=5000
                )
                T_est = te.est_quad_linear_robust(xyz0_corr, xyz1_corr)

                loss = corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
                loss_meter.update(loss)

                rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
                rte_meter.update(rte)
                rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
                if not np.isnan(rre):
                    rre_meter.update(rre)

                hit_ratio = self.evaluate_hit_ratio(
                    xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh
                )
                hit_ratio_meter.update(hit_ratio)
                feat_match_ratio.update(hit_ratio > 0.05)
                matching_timer.toc()

                num_data += 1

                if self.clean_cache:
                    torch.cuda.empty_cache()

                if batch_idx % 100 == 0 and batch_idx > 0:
                    logging.info(
                        " ".join(
                            [
                                f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
                                f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
                                f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
                                f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
                            ]
                        )
                    )
                    data_timer.reset()
            except Exception as e:
                raise e

        logging.info(
            " ".join(
                [
                    f"Final Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
                    f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}",
                ]
            )
        )
        return {
            "loss": loss_meter.avg,
            "rre": rre_meter.avg,
            "rte": rte_meter.avg,
            "feat_match_ratio": feat_match_ratio.avg,
            "hit_ratio": hit_ratio_meter.avg,
        }
