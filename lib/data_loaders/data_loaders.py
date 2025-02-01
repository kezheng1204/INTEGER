import logging
import random
import torch
import torch.utils.data
import numpy as np
import glob
import os
import os.path as osp
from scipy.linalg import expm, norm
import pathlib
import copy
from easydict import EasyDict as edict

from util.pointcloud import get_matching_indices, make_open3d_point_cloud
import lib.transforms as t
from lib.timer import Timer
from lib.partition import generate_pair
from lib.utils import get_transform_from_rotation_translation

import MinkowskiEngine as ME
import open3d as o3d

# import dask.dataframe as dd
from pytorch3d.ops.knn import knn_points
import pytorch3d

from lib.data_loaders.base import *
from lib.data_loaders.mixed_density import *


ALL_DATASETS = [
    ThreeDMatchPairDataset,
    KITTIPairDataset,
    KittiNFramePairDataset,
    KittiRandDistPairDataset,
    KittiContinuousFramePairDataset,
    KittiContinuousFrameMixedDensityGroupDataset,
]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def collate_pair_fn(list_data):
    (
        xyz0,
        xyz1,
        coords0,
        coords1,
        feats0,
        feats1,
        matching_inds,
        trans,
        frame_distances,
    ) = list(zip(*list_data))
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f"Can not convert to torch tensor, {x}")

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]
        if len(matching_inds[batch_id]) != 0:

            trans_batch.append(to_tensor(trans[batch_id]).float())

            matching_inds_batch.append(
                torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds)
            )
            len_batch.append([N0, N1])
        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    return {
        "pcd0": list(xyz0),
        "pcd1": list(xyz1),
        "sinput0_C": coords_batch0,
        "sinput0_F": feats_batch0.float(),
        "sinput1_C": coords_batch1,
        "sinput1_F": feats_batch1.float(),
        "correspondences": matching_inds_batch,
        "T_gt": trans_batch,
        "len_batch": len_batch,
        "frame_distance": frame_distances,
    }


def collate_mixed_density_pair_fn(list_data):
    (
        xyz0,
        xyz1,
        coords0,
        coords1,
        feats0,
        feats1,
        matching_inds,
        trans,
        frame_distances,
        supplement_dicts,
    ) = list(zip(*list_data))
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f"Can not convert to torch tensor, {x}")

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]
        if len(matching_inds[batch_id]) != 0:

            trans_batch.append(to_tensor(trans[batch_id]).float())

            matching_inds_batch.append(
                torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds)
            )
            len_batch.append([N0, N1])
        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    supplement_xyz0 = []
    supplement_xyz1 = []
    supplement_coords0 = []
    supplement_feats0 = []
    supplement_coords1 = []
    supplement_feats1 = []

    for supplement_dict in supplement_dicts:
        supplement_xyz0.append(supplement_dict["xyz_0_th"])
        supplement_xyz1.append(supplement_dict["xyz_1_th"])

        supplement_coords0.append(supplement_dict["coords_0"])
        supplement_feats0.append(supplement_dict["feats_0"])
        supplement_coords1.append(supplement_dict["coords_1"])
        supplement_feats1.append(supplement_dict["feats_1"])

    supplement_coords_batch0, supplement_feats_batch0 = ME.utils.sparse_collate(
        supplement_coords0, supplement_feats0
    )
    supplement_coords_batch1, supplement_feats_batch1 = ME.utils.sparse_collate(
        supplement_coords1, supplement_feats1
    )

    return {
        "pcd0": list(xyz0),
        "pcd1": list(xyz1),
        "supplement_pcd0": list(supplement_xyz0),
        "supplement_pcd1": list(supplement_xyz1),
        "sinput0_C": coords_batch0,
        "sinput0_F": feats_batch0.float(),
        "sinput1_C": coords_batch1,
        "sinput1_F": feats_batch1.float(),
        "correspondences": matching_inds_batch,
        "T_gt": trans_batch,
        "len_batch": len_batch,
        "frame_distance": frame_distances,
        "supplement0_C": supplement_coords_batch0,
        "supplement0_F": supplement_feats_batch0.float(),
        "supplement1_C": supplement_coords_batch1,
        "supplement1_F": supplement_feats_batch1.float(),
    }

def make_data_loader(config, phase, batch_size, num_threads=0, shuffle=None):
    assert phase in ["train", "trainval", "val", "test","testlo"]
    if shuffle is None:
        # shuffle = phase != "test"
        shuffle = "test" not in phase

    if config.dataset not in dataset_str_mapping.keys():
        logging.error(
            f"Dataset {config.dataset}, does not exists in "
            + ", ".join(dataset_str_mapping.keys())
        )

    print(f"Using dataset: {config.dataset}")
    Dataset = dataset_str_mapping[config.dataset]

    if phase not in ["train", "trainval"]:
        if (config.dataset == "KittiContinuousFrameMixedDensityGroupDataset"
            or config.dataset == "KittiContinuousFrameMixedDensityGroupFBRegularizedDataset"
        ):
            Dataset = dataset_str_mapping["KittiRandDistPairDataset"]

    use_random_scale = False
    use_random_rotation = False
    transforms = []
    if phase in ["train", "trainval"]:
        use_random_rotation = config.use_random_rotation
        use_random_scale = config.use_random_scale
        transforms += [t.Jitter()]

    dset = Dataset(
        phase,
        transform=t.Compose(transforms),
        random_scale=use_random_scale,
        random_rotation=use_random_rotation,
        manual_seed=True,
        config=config,
    )

    collate_function = collate_pair_fn

    if Dataset == KittiContinuousFrameMixedDensityGroupDataset and phase in [
        "train",
        "trainval",
    ]:
        collate_function = collate_mixed_density_pair_fn

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=collate_function,
        pin_memory=False,
        drop_last=True,
    )

    return loader
