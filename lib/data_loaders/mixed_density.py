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

from lib.data_loaders.base import KittiContinuousFramePairDataset, sample_random_trans


class KittiContinuousFrameMixedDensityGroupDataset(KittiContinuousFramePairDataset):
    def __init__(
        self,
        phase,
        transform=None,
        random_rotation=True,
        random_scale=True,
        manual_seed=False,
        config=None,
    ):
        super(KittiContinuousFrameMixedDensityGroupDataset, self).__init__(
            phase,
            transform,
            random_rotation,
            random_scale,
            manual_seed,
            config,
        )
        self.enable_mixed_density = config.enable_mixed_density and phase in ["train", "trainval"]
        

    def mixed_density_sample(self, xyz_0, xyz_1, xyz0_th, xyz1_th):
        voxel_size = float(self.voxel_size) * float(2)
        _, sel_0 = ME.utils.sparse_quantize(xyz_0 / voxel_size, return_index=True)
        _, sel_1 = ME.utils.sparse_quantize(xyz_1 / voxel_size, return_index=True)

        # apply voxelization
        xyz_0_th = xyz_0[sel_0]
        xyz_1_th = xyz_1[sel_1]
        del sel_0
        del sel_1

        coords_0 = torch.floor(xyz_0_th / voxel_size)
        coords_1 = torch.floor(xyz_1_th / voxel_size)
        feats_0 = torch.ones((len(coords_0), 1))
        feats_1 = torch.ones((len(coords_1), 1))

        if self.transform:
            coords_0, feats_0 = self.transform(coords_0, feats_0)
            coords_1, feats_1 = self.transform(coords_1, feats_1)


        supplement_dict = {
            "xyz_0_th": xyz_0_th.float(),
            "xyz_1_th": xyz_1_th.float(),
            "coords_0": coords_0.int(),
            "coords_1": coords_1.int(),
            "feats_0": feats_0.float(),
            "feats_1": feats_1.float(),
        }

        return supplement_dict

    def __getitem__(self, idx):
        # Note that preparation procedures with or without complement frames are very much different,
        # we might as well just throw them in an if-else case, for simplicity of tuning and debugging
        prepare_timer, icp_timer, rot_crop_timer = Timer(), Timer(), Timer()
        prepare_timer.tic()
        drive, t_0, t_1 = self.files[idx]
        positions = self.get_slam_odometry(drive, [t_0, t_1])
        pos_0, pos_1 = positions[0:2]

        # load two center point clouds
        xyz_0 = self._get_xyz(drive, t_0)
        xyz_1 = self._get_xyz(drive, t_1)
        prepare_timer.toc()

        icp_timer.tic()
        # determine icp result between t0 and t1
        # This item is not used during training. Only testing involves calculating RR using this label.
        M2 = np.linalg.inv(pos_1) @ pos_0
        icp_timer.toc()

        # add random rotation if needed, note that the same rotation is applied to both curr and nghb
        rot_crop_timer.tic()
        if self.random_rotation:
            T0 = sample_random_trans(xyz_0, self.randg, np.pi / 4)
            T1 = sample_random_trans(xyz_1, self.randg, np.pi / 4)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            xyz_0 = self.apply_transform(xyz_0, T0)
            xyz_1 = self.apply_transform(xyz_1, T1)
        else:
            trans = M2
        rot_crop_timer.toc()

        # random scaling
        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz_0 = scale * xyz_0
            xyz_1 = scale * xyz_1
            trans[:3, 3] = scale * trans[:3, 3]

        # voxelization
        xyz_0 = torch.from_numpy(xyz_0)
        xyz_1 = torch.from_numpy(xyz_1)

        # Make point clouds using voxelized points
        _, sel_0 = ME.utils.sparse_quantize(xyz_0 / self.voxel_size, return_index=True)
        _, sel_1 = ME.utils.sparse_quantize(xyz_1 / self.voxel_size, return_index=True)

        pcd_0 = make_open3d_point_cloud(xyz_0[sel_0])
        pcd_1 = make_open3d_point_cloud(xyz_1[sel_1])

        if (
            self.MAX_DIST <= 1
            and self.phase == "train"
            and not self.skip_initialization
        ):
            # Base mode training, we default an identity transformation.
            matches = get_matching_indices(
                pcd_0, pcd_1, np.identity(4), matching_search_voxel_size
            )
            # matches = self.get_matching_indices_pytorch3d(xyz_0[sel_0], xyz_1[sel_1], np.identity(4), matching_search_voxel_size)
        elif self.phase != "train" or self.supervised == True:
            # Extension mode val/test, GT pose is used to assess model performance.
            matches = get_matching_indices(
                pcd_0, pcd_1, trans, matching_search_voxel_size
            )
        else:
            # Extenstion mode training, fill in dummy value for compatibility
            matches = torch.zeros((1, 2))
        if len(matches) == 0:
            # if two PCs share no overlap, then randomly subtitute another pair
            # print(f"Matching indices small at {drive}, {t_0}, {t_1},len()={len(matches)}")
            return self.__getitem__(np.random.choice(self.__len__(), 1)[0])

        # apply voxelization
        xyz_0_th = xyz_0[sel_0]
        xyz_1_th = xyz_1[sel_1]
        del sel_0
        del sel_1

        coords_0 = torch.floor(xyz_0_th / self.voxel_size)
        coords_1 = torch.floor(xyz_1_th / self.voxel_size)
        feats_0 = torch.ones((len(coords_0), 1))
        feats_1 = torch.ones((len(coords_1), 1))

        if self.transform:
            coords_0, feats_0 = self.transform(coords_0, feats_0)
            coords_1, feats_1 = self.transform(coords_1, feats_1)

        if self.enable_mixed_density:
            supplement_dict = self.mixed_density_sample(
                xyz_0, xyz_1, xyz_0_th.float(), xyz_1_th.float()
            )
            return (
                xyz_0_th.float(),
                xyz_1_th.float(),
                coords_0.int(),
                coords_1.int(),
                feats_0.float(),
                feats_1.float(),
                matches,
                trans,
                t_1 - t_0,
                supplement_dict,
            )
        else:
            return (
                xyz_0_th.float(),
                xyz_1_th.float(),
                coords_0.int(),
                coords_1.int(),
                feats_0.float(),
                feats_1.float(),
                matches,
                trans,
                t_1 - t_0,
            )
    
