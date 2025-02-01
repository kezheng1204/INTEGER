import numpy as np
import open3d
import random
import torch
import pickle
import yaml

from typing import Tuple, List, Optional, Union, Any

from typing import Callable
from collections import OrderedDict


def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def get_transform_from_rotation_translation(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def load_config(path, return_original=False):
    """
    Loads config file:

    Args:
        path (str): path to the config file

    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            if k in config:
                print("Key is conflicted in configuration.")
            config[k] = v

    if return_original:
        return config, cfg

    return config


def optional_config(config, key, defval=None):
    if hasattr(config, key):
        return getattr(config, key)
    else:
        return defval


def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x
