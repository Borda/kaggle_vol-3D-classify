import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _tuple_int(t: Tensor) -> tuple:
    return tuple(t.numpy().astype(int))


def resize_volume(volume: Tensor, size: int = 128) -> Tensor:
    """Resize volume with preservimg aspect ration and being centered

    >>> vol = torch.rand(64, 64, 48)
    >>> vol = resize_volume(vol, 32)
    >>> vol.shape
    torch.Size([32, 32, 32])
    """
    shape_old = torch.tensor(volume.shape)
    shape_new = torch.tensor([size] * 3)
    scale = torch.max(shape_old.to(float) / shape_new)
    shape_scale = shape_old / scale
    # print(f"{shape_old} >> {shape_scale} >> {shape_new}")
    vol_ = F.interpolate(
        volume.unsqueeze(0).unsqueeze(0), size=_tuple_int(shape_scale), mode="trilinear", align_corners=False
    )[0, 0]
    offset = _tuple_int((shape_new - shape_scale) / 2)
    volume = torch.zeros(*_tuple_int(shape_new), dtype=volume.dtype)
    shape_scale = _tuple_int(shape_scale)
    volume[
        offset[0] : offset[0] + shape_scale[0],
        offset[1] : offset[1] + shape_scale[1],
        offset[2] : offset[2] + shape_scale[2],
    ] = vol_
    return volume


def find_dim_min(vec: list, thr: float) -> int:
    high = np.array(vec) >= thr
    return np.argmax(high)


def find_dim_max(vec: list, thr: float) -> int:
    high = np.array(vec) >= thr
    return len(high) - np.argmax(high[::-1])


def crop_volume(volume: Tensor, thr: float = 1e-6) -> Tensor:
    """Crop volume froma ll sideds till cumlative val reach threshold.

    >>> rnd = torch.random.manual_seed(42)
    >>> vol = torch.rand(64, 64, 48, generator=rnd)
    >>> vol = crop_volume(vol, 32)
    >>> vol.shape
    torch.Size([64, 64, 48])
    """
    dims_x = torch.sum(torch.sum(volume, 1), -1) / np.prod(volume.shape)
    dims_y = torch.sum(torch.sum(volume, 0), -1) / np.prod(volume.shape)
    dims_z = torch.sum(torch.sum(volume, 0), 0) / np.prod(volume.shape)
    return volume[
        find_dim_min(dims_x, thr) : find_dim_max(dims_x, thr),
        find_dim_min(dims_y, thr) : find_dim_max(dims_y, thr),
        find_dim_min(dims_z, thr) : find_dim_max(dims_z, thr),
    ]
