import glob
import os
import re
from typing import Optional, Tuple

import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers import apply_voi_lut
from torch import Tensor


def parse_name_index(dcm_path) -> int:
    res = re.match(r".*-(\d+)\.dcm", dcm_path).groups()
    assert len(res) == 1
    return int(res[0])


def load_dicom(path_file: str) -> Optional[np.ndarray]:
    dicom = pydicom.dcmread(path_file)
    # TODO: adjust spacing in particular dimension according DICOM meta
    try:
        img = apply_voi_lut(dicom.pixel_array, dicom).astype(np.float32)
    except RuntimeError as err:
        print(err)
        return None
    return img


def load_volume(path_volume: str, percentile: Optional[int] = 0.01) -> Tensor:
    path_slices = glob.glob(os.path.join(path_volume, '*.dcm'))
    path_slices = sorted(path_slices, key=parse_name_index)
    vol = []
    for p_slice in path_slices:
        img = load_dicom(p_slice)
        if img is None:
            continue
        vol.append(img.T)
    volume = torch.tensor(vol, dtype=torch.float32)
    if percentile is not None:
        # get extreme values
        p_low = np.quantile(volume, percentile) if percentile else volume.min()
        p_high = np.quantile(volume, 1 - percentile) if percentile else volume.max()
        # normalize
        volume = (volume - p_low) / (p_high - p_low)
    return volume.T


def interpolate_volume(volume: Tensor) -> Tensor:
    vol_shape = volume.shape
    d_new = min(vol_shape[:2])
    # assert vol_shape[0] == vol_shape[1], f"mixed shape: {vol_shape}"
    if d_new == vol_shape[2]:
        return volume
    vol_size = (vol_shape[0], vol_shape[1], d_new)
    return F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=vol_size, mode="trilinear", align_corners=False)[0, 0]


def _tuple_int(t: Tensor) -> tuple:
    return tuple(t.numpy().astype(int))


def resize_volume(volume: Tensor, size: int = 128) -> Tensor:
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
    volume[offset[0]:offset[0] + shape_scale[0], offset[1]:offset[1] + shape_scale[1],
           offset[2]:offset[2] + shape_scale[2]] = vol_
    return volume


def find_dim_min(vec: list, thr: float) -> int:
    high = np.array(vec) >= thr
    return np.argmax(high)


def find_dim_max(vec: list, thr: float) -> int:
    high = np.array(vec) >= thr
    return len(high) - np.argmax(high[::-1])


def crop_volume(volume: Tensor, thr: float = 1e-6) -> Tensor:
    dims_x = torch.sum(torch.sum(volume, 1), -1) / np.prod(volume.shape)
    dims_y = torch.sum(torch.sum(volume, 0), -1) / np.prod(volume.shape)
    dims_z = torch.sum(torch.sum(volume, 0), 0) / np.prod(volume.shape)
    return volume[find_dim_min(dims_x, thr):find_dim_max(dims_x, thr),
                  find_dim_min(dims_y, thr):find_dim_max(dims_y, thr),
                  find_dim_min(dims_z, thr):find_dim_max(dims_z, thr)]


def show_volume_slice(axarr_, vol_slice, ax_name: str, v_min_max: tuple = (0., 1.)):
    axarr_[0].set_title(f"axis: {ax_name}")
    axarr_[0].imshow(vol_slice, cmap="gray", vmin=v_min_max[0], vmax=v_min_max[1])
    axarr_[1].plot(torch.sum(vol_slice, 1), list(range(vol_slice.shape[0]))[::-1])
    axarr_[1].plot(list(range(vol_slice.shape[1])), torch.sum(vol_slice, 0))
    axarr_[1].set_aspect('equal')
    axarr_[1].grid()


def idx_middle_if_none(volume: Tensor, *xyz: Optional[int]):
    xyz = list(xyz)
    vol_shape = volume.shape
    for i, d in enumerate(xyz):
        if d is None:
            xyz[i] = int(vol_shape[i] / 2)
        assert 0 <= xyz[i] < vol_shape[i]
    return xyz


def show_volume(
    volume: Tensor,
    x: Optional[int] = None,
    y: Optional[int] = None,
    z: Optional[int] = None,
    fig_size: Tuple[int, int] = (14, 9),
    v_min_max: tuple = (0., 1.),
):
    x, y, z = idx_middle_if_none(volume, x, y, z)
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=fig_size)
    print(f"share: {volume.shape}, x={x}, y={y}, z={y}  >> {volume.dtype}")
    show_volume_slice(axarr[:, 0], volume[x, :, :], "X", v_min_max)
    show_volume_slice(axarr[:, 1], volume[:, y, :], "Y", v_min_max)
    show_volume_slice(axarr[:, 2], volume[:, :, z], "Z", v_min_max)
    # plt.show(fig)
    return fig
