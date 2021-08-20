import glob
import os
import re
from typing import Optional, Tuple

import numpy as np
import pydicom
import torch
from matplotlib import pyplot as plt
from pydicom.pixel_data_handlers import apply_voi_lut
from torch import Tensor


def parse_index(dcm_path) -> int:
    res = re.match(r".*-(\d+)\.dcm", dcm_path).groups()
    assert len(res) == 1
    return int(res[0])


def load_dicom(path_file: str) -> Optional[np.ndarray]:
    dicom = pydicom.dcmread(path_file)
    try:
        img = apply_voi_lut(dicom.pixel_array, dicom)
    except RuntimeError as err:
        print(err)
        return None
    return img


def load_volume(path_volume: str, percentile: Optional[int] = 0.01) -> Tensor:
    path_slices = glob.glob(os.path.join(path_volume, '*.dcm'))
    path_slices = sorted(path_slices, key=parse_index)
    vol = []
    for p_slice in path_slices:
        img = load_dicom(p_slice)
        if img is None:
            continue
        vol.append(img)
    volume = torch.tensor(vol)
    if percentile is not None:
        # get extreme values
        p_low = np.quantile(volume, percentile) if percentile else volume.min()
        p_high = np.quantile(volume, 1 - percentile) if percentile else volume.max()
        # normalize
        volume = (volume.to(float) - p_low) / (p_high - p_low)
    return volume


def interpolate_volume(volume: Tensor) -> Tensor:
    vol_shape = volume.shape
    assert vol_shape[1] == vol_shape[2]
    if vol_shape[0] == vol_shape[1]:
        return volume
    d0 = vol_shape[0] - 1
    d1 = vol_shape[1]
    vol_ = Tensor(d1, d1, d1)
    step = float(d0) / d1
    for i, pt in enumerate([i * step for i in range(d1)]):
        i_0, i_1 = int(pt), int(np.ceil(pt))
        if i_0 == i_1:
            vol_[i, :, :] = volume[i_0, :, :]
        else:
            vol_[i, :, :] = (i_1 - pt) * volume[i_0, :, :] + (pt - i_0) * volume[i_1, :, :]
    return vol_


def find_dim_min(vec: list, thr: float) -> int:
    high = np.array(vec) >= thr
    return np.argmax(high)


def find_dim_max(vec: list, thr: float) -> int:
    high = np.array(vec) >= thr
    return len(high) - np.argmax(high[::-1])


def crop_volume(volume: Tensor, thr: float = 100) -> Tensor:
    dims_x = torch.sum(torch.sum(volume, 1), -1)
    dims_y = torch.sum(torch.sum(volume, 0), -1)
    dims_z = torch.sum(torch.sum(volume, 0), 0)
    return volume[find_dim_min(dims_x, thr):find_dim_max(dims_x, thr),
                  find_dim_min(dims_y, thr):find_dim_max(dims_y, thr),
                  find_dim_min(dims_z, thr):find_dim_max(dims_z, thr)]


def show_volume_slice(axarr_, vol_slice, ax_name: str):
    axarr_[0].set_title(f"axis: {ax_name}")
    axarr_[0].imshow(vol_slice, cmap="gray", vmin=0, vmax=1)
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
    fig_size: Tuple[int, int] = (14, 9)
):
    x, y, z = idx_middle_if_none(volume, x, y, z)
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=fig_size)
    print(f"share: {volume.shape}, x={x}, y={y}, z={y}")
    show_volume_slice(axarr[:, 0], volume[x, :, :], "X")
    show_volume_slice(axarr[:, 1], volume[:, y, :], "Y")
    show_volume_slice(axarr[:, 2], volume[:, :, z], "Z")
    # plt.show(fig)
    return fig
