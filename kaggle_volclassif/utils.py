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


def load_volume(path_volume: str, percentile: Optional[float] = 0.01) -> Tensor:
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


def interpolate_volume(volume: Tensor, vol_size: Optional[Tuple[int, int, int]] = None) -> Tensor:
    """Interpolate volume in last (Z) dimension

    >>> vol = torch.rand(64, 64, 12)
    >>> vol2 = interpolate_volume(vol)
    >>> vol2.shape
    torch.Size([64, 64, 64])
    >>> vol2 = interpolate_volume(vol, vol_size=(64, 64, 24))
    >>> vol2.shape
    torch.Size([64, 64, 24])
    """
    vol_shape = tuple(volume.shape)
    if not vol_size:
        d_new = min(vol_shape[:2])
        vol_size = (vol_shape[0], vol_shape[1], d_new)
    # assert vol_shape[0] == vol_shape[1], f"mixed shape: {vol_shape}"
    if vol_shape == vol_size:
        return volume
    return F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=vol_size, mode="trilinear", align_corners=False)[0, 0]


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
    """Show volume in the three axis/cuts.

    >>> show_volume(torch.rand((64, 64, 64), dtype=torch.float32))
    shape: torch.Size([64, 64, 64]), x=32, y=32, z=32  >> torch.float32
    <Figure size 1400x900 with 6 Axes>
    """
    x, y, z = idx_middle_if_none(volume, x, y, z)
    fig, axarr = plt.subplots(nrows=2, ncols=3, figsize=fig_size)
    print(f"shape: {volume.shape}, x={x}, y={y}, z={y}  >> {volume.dtype}")
    show_volume_slice(axarr[:, 0], volume[x, :, :], "X", v_min_max)
    show_volume_slice(axarr[:, 1], volume[:, y, :], "Y", v_min_max)
    show_volume_slice(axarr[:, 2], volume[:, :, z], "Z", v_min_max)
    # plt.show(fig)
    return fig
