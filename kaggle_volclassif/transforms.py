import random
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import rising.transforms as rtr
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


def rising_resize(size: Union[str, Tuple] = 64, **batch) -> Dict[str, Any]:
    """Augmentation.

    >>> batch = {"data": torch.rand(2, 64, 64, 12)}
    >>> batch2 = rising_resize(32, **batch)
    >>> batch2["data"].shape
    torch.Size([2, 32, 32, 32])
    >>> batch2 = rising_resize((32, 32, 16), **batch)
    >>> batch2["data"].shape
    torch.Size([2, 32, 32, 16])
    """
    data = batch["data"]
    assert len(data.shape) == 4
    if isinstance(size, int):
        size = (size, size, size)
    imgs = [
        F.interpolate(img.unsqueeze(0).unsqueeze(0), size=size, mode="trilinear", align_corners=False)[0, 0]
        for img in data
    ]
    batch.update({"data": torch.stack(imgs, dim=0)})
    return batch


def rising_zero_mean(mean: float = 0.5, std: float = 0.2, **batch) -> Dict[str, Any]:
    img = batch["data"]
    batch.update({"data": (img - mean) / std})
    return batch


class RandomAffine(rtr.BaseAffine):
    """Base Affine with random parameters for scale, rotation and translation
    taken from this notebooks: https://github.com/PhoenixDL/rising/blob/master/notebooks/lightning_segmentation.ipynb
    """

    def __init__(
        self,
        scale_range: tuple,
        rotation_range: tuple,
        translation_range: tuple,
        degree: bool = True,
        image_transform: bool = True,
        keys: Sequence = ("data",),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = "nearest",
        padding_mode: str = "zeros",
        align_corners: bool = False,
        reverse_order: bool = False,
        **kwargs,
    ):
        super().__init__(
            scale=None,
            rotation=None,
            translation=None,
            degree=degree,
            image_transform=image_transform,
            keys=keys,
            grad=grad,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            reverse_order=reverse_order,
            **kwargs,
        )

        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def assemble_matrix(self, **data) -> torch.Tensor:
        ndim = data[self.keys[0]].ndim - 2

        if self.scale_range is not None:
            self.scale = [random.uniform(*self.scale_range) for _ in range(ndim)]

        if self.translation_range is not None:
            self.translation = [random.uniform(*self.translation_range) for _ in range(ndim)]

        if self.rotation_range is not None:
            if ndim == 3:
                self.rotation = [random.uniform(*self.rotation_range) for _ in range(ndim)]
            elif ndim == 1:
                self.rotation = random.uniform(*self.rotation_range)

        return super().assemble_matrix(**data)
