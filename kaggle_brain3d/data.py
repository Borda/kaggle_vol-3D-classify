import glob
import logging
import os
import random
from functools import partial
from multiprocessing import Pool
from typing import Optional, Sequence, Union

import pandas as pd
import rising.transforms as rtr
import torch
from pytorch_lightning import LightningDataModule
from rising.loading import DataLoader
from rising.random import DiscreteParameter
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kaggle_brain3d.utils import crop_volume, interpolate_volume, load_volume, resize_volume

SCAN_TYPES = ("FLAIR", "T1w", "T1CE", "T2w")
# define transformations
VAL_TRANSFORMS = [
    rtr.NormZeroMeanUnitStd(keys=["data"]),
]

TRAIN_TRANSFORMS = [
    rtr.NormZeroMeanUnitStd(keys=["data"]),
    rtr.Rot90((0, 1, 2), keys=["data"], p=0.5),
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["data"]),
    # rtr.Rotate(UniformParameter(0, 180), degree=True),
]


class BrainScansDataset(Dataset):

    def __init__(
        self,
        image_dir: str = 'train',
        df_table: Union[str, pd.DataFrame] = 'train_labels.csv',
        scan_types: Sequence[str] = ("FLAIR", "T2w"),
        cache_dir: Optional[str] = None,
        crop_thr: float = 1e-6,
        mode: str = 'train',
        split: float = 0.8,
        in_memory: bool = False,
        random_state=42,
    ):
        self.image_dir = image_dir
        self.scan_types = scan_types
        self.cache_dir = cache_dir
        self.crop_thr = crop_thr
        self.mode = mode
        self.in_memory = in_memory

        # set or load the config table
        if isinstance(df_table, pd.DataFrame):
            assert all(c in df_table.columns for c in ["BraTS21ID", "MGMT_value"])
            self.table = df_table
        elif isinstance(df_table, str):
            assert os.path.isfile(df_table), f"missing file: {df_table}"
            self.table = pd.read_csv(df_table)
        else:
            raise ValueError(f'unrecognised input for DataFrame/CSV: {df_table}')

        # shuffle data
        self.table = self.table.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # split dataset
        assert 0.0 <= split <= 1.0, f"split {split} is out of range"
        frac = int(split * len(self.table))
        self.table = self.table[:frac] if mode == 'train' else self.table[frac:]

        # populate images/labels
        self.images = []
        self.labels = []
        for _, row in self.table.iterrows():
            id_ = row["BraTS21ID"]
            name = id_ if isinstance(id_, str) else "%05d" % id_
            imgs = [os.path.join(name, tp) for tp in self.scan_types]
            imgs = [p for p in imgs if os.path.isdir(os.path.join(self.image_dir, p))]
            self.images += imgs
            self.labels += [row["MGMT_value"]] * len(imgs)
        assert len(self.images) == len(self.labels)

    @staticmethod
    def load_image(rltv_path: str, image_dir: str, cache_dir: str, crop_thr: float):
        vol_path = os.path.join(cache_dir or "", f"{rltv_path}.pt")
        if os.path.isfile(vol_path):
            try:
                return torch.load(vol_path)
            except (EOFError, RuntimeError):
                print(f"failed loading: {vol_path}")
        img_path = os.path.join(image_dir, rltv_path)
        assert os.path.isdir(img_path)
        img = load_volume(img_path)
        img = interpolate_volume(img)
        if crop_thr is not None:
            img = crop_volume(img, thr=crop_thr)
        if cache_dir:
            os.makedirs(os.path.dirname(vol_path), exist_ok=True)
            torch.save(img, vol_path)
        return img

    def _load_image(self, rltv_path: str):
        return BrainScansDataset.load_image(rltv_path, self.image_dir, self.cache_dir, self.crop_thr)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        img_ = self.images[idx]
        img = self._load_image(img_) if isinstance(img_, str) else img_
        if self.in_memory:
            self.images[idx] = img
        # in case of predictions, return image name as label
        label = label if label is not None else img_
        return {"data": img.unsqueeze(0), "label": label}

    def __len__(self) -> int:
        return len(self.images)


def rising_resize(size: int = 64, **batch):
    img = batch["data"]
    assert len(img.shape) == 4
    img_ = []
    for i in range(img.shape[0]):
        img_.append(resize_volume(img[i], size))
    batch.update({"data": torch.stack(img_, dim=0)})
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
        keys: Sequence = ('data', ),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = 'nearest',
        padding_mode: str = 'zeros',
        align_corners: bool = False,
        reverse_order: bool = False,
        **kwargs
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
            **kwargs
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


class BrainScansDM(LightningDataModule):

    def __init__(
        self,
        data_dir: str = '.',
        path_csv: str = 'train_labels.csv',
        cache_dir: str = '.',
        scan_types: Sequence[str] = ("FLAIR", "T2w"),
        crop_thr: float = 1e-6,
        in_memory: bool = False,
        input_size: int = 64,
        batch_size: int = 4,
        num_workers: int = None,
        train_transforms=None,
        valid_transforms=None,
        split: float = 0.8,
    ):
        super().__init__()
        # path configurations
        assert os.path.isdir(data_dir), f"missing folder: {data_dir}"
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.cache_dir = cache_dir

        if not os.path.isfile(path_csv):
            path_csv = os.path.join(data_dir, path_csv)
        assert os.path.isfile(path_csv), f"missing table: {path_csv}"
        self.path_csv = path_csv

        # other configs
        self.scan_types = scan_types
        self.crop_thr = crop_thr
        self.input_size = input_size
        self.batch_size = batch_size
        self.split = split
        self.in_memory = in_memory
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        # need to be filled in setup()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_table = []
        self.test_dataset = None
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    def prepare_data(self, num_proc: int = 0):
        if not self.cache_dir:
            return
        ds = BrainScansDataset(
            image_dir=self.train_dir,
            df_table=self.path_csv,
            scan_types=self.scan_types,
            split=1.0,
            cache_dir=self.cache_dir,
            crop_thr=self.crop_thr,
            in_memory=False,
        )
        # for im in ds.images:
        #     ds._load_image(im)

        if num_proc > 1:
            pool = Pool(processes=num_proc)
            mapping = pool.imap_unordered
        else:
            pool = None
            mapping = map

        pbar = tqdm(desc=f"preparing/caching scans @{num_proc} jobs", total=len(ds))
        _cache_img = partial(
            BrainScansDataset.load_image, image_dir=ds.image_dir, cache_dir=ds.cache_dir, crop_thr=ds.crop_thr
        )
        for _ in mapping(_cache_img, ds.images):
            pbar.update()

        if pool:
            pool.close()
            pool.join()

    def setup(self, *_, **__) -> None:
        """Prepare datasets"""
        ds_defaults = dict(
            image_dir=self.train_dir,
            df_table=self.path_csv,
            scan_types=self.scan_types,
            cache_dir=self.cache_dir,
            crop_thr=self.crop_thr,
            split=self.split,
            in_memory=self.in_memory,
        )
        self.train_dataset = BrainScansDataset(**ds_defaults, mode='train')
        logging.info(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = BrainScansDataset(**ds_defaults, mode='valid')
        logging.info(f"validation dataset: {len(self.valid_dataset)}")

        if not os.path.isdir(self.test_dir):
            return
        ls_cases = [os.path.basename(p) for p in glob.glob(os.path.join(self.test_dir, '*'))]
        self.test_table = [dict(BraTS21ID=n, MGMT_value=0.5) for n in ls_cases]
        self.test_dataset = BrainScansDataset(
            image_dir=self.test_dir,
            df_table=pd.DataFrame(self.test_table),
            scan_types=self.scan_types,
            cache_dir=self.cache_dir,
            crop_thr=self.crop_thr,
            split=0,
            mode='test'
        )
        logging.info(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            sample_transforms=partial(rising_resize, size=self.input_size),  # todo: resize to fix size
            batch_transforms=self.train_transforms,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sample_transforms=partial(rising_resize, size=self.input_size),  # todo: resize to fix size
            batch_transforms=self.valid_transforms,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_dataset:
            logging.warning('no testing images found')
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
            sample_transforms=partial(rising_resize, size=self.input_size),  # todo: resize to fix size
            batch_transforms=self.valid_transforms,
            pin_memory=torch.cuda.is_available(),
        )
