import glob
import logging
import os
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import pandas as pd
import rising.transforms as rtr
import torch
from pytorch_lightning import LightningDataModule
from rising.loading import DataLoader
from rising.random import DiscreteParameter
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kaggle_volclassif.transforms import crop_volume, RandomAffine, rising_resize, rising_zero_mean
from kaggle_volclassif.utils import interpolate_volume, load_volume

SCAN_TYPES = ("FLAIR", "T1w", "T1CE", "T2w")
# Crop Dataset >> mean: 0.13732214272022247 STD: 0.24326834082603455
# Full Dataset >> mean: 0.09317479282617569 STD: 0.2139676809310913
rising_norm_intensity = partial(rising_zero_mean, mean=0.093, std=0.214)

# define transformations
TRAIN_TRANSFORMS = [
    rtr.Rot90((0, 1, 2), keys=["data"], p=0.5),
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["data"]),
    RandomAffine(scale_range=(0.9, 1.1), rotation_range=(-10, 10), translation_range=(-0.1, 0.1)),
    rising_norm_intensity,
]
VAL_TRANSFORMS = [
    rising_norm_intensity,
]


class BrainScansDataset(Dataset):

    def __init__(
        self,
        image_dir: str = 'train',
        df_table: Union[str, pd.DataFrame] = 'train_labels.csv',
        scan_types: Union[str, Sequence[str]] = ("FLAIR", "T2w"),
        cache_dir: Optional[str] = None,
        vol_size: Optional[Tuple[int, int, int]] = None,
        crop_thr: Optional[float] = 1e-6,
        mode: str = 'train',
        split: float = 0.8,
        in_memory: bool = False,
        random_state=42,
    ):
        self.image_dir = image_dir
        self.scan_types = (scan_types, ) if isinstance(scan_types, str) else scan_types
        self.cache_dir = cache_dir
        self.vol_size = vol_size
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
    def cached_image(rltv_path: str, cache_dir: str) -> str:
        return os.path.join(cache_dir or "", f"{rltv_path}.pt")

    @staticmethod
    def load_image(
        rltv_path: str,
        image_dir: str,
        cache_dir: Optional[str] = None,
        crop_thr: Optional[float] = None,
        vol_size: Optional[Tuple[int, int, int]] = None,
        overwrite: bool = False
    ) -> Tensor:
        vol_path = BrainScansDataset.cached_image(rltv_path, cache_dir)
        if os.path.isfile(vol_path) and not overwrite:
            try:
                return torch.load(vol_path).to(torch.float32)
            except (EOFError, RuntimeError):
                print(f"failed loading: {vol_path}")
        img_path = os.path.join(image_dir, rltv_path)
        assert os.path.isdir(img_path)
        img = load_volume(img_path)
        img = interpolate_volume(img, vol_size=vol_size)
        if crop_thr is not None:
            img = crop_volume(img, thr=crop_thr)
        if cache_dir:
            os.makedirs(os.path.dirname(vol_path), exist_ok=True)
            if not os.path.isfile(vol_path) or overwrite:
                torch.save(img.to(torch.float16), vol_path)
        return img

    def _load_image(self, rltv_path: str) -> Tensor:
        return BrainScansDataset.load_image(
            rltv_path,
            image_dir=self.image_dir,
            cache_dir=self.cache_dir,
            crop_thr=self.crop_thr,
            vol_size=self.vol_size,
        )

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


class BrainScansDM(LightningDataModule):

    def __init__(
        self,
        data_dir: str = '.',
        path_csv: str = 'train_labels.csv',
        cache_dir: str = '.',
        scan_types: Sequence[str] = ("FLAIR", "T2w"),
        vol_size: Union[None, int, Tuple[int, int, int]] = 64,
        crop_thr: Optional[float] = 1e-6,
        in_memory: bool = False,
        batch_size: int = 4,
        num_workers: Optional[int] = None,
        train_transforms=None,
        valid_transforms=None,
        split: float = 0.8,
        **kwargs_dataloader,
    ):
        super().__init__()
        # path configurations
        assert os.path.isdir(data_dir), f"missing folder: {data_dir}"
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.cache_dir = cache_dir
        self.vol_size = (vol_size, vol_size, vol_size) if isinstance(vol_size, int) else vol_size

        if not os.path.isfile(path_csv):
            path_csv = os.path.join(data_dir, path_csv)
        assert os.path.isfile(path_csv), f"missing table: {path_csv}"
        self.path_csv = path_csv

        # other configs
        self.scan_types = scan_types
        self.crop_thr = crop_thr
        self.batch_size = batch_size
        self.split = split
        self.in_memory = in_memory
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.kwargs_dataloader = kwargs_dataloader

        # need to be filled in setup()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_table = []
        self.test_dataset = None
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.image_mean = None
        self.image_std = None

    @property
    def ds_defaults(self) -> Dict[str, Any]:
        # some other configs
        return dict(
            scan_types=self.scan_types,
            vol_size=self.vol_size,
            crop_thr=self.crop_thr,
        )

    @property
    def dl_defaults(self) -> Dict[str, Any]:
        return dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sample_transforms=partial(rising_resize, size=self.vol_size),
        )

    @staticmethod
    def check_image_cache(
        rltv_path: str,
        image_dir: str,
        cache_dir: str,
        vol_size: Optional[Tuple[int, int, int]] = None,
        overwrite: bool = False,
        **kwargs_load,
    ) -> Optional[Tensor]:
        vol_path = BrainScansDataset.cached_image(rltv_path, cache_dir)
        if os.path.isfile(vol_path) and not overwrite:
            return None
        return BrainScansDataset.load_image(
            rltv_path, image_dir=image_dir, cache_dir=cache_dir, vol_size=vol_size, overwrite=overwrite, **kwargs_load
        )

    def prepare_data(self, num_proc: int = 0, dataset: Optional[BrainScansDataset] = None, overwrite: bool = False):
        if not self.cache_dir:
            return

        if not dataset:
            logging.info("using temporary dataset from all train table")
            dataset = BrainScansDataset(
                image_dir=self.train_dir,
                df_table=self.path_csv,
                split=1.0,
                in_memory=False,
                **self.ds_defaults,
            )
        # for im in ds.images:
        #     ds._load_image(im)

        if num_proc > 1:
            pool = Pool(processes=num_proc)
            mapping = pool.imap_unordered
        else:
            pool = None
            mapping = map

        imgs_mean, imgs_std = [], []
        pbar = tqdm(desc=f"preparing/caching scans @{num_proc} jobs", total=len(dataset))
        _cache_img = partial(
            BrainScansDM.check_image_cache,
            image_dir=dataset.image_dir,
            cache_dir=dataset.cache_dir,
            crop_thr=dataset.crop_thr,
            overwrite=overwrite,
        )
        for img in mapping(_cache_img, dataset.images):
            # ToDo: Otsu threshold and compute mean/STD only on brain
            if img is not None:
                imgs_mean.append(img.mean().item())
                imgs_std.append(img.std().item())
            pbar.update()

        if pool:
            pool.close()
            pool.join()

        self.image_mean = torch.mean(torch.tensor(imgs_mean)) if imgs_mean else None
        self.image_std = torch.mean(torch.tensor(imgs_std)) if imgs_std else None
        print(f"Dataset >> mean: {self.image_mean} STD: {self.image_std}")

    def setup(self, *_, **__) -> None:
        """Prepare datasets"""
        ds_training = dict(
            image_dir=self.train_dir,
            cache_dir=self.cache_dir,
            df_table=self.path_csv,
            split=self.split,
            in_memory=self.in_memory,
            **self.ds_defaults,
        )
        self.train_dataset = BrainScansDataset(**ds_training, mode='train')
        logging.info(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = BrainScansDataset(**ds_training, mode='valid')
        logging.info(f"validation dataset: {len(self.valid_dataset)}")

        if not os.path.isdir(self.test_dir):
            return
        ls_cases = [os.path.basename(p) for p in glob.glob(os.path.join(self.test_dir, '*'))]
        self.test_table = [dict(BraTS21ID=n, MGMT_value=None) for n in ls_cases]
        self.test_dataset = BrainScansDataset(
            image_dir=self.test_dir,
            df_table=pd.DataFrame(self.test_table),
            split=0,
            mode='test',
            **self.ds_defaults,
        )
        logging.info(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_transforms=self.train_transforms,
            **self.dl_defaults,
            **self.kwargs_dataloader,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_transforms=self.valid_transforms,
            **self.dl_defaults,
            **self.kwargs_dataloader,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_dataset:
            logging.warning('no testing images found')
            return
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_transforms=self.valid_transforms,
            **self.dl_defaults,
            **self.kwargs_dataloader,
        )
