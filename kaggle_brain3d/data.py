import glob
import logging
import os
from functools import partial
from multiprocessing import Pool
from typing import Optional, Sequence, Union

import pandas as pd
import rising.transforms as rtr
import torch
from pytorch_lightning import LightningDataModule
from rising.loading import DataLoader
from rising.random import DiscreteParameter
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kaggle_brain3d.transforms import crop_volume, RandomAffine, rising_resize, rising_zero_mean
from kaggle_brain3d.utils import interpolate_volume, load_volume

SCAN_TYPES = ("FLAIR", "T1w", "T1CE", "T2w")
# Dataset >> mean: 0.13732214272022247 STD: 0.24326834082603455
rising_norm_intensity = partial(rising_zero_mean, mean=0.137, std=0.243)

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
    def load_image(rltv_path: str, image_dir: str, cache_dir: str, crop_thr: float) -> Tensor:
        vol_path = os.path.join(cache_dir or "", f"{rltv_path}.pt")
        if os.path.isfile(vol_path):
            try:
                return torch.load(vol_path).to(torch.float32)
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
            torch.save(img.to(torch.float16), vol_path)
        return img

    def _load_image(self, rltv_path: str) -> Tensor:
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
        self.image_mean = None
        self.image_std = None

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

        imgs_mean, imgs_std = [], []
        pbar = tqdm(desc=f"preparing/caching scans @{num_proc} jobs", total=len(ds))
        _cache_img = partial(
            BrainScansDataset.load_image, image_dir=ds.image_dir, cache_dir=ds.cache_dir, crop_thr=ds.crop_thr
        )
        for img in mapping(_cache_img, ds.images):
            # ToDo: Otsu threshold and compute mean/STD only on brain
            imgs_mean.append(img.mean().item())
            imgs_std.append(img.std().item())
            pbar.update()

        if pool:
            pool.close()
            pool.join()

        self.image_mean = torch.mean(torch.tensor(imgs_mean))
        self.image_std = torch.mean(torch.tensor(imgs_std))
        print(f"Dataset >> mean: {self.image_mean} STD: {self.image_std}")

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
            sample_transforms=partial(rising_resize, size=self.input_size),
            batch_transforms=self.train_transforms,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sample_transforms=partial(rising_resize, size=self.input_size),
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
            sample_transforms=partial(rising_resize, size=self.input_size),
            batch_transforms=self.valid_transforms,
            pin_memory=torch.cuda.is_available(),
        )
