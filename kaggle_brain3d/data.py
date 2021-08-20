import os
from typing import Optional, Sequence, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from kaggle_brain3d.utils import crop_volume, interpolate_volume, load_volume


class BrainScansDataset(Dataset):

    def __init__(
        self,
        image_dir: str = 'train',
        df_table: Union[str, pd.DataFrame] = 'train_labels.csv',
        scan_types: Sequence[str] = ("FLAIR", "T2w"),
        transforms=None,
        mode: str = 'train',
        split: float = 0.8,
        cache_dir: Optional[str] = None,
        crop_thr: float = 100,
        random_state=42,
    ):
        self.image_dir = image_dir
        self.scan_types = scan_types
        self.transforms = transforms
        self.mode = mode
        self.cache_dir = cache_dir
        self.crop_thr = crop_thr

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
            self.images += [os.path.join("%05d" % row["BraTS21ID"], tp) for tp in self.scan_types]
            self.labels += [row["MGMT_value"]] * len(self.scan_types)
        # filter existing
        self.images = [p for p in self.images if os.path.exists(os.path.join(self.image_dir, p))]
        assert len(self.images) == len(self.labels), f"missing some images as {len(self.images)} != {len(self.labels)}"

    def _load_image(self, rltv_path: str):
        if self.cache_dir:
            vol_path = os.path.join(self.cache_dir, f"{rltv_path}.pt")
            assert os.path.isfile(vol_path), f"missing cached: '{vol_path}'"
            return torch.load(vol_path)
        img_path = os.path.join(self.image_dir, rltv_path)
        assert os.path.isdir(img_path)
        img = load_volume(img_path)
        img = interpolate_volume(img)
        if self.crop_thr is not None:
            img = crop_volume(img, thr=self.crop_thr)
        if self.cache_dir:
            vol_path = os.path.join(self.cache_dir, f"{rltv_path}.pt")
            torch.save(img, vol_path)
        return img

    def __getitem__(self, idx: int) -> tuple:
        label = self.labels[idx]
        img_name = self.images[idx]
        img = self._load_image(img_name)

        # augmentation
        if self.transforms:
            raise NotImplementedError
        # in case of predictions, return image name as label
        label = label if label is not None else img_name
        return img, label

    def __len__(self) -> int:
        return len(self.images)
