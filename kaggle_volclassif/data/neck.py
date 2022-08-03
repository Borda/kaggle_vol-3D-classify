import glob
import logging
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from rising.loading import DataLoader
from torch.utils.data import Dataset

from kaggle_volclassif.transforms import rising_resize


class SpineScansDataset(Dataset):

    def __init__(
        self,
        volume_dir: str = 'train_volumes',
        df_table: Union[str, pd.DataFrame] = 'train.csv',
        mode: str = 'train',
        split: float = 0.8,
        in_memory: bool = False,
        random_state=42,
    ):
        self.volume_dir = volume_dir
        self.mode = mode
        self.in_memory = in_memory

        # set or load the config table
        if isinstance(df_table, pd.DataFrame):
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
        self.label_names = sorted(c for c in self.table.columns if c.startswith("C"))
        self.labels = self.table[self.label_names].values if self.label_names else [None] * len(self.table)
        self.volumes = [os.path.join(volume_dir, f"{row['StudyInstanceUID']}.pt") for _, row in self.table.iterrows()]
        assert len(self.volumes) == len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        vol_ = self.volumes[idx]
        if isinstance(vol_, str):
            try:
                vol = torch.load(vol_).to(torch.float32)
            except (EOFError, RuntimeError):
                print(f"failed loading: {vol_}")
        else:
            vol = vol_
        if self.in_memory:
            self.volumes[idx] = vol
        # in case of predictions, return image name as label
        label = label if label is not None else vol_
        return {"data": vol.unsqueeze(0), "label": label}

    def __len__(self) -> int:
        return len(self.volumes)


class SpineScansDM(LightningDataModule):

    def __init__(
        self,
        data_dir: str = '.',
        path_csv: str = 'train.csv',
        vol_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
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
        self.train_dir = os.path.join(data_dir, 'train_volumes')
        self.test_dir = os.path.join(data_dir, 'test_volumes')

        if not os.path.isfile(path_csv):
            path_csv = os.path.join(data_dir, path_csv)
        assert os.path.isfile(path_csv), f"missing table: {path_csv}"
        self.path_csv = path_csv

        # other configs
        self.vol_size = vol_size
        self.batch_size = batch_size
        self.split = split
        self.in_memory = in_memory
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.kwargs_dataloader = kwargs_dataloader

        # need to be filled in setup()
        self.test_table = []
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self._label_names = {}
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    @property
    def dl_defaults(self) -> Dict[str, Any]:
        return dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sample_transforms=partial(rising_resize, size=self.vol_size),
        )

    @property
    def num_labels(self) -> int:
        return len(self._label_name)

    def setup(self, *_, **__) -> None:
        """Prepare datasets"""
        ds_training = dict(
            volume_dir=self.train_dir,
            df_table=self.path_csv,
            split=self.split,
            in_memory=self.in_memory,
        )
        self.train_dataset = SpineScansDataset(**ds_training, mode='train')
        logging.info(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = SpineScansDataset(**ds_training, mode='valid')
        logging.info(f"validation dataset: {len(self.valid_dataset)}")
        self._label_name = sorted(set(self.train_dataset.label_names + self.valid_dataset.label_names))

        if not os.path.isdir(self.test_dir):
            logging.warning(f"Missing test folder: {self.test_dir}")
            return
        ls_cases = [os.path.basename(p) for p in glob.glob(os.path.join(self.test_dir, '*'))]
        self.test_table = [dict(StudyInstanceUID=os.path.splitext(n)[0]) for n in ls_cases]
        self.test_dataset = SpineScansDataset(
            self.test_dir,
            df_table=pd.DataFrame(self.test_table),
            split=0,
            mode='test',
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
            logging.warning('no testing data found')
            return
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_transforms=self.valid_transforms,
            **self.dl_defaults,
            **self.kwargs_dataloader,
        )
