import os
from typing import Union, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class SpineScansDataset(Dataset):

    def __init__(
        self,
        volume_dir: str = 'train_volumes',
        df_table: Union[str, pd.DataFrame] = 'train.csv',
        vol_size: Optional[Tuple[int, int, int]] = None,
        mode: str = 'train',
        split: float = 0.8,
        in_memory: bool = False,
        random_state=42,
    ):
        self.volume_dir = volume_dir
        self.vol_size = vol_size
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
        self.label_names = sorted([c for c in self.table.columns if c.startswith("C")])
        self.labels = self.table[self.label_names].values
        self.volumes = [os.path.join(volume_dir, f"{row['StudyInstanceUID']}.pt") for _, row in self.table.iterrows()]
        assert len(self.volumes) == len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        vol = self.volumes[idx]
        if isinstance(vol, str):
            try:
                vol = torch.load(vol).to(torch.float32)
            except (EOFError, RuntimeError):
                print(f"failed loading: {vol}")
        if self.in_memory:
            self.volumes[idx] = vol
        # in case of predictions, return image name as label
        label = label if label is not None else vol
        return {"data": vol.unsqueeze(0), "label": label}

    def __len__(self) -> int:
        return len(self.volumes)