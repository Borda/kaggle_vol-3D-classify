import os
import random
import shutil
from typing import Sequence, Union

import pandas as pd
from pydicom.data import get_testdata_file

from kaggle_brain3d.data import BrainScansDataset, BrainScansDM
from kaggle_brain3d.utils import load_volume


def _generate_sample_volume(path_folder: str, nb: int = 10):
    for i in range(nb):
        path_img = os.path.join(path_folder, "img-%d.dcm" % i)
        shutil.copy(get_testdata_file("CT_small.dcm"), path_img)


def _generate_synthetic_dataset(
    path_folder: str,
    phase: str = "train",
    nb_users: int = 10,
    scans: Union[str, Sequence[str]] = 'FLAIR',
    dim_z: int = 20
):
    random.seed(7)
    path_imgs = os.path.join(path_folder, phase)
    labels = []
    for i in range(nb_users):
        user = "%05d" % i
        path_user = os.path.join(path_imgs, user)
        scans = [scans] if isinstance(scans, str) else scans
        for n in scans:
            path_scan = os.path.join(path_user, n)
            os.makedirs(path_scan, exist_ok=True)
            _generate_sample_volume(path_scan, dim_z)
        labels.append({"BraTS21ID": user, "MGMT_value": random.randint(0, 1)})

    path_csv = os.path.join(path_folder, 'train_labels.csv')
    pd.DataFrame(labels).set_index("BraTS21ID").to_csv(path_csv)


def test_load_volume(tmpdir):
    _generate_sample_volume(tmpdir)
    vol = load_volume(tmpdir)
    assert tuple(vol.shape) == (128, 128, 10)


def test_dataset(tmpdir):
    _generate_synthetic_dataset(tmpdir, scans='FLAIR', nb_users=10)
    path_imgs = os.path.join(tmpdir, 'train')
    path_csv = os.path.join(tmpdir, 'train_labels.csv')

    dataset = BrainScansDataset(image_dir=path_imgs, df_table=pd.read_csv(path_csv), scan_types="FLAIR")
    assert len(dataset) == 8
    dataset = BrainScansDataset(image_dir=path_imgs, df_table=path_csv, scan_types="FLAIR", cache_dir=tmpdir, split=1.0)
    assert len(dataset) == 10
    dataset = BrainScansDataset(image_dir=path_imgs, df_table=path_csv, scan_types="FLAIR", cache_dir=tmpdir)
    assert len(dataset) == 8


def test_datamodule(tmpdir):
    _generate_synthetic_dataset(tmpdir, scans='FLAIR', nb_users=10)

    dm = BrainScansDM(data_dir=tmpdir, scan_types="FLAIR", batch_size=2, cache_dir=tmpdir)
    dm.prepare_data()
    dm.setup()
    assert len(dm.train_dataloader()) == 4
    assert len(dm.val_dataloader()) == 1
