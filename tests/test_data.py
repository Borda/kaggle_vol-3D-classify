import os
import random
import shutil
from typing import Sequence, Union

import pandas as pd
from pydicom.data import get_testdata_file

from kaggle_volclassif.data import BrainScansDataset, BrainScansDM
from kaggle_volclassif.utils import load_dicom, load_volume_brain, load_volume_neck, norm_image


def _generate_sample_volume_brain(path_folder: str, nb: int = 10):
    for i in range(nb):
        path_img = os.path.join(path_folder, "img-%d.dcm" % i)
        shutil.copy(get_testdata_file("CT_small.dcm"), path_img)


def _generate_sample_volume_neck(path_folder: str, nb: int = 10):
    for i in range(nb):
        path_img = os.path.join(path_folder, f"{i + 1}.dcm")
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
            _generate_sample_volume_brain(path_scan, dim_z)
        labels.append({"BraTS21ID": user, "MGMT_value": random.randint(0, 1)})

    path_csv = os.path.join(path_folder, 'train_labels.csv')
    pd.DataFrame(labels).set_index("BraTS21ID").to_csv(path_csv)


def test_load_image(tmpdir):
    path_img = os.path.join(tmpdir, "img-sample.dcm")
    shutil.copy(get_testdata_file("CT_small.dcm"), path_img)
    img = load_dicom(path_img)
    assert tuple(img.shape) == (128, 128)
    img = norm_image(img)
    assert tuple(img.shape) == (128, 128)


def test_load_volume_brain(tmpdir):
    _generate_sample_volume_brain(tmpdir)
    vol = load_volume_brain(tmpdir)
    assert tuple(vol.shape) == (128, 128, 10)


def test_load_volume_neck(tmpdir):
    _generate_sample_volume_neck(tmpdir)
    vol = load_volume_neck(tmpdir, size=(64, 64, 32))
    assert tuple(vol.shape) == (64, 64, 32)


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
