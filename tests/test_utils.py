import os
import shutil

from pydicom.data import get_testdata_file

from kaggle_volclassif.utils import load_dicom, load_volume_brain, load_volume_neck, norm_image


def _generate_sample_volume_brain(path_folder: str, nb: int = 10):
    for i in range(nb):
        path_img = os.path.join(path_folder, "img-%d.dcm" % i)
        shutil.copy(get_testdata_file("CT_small.dcm"), path_img)


def _generate_sample_volume_neck(path_folder: str, nb: int = 10):
    for i in range(nb):
        path_img = os.path.join(path_folder, f"{i + 1}.dcm")
        shutil.copy(get_testdata_file("CT_small.dcm"), path_img)


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
