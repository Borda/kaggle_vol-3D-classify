import os
import shutil

from pydicom.data import get_testdata_file

from kaggle_brain3d.utils import load_volume


def _generate_sample_volume(path_folder: str, nb: int = 10):
    for i in range(nb):
        path_img = os.path.join(path_folder, "img-%d.dcm" % i)
        shutil.copy(get_testdata_file("CT_small.dcm"), path_img)


def test_load_volume(tmpdir):
    _generate_sample_volume(tmpdir)
    vol = load_volume(tmpdir)
    assert tuple(vol.shape) == (128, 128, 10)
