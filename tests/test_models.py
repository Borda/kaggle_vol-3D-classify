import os

import pytest
from efficientnet_pytorch_3d import EfficientNet3D

from kaggle_brain3d.models import LitBrainMRI

_PATH_HERE = os.path.dirname(__file__)


@pytest.mark.parametrize("net", ["efficientnet-b0", EfficientNet3D.from_name("efficientnet-b0")])
def test_create_model(net):
    LitBrainMRI(net=net)
