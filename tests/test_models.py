import os

import pytest
from monai.networks.nets import EfficientNetBN

from kaggle_brain3d.models import LitBrainMRI

_PATH_HERE = os.path.dirname(__file__)


@pytest.mark.parametrize("net", ["efficientnet-b0", EfficientNetBN("efficientnet-b0")])
def test_create_model(net):
    LitBrainMRI(net=net)
