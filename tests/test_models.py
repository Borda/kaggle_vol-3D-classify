import os

import pytest
from monai.networks.nets import EfficientNetBN, resnet18
from pytorch_lightning import Trainer

from kaggle_brain3d.data import BrainScansDM
from kaggle_brain3d.models import LitBrainMRI
from tests.test_data import _generate_synthetic_dataset

_PATH_HERE = os.path.dirname(__file__)


@pytest.mark.parametrize("net", ["efficientnet-b0", EfficientNetBN("efficientnet-b0")])
def test_create_model(net):
    LitBrainMRI(net=net)


def test_train_model(tmpdir):
    _generate_synthetic_dataset(tmpdir, scans='FLAIR', nb_users=30)

    dm = BrainScansDM(
        data_dir=tmpdir,
        scan_types="FLAIR",
        batch_size=2,
        cache_dir=tmpdir,
        crop_thr=None,
        split=0.6,
        # train_transforms=rtr.Compose(TRAIN_TRANSFORMS, transform_call=default_transform_call),
        # valid_transforms=rtr.Compose(VAL_TRANSFORMS, transform_call=default_transform_call),
    )
    dm.prepare_data()
    net = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=2)
    model = LitBrainMRI(net=net)

    trainer = Trainer(max_epochs=1, gpus=0)
    trainer.fit(model, datamodule=dm)
