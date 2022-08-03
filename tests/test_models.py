import os

import pytest
from monai.networks.nets import EfficientNetBN, resnet18
from pytorch_lightning import seed_everything, Trainer

from kaggle_volclassif.data.brain import BrainScansDM
from kaggle_volclassif.models import LitBrainMRI, make_submission_brain
from tests.test_data import _generate_synthetic_dataset

_PATH_HERE = os.path.dirname(__file__)


@pytest.mark.parametrize("net", ["efficientnet-b0", EfficientNetBN("efficientnet-b0")])
def test_create_model(net):
    LitBrainMRI(net=net)


@pytest.mark.parametrize("prepare", [True, False])
def test_train_model(tmpdir, prepare):
    seed_everything(42)
    _generate_synthetic_dataset(tmpdir, phase="train", scans='FLAIR', nb_users=20)
    _generate_synthetic_dataset(tmpdir, phase="test", scans='FLAIR', nb_users=5)

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
    if prepare:
        dm.prepare_data()
    net = resnet18(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=1)
    model = LitBrainMRI(net=net)

    trainer = Trainer(max_epochs=2, gpus=0)
    trainer.fit(model, datamodule=dm)

    df_sub = make_submission_brain(model, dm.test_dataloader())
    assert len(df_sub) == 5
