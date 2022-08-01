import logging
import os
from typing import Any, Optional, Sequence, Tuple, Type, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN, ResNet, resnet18
from pytorch_lightning import Callback, LightningModule
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC, F1
from tqdm.auto import tqdm


def create_pretrained_medical_resnet(
    pretrained_path: str,
    model_constructor: callable = resnet18,
    spatial_dims: int = 3,
    n_input_channels: int = 1,
    num_classes: int = 1,
    **kwargs_monai_resnet: Any
) -> Tuple[ResNet, Sequence[str]]:
    """This si specific constructor for MONAI ResNet module loading MedicalNEt weights.

    See:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """
    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
    logging.debug(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
    logging.debug(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
    logging.debug(f"unused pretrained: {len(unused)}")
    assert len(inside) > len(missing)
    assert len(inside) > len(unused)

    pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net.load_state_dict(pretrain['state_dict'], strict=False)
    return net, inside


# class FineTuneCB(BaseFinetuning):
#     def __init__(self, unfreeze_at_epoch=10):
#         self._unfreeze_at_epoch = unfreeze_at_epoch
#
#     def freeze_before_training(self, pl_module):
#         pass
#
#     def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
#         # When `current_epoch` is 10, feature_extractor will start training.
#         if current_epoch == self._unfreeze_at_epoch:
#             self.unfreeze_and_add_param_group(
#                 modules=pl_module.net,
#                 optimizer=optimizer,
#                 train_bn=True,
#             )


class FineTuneCB(Callback):
    # add callback to freeze/unfreeze trained layers
    def __init__(self, unfreeze_epoch: int) -> None:
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        for n, param in pl_module.net.named_parameters():
            param.requires_grad = True
        optimizers, _ = pl_module.configure_optimizers()
        trainer.optimizers = optimizers


class LitBrainMRI(LightningModule):

    def __init__(
        self,
        net: Union[nn.Module, str] = "efficientnet-b0",
        pretrained_params: Optional[Sequence[str]] = None,
        lr: float = 1e-3,
        optimizer: Optional[Type[Optimizer]] = None,
    ):
        super().__init__()
        if isinstance(net, str):
            self.name = net
            net = EfficientNetBN(net, spatial_dims=3, in_channels=1, num_classes=1)
        else:
            self.name = net.__class__.__name__
        self.net = net
        self.pretrained_params = set(pretrained_params) if pretrained_params else set()
        for n, param in self.net.named_parameters():
            param.requires_grad = bool(n not in self.pretrained_params)
        self.learning_rate = lr
        self.optimizer = optimizer or AdamW

        self.train_auroc = AUROC(num_classes=1, compute_on_step=False)
        self.train_acc = Accuracy(num_classes=1)
        self.train_f1_score = F1()
        self.val_auroc = AUROC(num_classes=1, compute_on_step=False)
        self.val_acc = Accuracy(num_classes=1)
        self.val_f1_score = F1()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x)[:, 0])

    @staticmethod
    def compute_loss(y_hat: Tensor, y: Tensor):
        return F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("train/loss", loss, prog_bar=False)
        self.log("train/acc", self.train_acc(y_hat, y), prog_bar=False)
        self.log("train/f1", self.train_f1_score(y_hat, y), prog_bar=True)
        self.train_auroc.update(y_hat, y)
        try:  # ToDo: use balanced sampler
            self.log('train/auroc', self.train_auroc, on_step=False, on_epoch=True)
        except ValueError:
            pass
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("valid/loss", loss, prog_bar=False)
        self.log("valid/acc", self.val_acc(y_hat, y), prog_bar=True)
        self.log("valid/f1", self.val_f1_score(y_hat, y), prog_bar=True)
        self.val_auroc.update(y_hat, y)
        try:  # ToDo: use balanced sampler
            self.log('valid/auroc', self.val_auroc, on_step=False, on_epoch=True)
        except ValueError:
            pass

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]


def make_submission(model: LightningModule, dataloader: DataLoader, device: str = "cpu") -> pd.DataFrame:
    model.eval()
    model.to(device)
    submission = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        imgs = batch.get("data")
        with torch.no_grad():
            preds = model(imgs.to(device))
        submission += [
            dict(BraTS21ID5=n.split(os.path.sep)[0], MRI_type=n.split(os.path.sep)[-1], MGMT_value=p.item())
            for n, p in zip(batch.get("label"), preds)
        ]

    df_submission = pd.DataFrame(submission)
    df_submission["BraTS21ID"] = df_submission["BraTS21ID5"].apply(int)
    df_submission.set_index("BraTS21ID", inplace=True)
    return df_submission
