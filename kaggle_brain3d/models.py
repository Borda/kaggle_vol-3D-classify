from typing import Union

import torch
import torch.nn.functional as F
from efficientnet_pytorch_3d import EfficientNet3D
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import Accuracy, F1, Precision


class LitBrainMRI(LightningModule):

    def __init__(
        self,
        net: Union[nn.Module, str] = "efficientnet-b0",
        lr: float = 1e-4,
    ):
        super().__init__()
        if isinstance(net, str):
            self.name = net
            net = EfficientNet3D.from_name(net, override_params={'num_classes': 2}, in_channels=1)
        else:
            self.name = net.__class__.__name__
        self.net = net
        self.learning_rate = lr

        self.train_accuracy = Accuracy()
        self.train_precision = Precision()
        self.train_f1_score = F1()
        self.val_accuracy = Accuracy()
        self.val_precision = Precision()
        self.val_f1_score = F1()

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(self.net(x))

    def compute_loss(self, y_hat: Tensor, y: Tensor):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", self.train_accuracy(y_hat, y), prog_bar=False)
        self.log("train_prec", self.train_precision(y_hat, y), prog_bar=False)
        self.log("train_f1", self.train_f1_score(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("valid_loss", loss, prog_bar=False)
        self.log("valid_acc", self.val_accuracy(y_hat, y), prog_bar=True)
        self.log("valid_prec", self.val_precision(y_hat, y), prog_bar=True)
        self.log("valid_f1", self.val_f1_score(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]
