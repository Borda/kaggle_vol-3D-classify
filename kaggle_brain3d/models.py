from typing import Optional, Union

import torch.nn.functional as F
from monai.networks.nets import EfficientNetBN
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, F1, Precision


class LitBrainMRI(LightningModule):

    def __init__(
        self,
        net: Union[nn.Module, str] = "efficientnet-b0",
        lr: float = 1e-4,
        optimizer: Optional[Optimizer] = None,
    ):
        super().__init__()
        if isinstance(net, str):
            self.name = net
            net = EfficientNetBN(net, spatial_dims=3, in_channels=1, num_classes=2)
        else:
            self.name = net.__class__.__name__
        self.net = net
        self.learning_rate = lr
        self.optimizer = optimizer or Adam(self.net.parameters(), lr=self.learning_rate)

        self.train_accuracy = Accuracy()
        self.train_precision = Precision()
        self.train_f1_score = F1()
        self.val_accuracy = Accuracy()
        self.val_precision = Precision()
        self.val_f1_score = F1()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def compute_loss(self, y_hat: Tensor, y: Tensor):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        y_hat = F.softmax(y_hat)
        self.log("train_acc", self.train_accuracy(y_hat, y), prog_bar=False)
        self.log("train_prec", self.train_precision(y_hat, y), prog_bar=False)
        self.log("train_f1", self.train_f1_score(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("valid_loss", loss, prog_bar=False)
        y_hat = F.softmax(y_hat)
        self.log("valid_acc", self.val_accuracy(y_hat, y), prog_bar=True)
        self.log("valid_prec", self.val_precision(y_hat, y), prog_bar=True)
        self.log("valid_f1", self.val_f1_score(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        scheduler = CosineAnnealingLR(self.optimizer, self.trainer.max_epochs, 0)
        return [self.optimizer], [scheduler]
