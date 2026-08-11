# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Brain Tumor Classification with PyTorch⚡Lightning & ResNet 3D
#
# The goal of this challenge is to Predict the status of a genetic biomarker important for brain cancer treatment.
#
# All the code is refered from public repository: https://github.com/Borda/kaggle_vol-3D-classify
# Any nice contribution is welcome!

# %%
# ! pip uninstall -y kaggle_volclassif
# ! pip install -q https://github.com/Borda/kaggle_vol-3D-classify/archive/refs/heads/main.zip
# # ! pip install -q https://github.com/Borda/kaggle_vol-3D-classify/archive/refs/tags/v0.3.2.zip
# ! pip install -q "pytorch-lightning>=1.3.8"
# ! pip uninstall -q -y wandb
# # !pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# ! pip list | grep torch

# %%
# ! ls -l /home/jirka/Datasets/rsna-miccai-brain-tumor
# ! nvidia-smi -L
# ! mkdir /home/jirka/TEMP/brain-tumor

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

import kaggle_volclassif

print(kaggle_volclassif.__version__)

# %% [markdown]
# ## Data exploration
#
# These 3 cohorts are structured as follows: Each independent case has a dedicated folder identified by a five-digit number.
# Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in DICOM format.
# The exact mpMRI scans included are:
#
# - **FLAIR**: Fluid Attenuated Inversion Recovery
# - **T1w**: T1-weighted pre-contrast
# - **T1Gd**: T1-weighted post-contrast
# - **T2w**: T2-weighted
#
# #### according to https://www.aapm.org/meetings/amos2/pdf/34-8205-79886-720.pdf
#
# - T1: weighting better deliniates anatomy
# - T2: weighting naturally shows pathology
#
# #### according to https://radiopaedia.org/articles/fluid-attenuated-inversion-recovery
#
# Fluid attenuated inversion recovery (FLAIR) is a special inversion recovery sequence with a long inversion time. This removes signal from the cerebrospinal fluid in the resulting images 1. Brain tissue on FLAIR images appears similar to T2 weighted images with grey matter brighter than white matter but CSF is dark instead of bright.
#
# To null the signal from fluid, the inversion time (TI) of the FLAIR pulse sequence is adjusted such that at equilibrium there is no net transverse magnetization of fluid.
#
# The FLAIR sequence is part of almost all protocols for imaging the brain, particularly useful in the detection of subtle changes at the periphery of the hemispheres and in the periventricular region close to CSF.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATH_DATASET = "/home/jirka/Datasets/rsna-miccai-brain-tumor"
PATH_MODELS = "/home/jirka/Workspace/pretrained_medical"
PATH_TEMP = "/home/jirka/TEMP/brain-tumor"
SCAN_TYPES = ("FLAIR", "T1w", "T1CE", "T2w")

df_train = pd.read_csv(os.path.join(PATH_DATASET, "train_labels.csv"))
df_train["BraTS21ID"] = df_train["BraTS21ID"].apply(lambda i: f"{i:05d}")
display(df_train.head())

# %% [markdown]
# See the dataset label distribution

# %%
_ = df_train["MGMT_value"].value_counts().plot(kind="pie", title="label distribution", autopct="%.1f%%")

# %% [markdown]
# For almost all scans we have all four types

# %%
scans = [os.path.basename(p) for p in glob.glob(os.path.join(PATH_DATASET, "train", "*", "*"))]
_ = pd.Series(scans).value_counts().plot(kind="bar", grid=True)

# %% [markdown]
# ### Interactive view
#
# showing particular scan in XYZ dimension/slices

# %%
from ipywidgets import IntSlider, interact

from kaggle_volclassif.transforms import crop_volume
from kaggle_volclassif.utils import interpolate_volume, load_volume, show_volume


def interactive_show(volume_path: str, crop_thr: float):
    print(f"loading: {volume_path}")
    volume = load_volume(volume_path, percentile=0)
    print(f"sample shape: {volume.shape} >> {volume.dtype}")
    volume = interpolate_volume(volume)
    print(f"interp shape: {volume.shape} >> {volume.dtype}")
    volume = crop_volume(volume, crop_thr)
    print(f"crop shape: {volume.shape} >> {volume.dtype}")
    vol_shape = volume.shape
    interact(
        lambda x, y, z: plt.show(show_volume(volume, x, y, z)),
        x=IntSlider(min=0, max=vol_shape[0], step=5, value=int(vol_shape[0] / 2)),
        y=IntSlider(min=0, max=vol_shape[1], step=5, value=int(vol_shape[1] / 2)),
        z=IntSlider(min=0, max=vol_shape[2], step=5, value=int(vol_shape[2] / 2)),
    )


PATH_SAMPLE_VOLUME = os.path.join(PATH_DATASET, "train", "00005", "FLAIR")

interactive_show(PATH_SAMPLE_VOLUME, crop_thr=1e-6)

# %% [markdown]
# ## Prepare dataset
#
# ### Pytorch Dataset
#
# The basic building block is traforming raw data to Torch Dataset.
# We have here loading particular DICOM images into a volume and saving as temp/cacher, so we do not need to take the very time demanding loading do next time - this boost the IO from about 2h to 8min
#
# At the end we show a few sample images from prepared dataset.

# %%
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from kaggle_volclassif.data import BrainScansDataset
from kaggle_volclassif.transforms import resize_volume

# ==============================

ds = BrainScansDataset(
    image_dir=os.path.join(PATH_DATASET, "train"),
    df_table=os.path.join(PATH_DATASET, "train_labels.csv"),
    crop_thr=None,
    cache_dir=PATH_TEMP,
)
for i in tqdm(range(2)):
    img = ds[i * 10]["data"]
    img = resize_volume(img[0])
    show_volume(img, fig_size=(9, 6))

# %% [markdown]
# ### Lightning DataModule
#
# It is constric to wrap all data-related peaces and define Pytoch dataloder for Training / Validation / Testing phase.
#
# At the end we show a few sample images from the fost training batch.

# %%
from functools import partial

import rising.transforms as rtr
from rising.loading import DataLoader, default_transform_call
from rising.random import DiscreteParameter, UniformParameter

from kaggle_volclassif.data import TRAIN_TRANSFORMS, VAL_TRANSFORMS, BrainScansDM
from kaggle_volclassif.transforms import RandomAffine, rising_zero_mean

# ==============================

dm = BrainScansDM(
    data_dir=PATH_DATASET,
    scan_types=["T2w"],
    # input_size=224,  # deprecated in v0.3
    vol_size=224,
    crop_thr=None,
    # crop_thr=1e-6,  # experimental crop threshold
    batch_size=12,  # for full model training
    # batch_size=6,  # for finetune head
    # cache_dir=None,
    cache_dir=PATH_TEMP,
    in_memory=True,
    num_workers=64,
    split=0.9,
    train_transforms=rtr.Compose(TRAIN_TRANSFORMS, transform_call=default_transform_call),
    valid_transforms=rtr.Compose(VAL_TRANSFORMS, transform_call=default_transform_call),
)
dm.prepare_data(num_proc=0)
dm.setup()
# dm.prepare_data(num_proc=0, dataset=dm.test_dataset)
print(f"Training batches: {len(dm.train_dataloader())}")
print(f"Validation batches: {len(dm.val_dataloader())}")
print(f"Test batches: {len(dm.test_dataloader())}")

# Quick view
for batch in dm.train_dataloader():
    for i in range(2):
        show_volume(batch["data"][i][0], fig_size=(6, 4), v_min_max=(-1.0, 3.0))
    break

# %% [markdown]
# ## Prepare 3D model
#
# LightningModule is the core of PL, it wrappes all model related peaces, mainly:
#
# - the model/architecture/weights
# - evaluation metrics
# - configs for optimizer and LR scheduler

# %%
from monai.networks.nets import SEResNet50, resnet10, resnet18, resnet34, resnet50
from torch.optim import ASGD, SGD, Adamax
from torchsummary import summary

from kaggle_volclassif.models import LitBrainMRI, create_pretrained_medical_resnet

# ==============================

PATH_PRETRAINED_WEIGHTS = os.path.join(PATH_MODELS, "resnet_34_23dataset.pth")
net, pretraineds_layers = create_pretrained_medical_resnet(PATH_PRETRAINED_WEIGHTS, model_constructor=resnet34)

# net = SEResNet50(spatial_dims=3, in_channels=1, pretrained=True, num_classes=2)

model = LitBrainMRI(net=net, pretrained_params=None, lr=5e-4, optimizer=Adamax)
# summary(model, input_size=(1, 128, 128, 128))

# %% [markdown]
# ## Train a model
#
# Lightning forces the following structure to your code which makes it reusable and shareable:
#
# - Research code (the LightningModule).
# - Engineering code (you delete, and is handled by the Trainer).
# - Non-essential research code (logging, etc... this goes in Callbacks).
# - Data (use PyTorch DataLoaders or organize them into a LightningDataModule).
#
# Once you do this, you can train on multiple-GPUs, TPUs, CPUs and even in 16-bit precision without changing your code!

# %%
import pytorch_lightning as pl

from kaggle_volclassif.models import FineTuneCB

torch.backends.cudnn.enabled = False

csv_logger = pl.loggers.CSVLogger(save_dir="logs/", name=model.name)
tb_logger = pl.loggers.TensorBoardLogger(save_dir="logs-tb/", name=model.name)
swa = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.6)
fine = FineTuneCB(unfreeze_epoch=10)
ckpt = pl.callbacks.ModelCheckpoint(
    monitor="valid/auroc",
    save_top_k=1,
    save_last=True,
    filename="checkpoint/{epoch:02d}-{valid/auroc:.4f}",
    mode="max",
)

# ==============================

trainer = pl.Trainer(
    # overfit_batches=5,
    # fast_dev_run=True,
    gpus=[1],
    callbacks=[ckpt, fine],  # , swa
    logger=[csv_logger, tb_logger],
    max_epochs=35,
    precision=16,
    benchmark=True,
    accumulate_grad_batches=12,
    # val_check_interval=0.5,
    progress_bar_refresh_rate=1,
    log_every_n_steps=5,
    weights_summary="top",
    auto_lr_find=True,
    #     auto_scale_batch_size='binsearch',
)

# ==============================

trainer.tune(
    model,
    datamodule=dm,
    lr_find_kwargs=dict(min_lr=2e-5, max_lr=1e-2, num_training=25),
    # scale_batch_size_kwargs=dict(max_trials=5),
)
print(f"Batch size: {dm.batch_size}")
print(f"Learning Rate: {model.learning_rate}")

# ==============================

# dm.batch_size = 12
trainer.fit(model=model, datamodule=dm)

# %% [markdown]
# ### Training progress

# %%
print(csv_logger.log_dir)
metrics = pd.read_csv(f"{csv_logger.log_dir}/metrics.csv")
display(metrics.head())

aggreg_metrics = []
agg_col = "epoch"
for i, dfg in metrics.groupby(agg_col):
    agg = dict(dfg.mean())
    agg[agg_col] = i
    aggreg_metrics.append(agg)

df_metrics = pd.DataFrame(aggreg_metrics)
df_metrics[["train/loss", "valid/loss"]].plot(grid=True, legend=True, xlabel=agg_col)
df_metrics[["train/f1", "train/auroc", "valid/f1", "valid/auroc"]].plot(grid=True, legend=True, xlabel=agg_col)

# %% [markdown]
# ## Predictions

# %%
model.eval()
model.cpu()
submission = []

for batch in dm.test_dataloader():
    print(batch.keys())
    print(batch.get("label"))
    imgs = batch.get("data")
    print(imgs.shape)
    with torch.no_grad():
        preds = model(imgs)
    print(preds)
    probs = torch.nn.functional.softmax(preds)
    print(probs)
    break

# %%
from kaggle_volclassif.models import make_submission

dm.batch_size = 2
df_submission = make_submission(model, dm.test_dataloader(), "cuda" if torch.cuda.is_available() else "cpu")
display(df_submission)
df_submission["MGMT_value"].to_csv("submission.csv")

# %%
df_submission[["MGMT_value"]].hist(bins=25)
