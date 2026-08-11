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
# # Brain Tumor Classification with PyTorch⚡Lightning & EfficientNet 3D
#
# The goal of this challenge is to Predict the status of a genetic biomarker important for brain cancer treatment.
#
# All the code is referred from public repository: https://github.com/Borda/kaggle_vol-3D-classify
# Any nice contribution is welcome!

# %%
# ! pip uninstall -y kaggle_volclassif
# ! pip install -q https://github.com/Borda/kaggle_vol-3D-classify/archive/refs/tags/v0.1.0.zip
# ! pip uninstall -q -y wandb
# ! pip list | grep torch

# ! ls -l $PATH_DATASET
# ! nvidia-smi
# ! mkdir $PATH_TEMP

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

from IPython.display import display

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
# - T1: weighting better delineates anatomy
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

_IS_KAGGLE = os.path.isdir("/kaggle/input")
_IS_COLAB = os.path.isdir("/content")
if _IS_KAGGLE:
    PATH_DATASET = os.environ.get("PATH_DATASET", "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification")
    PATH_TEMP = os.environ.get("PATH_TEMP", "/kaggle/working/brain-tumor")
elif _IS_COLAB:
    PATH_DATASET = os.environ.get("PATH_DATASET", "/content/rsna-miccai-brain-tumor")
    PATH_TEMP = os.environ.get("PATH_TEMP", "/content/brain-tumor")
else:
    PATH_DATASET = os.environ.get("PATH_DATASET", "data/rsna-miccai-brain-tumor")
    PATH_TEMP = os.environ.get("PATH_TEMP", "data/brain-tumor")
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
# The basic building block is transforming raw data to Torch Dataset.
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
# It is convenient to wrap all data-related pieces and define PyTorch dataloader for Training / Validation / Testing phase.
#
# At the end we show a few sample images from the first training batch.

# %%
import math

from monai.transforms import Compose, NormalizeIntensityd, RandAffined, RandAxisFlipd, RandRotate90d

from kaggle_volclassif.data import BrainScansDM

# ==============================

# Dataset >> mean: 0.13732214272022247 STD: 0.24326834082603455
norm_intensity = NormalizeIntensityd(keys=["data"], subtrahend=0.137, divisor=0.243)

# define transformations
TRAIN_TRANSFORMS = Compose([
    RandRotate90d(keys=["data"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
    RandAxisFlipd(keys=["data"], prob=0.5),
    RandAffined(
        keys=["data"],
        prob=0.5,
        scale_range=(0.1,) * 3,
        rotate_range=(math.radians(10),) * 3,
        translate_range=(6,) * 3,
        mode="nearest",
        padding_mode="zeros",
    ),
    norm_intensity,
])
VAL_TRANSFORMS = Compose([norm_intensity])

# ==============================

dm = BrainScansDM(
    data_dir=PATH_DATASET,
    scan_types=["T2w"],
    vol_size=224,
    crop_thr=1e-6,
    batch_size=3,
    cache_dir=PATH_TEMP,
    # in_memory=True,
    num_workers=6,
    train_transforms=TRAIN_TRANSFORMS,
    valid_transforms=VAL_TRANSFORMS,
)
# dm.prepare_data(3)
dm.setup()
print(f"Training batches: {len(dm.train_dataloader())} and Validation {len(dm.val_dataloader())}")

# Quick view
for batch in dm.train_dataloader():
    for i in range(2):
        show_volume(batch["data"][i][0], fig_size=(9, 6), v_min_max=(-1.0, 3.0))
    break

# %% [markdown]
# ## Prepare 3D model
#
# LightningModule is the core of PL, it wraps all model related pieces, mainly:
#
# - the model/architecture/weights
# - evaluation metrics
# - configs for optimizer and LR scheduler

# %%
from torchsummary import summary

from kaggle_volclassif.models import LitBrainMRI

# ==============================

model = LitBrainMRI(lr=1e-3)
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
from pytorch_lightning.tuner import Tuner

logger = pl.loggers.CSVLogger(save_dir="logs/", name=model.name)
swa = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.6)
ckpt = pl.callbacks.ModelCheckpoint(
    monitor="valid/f1",
    save_top_k=1,
    save_last=True,
    filename="checkpoint/{epoch:02d}-{valid_acc:.4f}-{valid_f1:.4f}",
    mode="max",
)

# ==============================

trainer = pl.Trainer(
    # overfit_batches=5,
    # fast_dev_run=True,
    accelerator="gpu",
    devices=1,
    callbacks=[ckpt],  # , swa
    logger=logger,
    max_epochs=30,
    precision=16,
    accumulate_grad_batches=12,
    # val_check_interval=0.5,
    log_every_n_steps=5,
)

# ==============================

tuner = Tuner(trainer)
tuner.lr_find(model, datamodule=dm, min_lr=2e-5, max_lr=1e-2, num_training=35)
print(f"Batch size: {dm.batch_size}")
print(f"Learning Rate: {model.learning_rate}")

# ==============================

trainer.fit(model=model, datamodule=dm)

# %% [markdown]
# ### Training progress

# %%
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
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
