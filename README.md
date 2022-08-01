# Kaggle: [Brain Tumor Radiogenomic Classification](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification)

[![CI complete testing](https://github.com/Borda/kaggle_vol-3D-classify/actions/workflows/ci_testing.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_vol-3D-classify/actions/workflows/ci_testing.yml)
[![Code formatting](https://github.com/Borda/kaggle_vol-3D-classify/actions/workflows/code-format.yml/badge.svg?branch=main&event=push)](https://github.com/Borda/kaggle_vol-3D-classify/actions/workflows/code-format.yml)
[![codecov](https://codecov.io/gh/Borda/kaggle_vol-3D-classify/branch/main/graph/badge.svg?token=bxqTQDXHvU)](https://codecov.io/gh/Borda/kaggle_vol-3D-classify)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Borda/kaggle_vol-3D-classify/main.svg)](https://results.pre-commit.ci/latest/github/Borda/kaggle_vol-3D-classify/main)

The goal of this challenge is to Predict the status of a genetic biomarker important for brain cancer treatment.

![Sample brain visual](./assets/brain3D_spl1.png)

With interpolation in Z dimension as it happens it is quite sparse:
![Sample brain visual](./assets/brain3D_spl2.png)

Each independent case has a dedicated folder identified by a five-digit number.
Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in DICOM format.
The exact mpMRI scans included are:

- **FLAIR**: Fluid Attenuated Inversion Recovery
- **T1w**: T1-weighted pre-contrast
- **T1Gd**: T1-weighted post-contrast
- **T2**: T2-weighted

The labels/targets are `MGMT_value`:

![Label distribution](./assets/labels.png)

## Experimentation

### install this tooling

A simple way how to use this basic functions:

```bash
! pip install https://github.com/Borda/kaggle_vol-3D-classify/archive/refs/heads/main.zip
```

### run notebooks in Kaggle

- [🧠 Brain Tumor Classif. ~ Lightning⚡EfficientNet3D](https://www.kaggle.com/jirkaborovec/brain-tumor-classif-lightning-efficientnet3d)
- [🧠 Brain Tumor Classif. ~ Lightning⚡MONAI-ResNet3D](https://www.kaggle.com/jirkaborovec/brain-tumor-classif-lightning-monai-resnet3d)

### local notebooks

- [Brain tumor classification meets PT-Lightning and MONAI EfficientNet3D](notebooks/Brain-tumor-classif_PT-Lightning_EfficientNet3D.ipynb)
- [Brain tumor classification meets PT-Lightning and pre-trained ResNet3D](notebooks/Brain-tumor-classif_PT-Lightning_ResNet3D.ipynb)

### some results

Training progress with EfficientNet3D with training  for 10 epochs > over 96% validation accuracy:

![Training process](./assets/metrics.png)
