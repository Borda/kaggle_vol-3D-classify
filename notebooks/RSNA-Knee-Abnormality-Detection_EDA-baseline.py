# %% [markdown]
# # RSNA Knee Abnormality Detection — 12-Label MRI Screening with ⚡PTL + timm

# This notebook builds an end-to-end baseline for the RSNA Knee Abnormality Detection challenge, where every knee MRI
# study must receive twelve independent confidence scores — one per finding, from ACL tears to Baker's cysts — and the
# leaderboard ranks us by the macro-averaged ROC AUC over those twelve columns.

# Our approach is deliberately the *simplest thing that respects the physics of the data*: pick one informative MRI
# series per study, sample a fixed number of slices from it, embed each slice with a pretrained 2D timm backbone, and
# pool the slice embeddings with a learned attention head (the "2.5D multiple-instance learning" recipe that the MRNet
# knee-MRI literature established). We wire it together with PyTorch Lightning so the training loop, metric handling,
# checkpointing and mixed precision stay declarative rather than hand-rolled.

# Competition: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

# %%
# JPEG Lossless and JPEG 2000 transfer syntaxes are present in this dataset — without the pylibjpeg trio a large
# share of series raise a decoder error at `.pixel_array` and silently disappear from training.
# A Python variable, not a shell one: each `!` line runs in its own subshell, so a shell assignment would not survive
# to the next line — `{PKGS}` is interpolated by IPython before the shell ever sees it.
PKGS = "pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
PKGS += " timm pytorch-lightning torchmetrics iterative-stratification seaborn"

# ! pip download -q {PKGS} --dest frozen_packages/
# ! pip install -q --no-index --find-links frozen_packages/ {PKGS} || pip install -q {PKGS}

# %% [markdown]
# ## 1. Imports, paths, and reproducibility

# Before any exploration we fix the entire execution context in a single place: which libraries are actually installed
# (versions matter — Lightning 2.x renamed enough of the 1.x API that a stale snippet fails loudly), where the data
# lives, and what the random seed is. Doing this once at the top means every later cell can be read without wondering
# which `pl` is in scope or whether a path was quietly redefined halfway through.

# Two deliberate choices are worth naming:

# - **`pytorch_lightning` namespace, not `lightning.pytorch`.** Both exist and both work, but a checkpoint written by
#   a module defined under one namespace and reloaded through the other is a classic silent failure. We pick one — the
#   one this repository pins — and use it consistently in the model definition and at checkpoint reload time.
# - **Stage constants stay out of this cell.** Batch size, slice count and learning rate appear immediately before the
#   stage that consumes them, so a reader tuning the model never has to scroll back to the top.

# %%
import glob
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import pytorch_lightning as pl
import seaborn as sns
import sklearn
import timm
import torch
import torchmetrics
from IPython.display import display
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# %matplotlib inline

# pydicom emits one UserWarning per file for the vendor-specific tags this dataset keeps; at ~820k files that noise
# would drown every other diagnostic, so we silence exactly that category and nothing broader.
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

SEED = 42
PATH_DATASET = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
# LOCAL OVERRIDE — set to a local copy of the competition data when running outside Kaggle; empty keeps the Kaggle path.
PATH_DATASET_OVERRIDE = ""
if PATH_DATASET_OVERRIDE:
    PATH_DATASET = PATH_DATASET_OVERRIDE
PATH_OUTPUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

print(f"python       : {sys.version.split()[0]}")
print(f"numpy        : {np.__version__}")
print(f"pandas       : {pd.__version__}")
print(f"scikit-learn : {sklearn.__version__}")
print(f"pydicom      : {pydicom.__version__}")
print(f"torch        : {torch.__version__}")
print(f"lightning    : {pl.__version__}")
print(f"torchmetrics : {torchmetrics.__version__}")
print(f"timm         : {timm.__version__}")
print(f"device       : {'cuda' if torch.cuda.is_available() else 'cpu'}")

_ = pl.seed_everything(SEED, workers=True)

# %% [markdown]
# ### Foundation lens

# The lens cell answers one question before we spend any compute: *is the data actually mounted where we think it is?*
# On Kaggle the answer is yes; anywhere else it is usually no, and a notebook that assumes otherwise dies twenty cells
# later with a confusing `FileNotFoundError`. So this cell asserts its inputs: the dataset directory and each of the
# five index tables must exist here, or execution stops immediately with a message naming the offending path. Every
# cell below is written for the case where those assertions passed — no cell degrades quietly on absent data.

# %%
# Fail fast: a missing dataset must crash here, not leave every cell below pretending to run.
assert os.path.isdir(PATH_DATASET), (
    f"dataset not mounted at {PATH_DATASET} — set PATH_DATASET_OVERRIDE to a local copy before running"
)

print(f"PATH_DATASET   : {PATH_DATASET}")
print(f"PATH_OUTPUT    : {PATH_OUTPUT}")

for _name in ["train.csv", "train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv"]:
    _path = os.path.join(PATH_DATASET, _name)
    assert os.path.isfile(_path), f"missing {_path} — the mounted directory is not the competition dataset"
    print(f"  found : {_path}")

# %% [markdown]
# ## 2. Exploratory data analysis

# EDA here is not a formality — this competition hides three structural traps in its schema, and each one changes the
# model we are allowed to build. First, only a *subset* of training studies carries the twelve labels; the rest ship a
# free-text radiology report instead, so "how many rows can I actually train on?" is a question with a surprising
# answer. Second, the imaging is 570 GB across ~820k DICOM files, so any design that reads every slice of every series
# is dead on arrival and we need evidence about series and slice counts to size a sampler. Third, the twelve findings
# are wildly imbalanced and correlated, which drives both the loss and the split strategy.

# Every check below ends with a printed finding and the design decision it forces. The section closes with a
# consolidated list of those decisions, which is the actual hand-off to the modelling sections.

# %%
# Grounded label names — spaces and the apostrophe in "Baker's" are part of the column headers, so this list is the
# single source of truth for label selection, metric width, and submission column order.
SAMPLE_N = 9
TARGET_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
print(f"targets: {len(TARGET_COLS)} -> {TARGET_COLS}")

# %% [markdown]
# ### Dataset overview

# We start from the two tables that describe the corpus: `train.csv` (one row per study, carrying patient sex, the
# free-text report, and the twelve label columns) and `train_series.csv` (one row per series, carrying the acquisition
# metadata that will drive our series selection). Loading them separately keeps the study-level and series-level
# questions distinct, which matters because they have different row counts and different missingness patterns.

# %%
# Study-level table: labels live here, and so does the report text we will inspect for the partial-label problem.
df_train = pd.read_csv(os.path.join(PATH_DATASET, "train.csv"))
print(f"train.csv shape: {df_train.shape}")
display(df_train.head())

# %%
# Series-level table: plane and fluid-sensitivity flags are the only way to choose a series without opening DICOMs.
df_series = pd.read_csv(os.path.join(PATH_DATASET, "train_series.csv"))
print(f"train_series.csv shape: {df_series.shape}")
display(df_series.head())

# %%
# Dtypes reveal whether the label columns arrived as float (the tell-tale sign of NaN-bearing partial labels).
display(df_train.dtypes.to_frame("dtype"))

# %%
# Missingness per column is the headline diagnostic for this competition: it quantifies the partial-label split.
_miss = pd.DataFrame({
    "missing": df_train.isna().sum(),
    "missing_pct": (df_train.isna().mean() * 100).round(2),
})
display(_miss)

# %%
# Descriptive stats over the label columns give prevalence and count in one view; count < len(df) means partial labels.
display(df_train[TARGET_COLS].describe().T[["count", "mean", "min", "max"]])

# %%
# A duplicated StudyInstanceUID would leak the same knee across the train/validation boundary.
_dup_studies = int(df_train["StudyInstanceUID"].duplicated().sum())
_dup_series = int(df_series["SeriesInstanceUID"].duplicated().sum())
print(f"duplicated StudyInstanceUID  : {_dup_studies}")
print(f"duplicated SeriesInstanceUID : {_dup_series}")
print(
    "FINDING: identifiers are unique -> IMPLICATION: a study-level split needs no de-duplication step."
    if _dup_studies == 0 and _dup_series == 0
    else "FINDING: duplicate identifiers exist -> IMPLICATION: de-duplicate before splitting or the split leaks."
)

# %%
# Referenced-file check on a small sample: the index promises directories exist, and a broken promise must surface here,
# not inside a DataLoader worker where the traceback is unreadable.
_probe = df_series.sample(min(SAMPLE_N, len(df_series)), random_state=SEED)
_found = 0
for _, _row in _probe.iterrows():
    _dir = os.path.join(PATH_DATASET, "train_series", _row["StudyInstanceUID"], _row["SeriesInstanceUID"])
    _n_files = len(glob.glob(os.path.join(_dir, "*.dcm")))
    _found += int(_n_files > 0)
    print(f"  {_n_files:4d} slices : {_row['SeriesInstanceUID'][:24]}...")
# Fail fast: zero resolved directories means the mounted layout does not match the index at all.
assert _found > 0, "no sampled series resolved to files — dataset layout differs from train_series.csv"
print(f"FINDING: {_found}/{len(_probe)} sampled series resolved to non-empty directories.")
print(
    "IMPLICATION: the index is trustworthy here; the Dataset still guards empty directories for the full corpus."
    if _found == len(_probe)
    else "IMPLICATION: some sampled directories are empty — the Dataset's empty-directory guard is load-bearing."
)

# %% [markdown]
# ### Target distribution

# The first plot answers the question that governs both our loss and our split: *how prevalent is each of the twelve
# findings?* Macro-averaged ROC AUC weights every label equally, so a finding present in 2% of studies contributes
# exactly as much to our score as one present in 40% — and rare labels are precisely where an unlucky validation split
# produces a meaningless AUC. We expect a steep prevalence gradient across the twelve columns.

# %%
# Prevalence is computed on labelled rows only; NaN rows would otherwise silently deflate every bar.
_prev = df_train[TARGET_COLS].mean(skipna=True).sort_values(ascending=False)
plt.figure(figsize=(10, 4))
_ = plt.bar(_prev.index, _prev.to_numpy(), color="steelblue")
_ = plt.xticks(rotation=45, ha="right")
_ = plt.xlabel("finding")
_ = plt.ylabel("positive rate")
_ = plt.title("Per-label prevalence across labelled training studies")
_ = plt.grid(True)
plt.tight_layout()
plt.show()
display(_prev.to_frame("positive_rate").round(4))

# %% [markdown]
# The bars span roughly a four-fold range on the labelled subset — effusion at the common end (~60%), MCL at the rare
# end (~16%). Two consequences follow directly.

# - **Loss**: plain `BCEWithLogitsLoss` remains the right default — AUC is threshold-free and rank-based, so it does not
#   need the class rebalancing that an accuracy-style objective would demand. Per-label `pos_weight` is the obvious
#   first upgrade to try if the rare columns underperform.
# - **Split**: a naive random split can hand the validation fold too few positives of a rare finding, at which point
#   that column's AUC becomes noise and the macro average inherits the noise. This is what motivates the multilabel-
#   stratified split two sections down.

# %% [markdown]
# Prevalence alone does not tell us whether the twelve columns are independent problems or one problem in twelve
# disguises. The co-occurrence heatmap below counts how often each pair of findings appears in the same study, which
# tells us whether a shared backbone with twelve heads is a reasonable inductive bias.

# %%
# Co-occurrence counts on complete rows only — a dot product over NaN-bearing columns would propagate NaN everywhere.
_labelled_mask = df_train[TARGET_COLS].notna().all(axis=1)
_lab = df_train.loc[_labelled_mask, TARGET_COLS].astype(int)
_cooc = _lab.T.dot(_lab)
plt.figure(figsize=(8, 6.5))
_ = sns.heatmap(_cooc, annot=True, fmt="d", cmap="viridis", cbar_kws={"label": "co-occurring studies"})
_ = plt.xlabel("finding")
_ = plt.ylabel("finding")
_ = plt.title("Label co-occurrence (diagonal = positives per label)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Off-diagonal mass is substantial — the three osteoarthritis compartments travel together, and effusion accompanies
# most acute injuries. That is direct evidence for a *shared representation with twelve linear heads* rather than
# twelve separate models: the backbone can learn "this knee is degenerated" once and let each head read it off. It also
# means our stratified split must balance label *combinations*, not each label in isolation.

# %% [markdown]
# ### Hypothesis 1 — the labels are imbalanced enough to threaten rare-column AUC

# We suspect the rarest findings have so few positives that a random 20% validation fold could contain a handful of
# them. The check below counts positives per label and projects how many land in validation, because a column with
# fewer than ~10 validation positives yields an AUC estimate too unstable to guide model selection.

# %%
# Projected validation positives per label decide whether stratification is optional or mandatory.
_pos = df_train.loc[_labelled_mask, TARGET_COLS].sum().astype(int)
_proj = (_pos * 0.2).round().astype(int)
display(pd.DataFrame({"positives": _pos, "projected_val_positives@20%": _proj}).sort_values("positives"))
_fragile = _proj[_proj < 10]
print(f"FINDING: {len(_fragile)}/{len(TARGET_COLS)} labels project to <10 val positives: {list(_fragile.index)}")
print(
    f"IMPLICATION: multilabel-stratified splitting is mandatory, and the macro AUC leans on {len(_fragile)}"
    " fragile columns — read per-run scores with caution."
    if len(_fragile)
    else "IMPLICATION: every label projects enough validation positives — stratification is a nicety, not a must."
)

# %% [markdown]
# ### Hypothesis 2 — only a subset of studies is labelled, and the reports cover the rest

# This is the defining twist of the competition. The organisers state that only a small subset of training studies
# carries the twelve per-condition labels and that the free-text `Report` column exists so competitors can derive
# labels for the remainder. The check quantifies the three populations — fully labelled, unlabelled but reported, and
# neither — because the ratio decides whether weak supervision is a nice-to-have or the whole game.

# %%
# Counting the labelled/unlabelled populations sizes both the honest baseline and the weak-supervision upside.
_n_total = len(df_train)
_n_labelled = int(df_train[TARGET_COLS].notna().all(axis=1).sum())
_n_partial = int(df_train[TARGET_COLS].notna().any(axis=1).sum()) - _n_labelled
_has_report = df_train["Report"].notna() & df_train["Report"].astype(str).str.strip().ne("")
_n_unlabelled_reported = int((~df_train[TARGET_COLS].notna().all(axis=1) & _has_report).sum())
print(f"studies total                        : {_n_total}")
print(f"fully labelled (all 12 present)      : {_n_labelled}")
print(f"partially labelled (some present)    : {_n_partial}")
print(f"unlabelled but carrying a report     : {_n_unlabelled_reported}")
_share = _n_labelled / max(_n_total, 1)
print(f"labelled share                       : {_share:.1%}")
if _share < 0.5:
    print(f"FINDING: only {_share:.1%} of studies carry labels — most are described by free text alone.")
    print("IMPLICATION: the baseline trains on fully labelled rows ONLY (never on NaN targets); report-derived")
    print("             weak labels are a gated extension, evaluated against the same clean validation fold.")
else:
    print(f"FINDING: {_share:.1%} of studies carry labels — supervision is plentiful.")
    print("IMPLICATION: weak labels from reports would add little; train on the labelled rows directly.")

# %% [markdown]
# ### Hypothesis 3 — series and slice counts allow a fixed-length sampler

# 570 GB is only tractable if we read a bounded number of slices per study. The organisers describe series of typically
# 20–45 slices with a median near 30 and a tail into the hundreds. If that holds, a fixed sample of ~24 slices captures
# most of the volume for a typical series while capping the cost of the outliers. We verify series-per-study from the
# index (free) and slice counts from a small directory sample (cheap).

# %%
# Series per study, straight from the index — this tells us how much choice the selection rule actually has.
_per_study = df_series.groupby("StudyInstanceUID")["SeriesInstanceUID"].count()
display(_per_study.describe().to_frame("series_per_study"))
plt.figure(figsize=(8, 3.5))
_ = plt.hist(_per_study.to_numpy(), bins=range(1, int(_per_study.max()) + 2), color="steelblue", align="left")
_ = plt.xlabel("series per study")
_ = plt.ylabel("number of studies")
_ = plt.title("How many series each study contains")
_ = plt.grid(True)
plt.tight_layout()
plt.show()
_med_series = int(_per_study.median())
print(f"FINDING: median {_med_series} series per study -> IMPLICATION: selecting ONE series per study cuts imaging")
print(f"         I/O roughly {_med_series}x; the unused series remain available for a later multi-series ensemble.")

# %%
# Slice counts require touching the filesystem, so we sample rather than scan all ~820k files.
_probe = df_series.sample(min(50, len(df_series)), random_state=SEED)
_counts = []
for _, _row in tqdm(_probe.iterrows(), total=len(_probe), desc="counting slices"):
    _dir = os.path.join(PATH_DATASET, "train_series", _row["StudyInstanceUID"], _row["SeriesInstanceUID"])
    _counts.append(len(glob.glob(os.path.join(_dir, "*.dcm"))))
_counts = pd.Series(_counts)
display(_counts.describe().to_frame("slices_per_series"))
if _counts.max() > 0:
    plt.figure(figsize=(8, 3.5))
    _ = plt.hist(_counts.to_numpy(), bins=20, color="darkorange")
    _ = plt.xlabel("slices per series")
    _ = plt.ylabel("number of series")
    _ = plt.title("Slice-count distribution over a random series sample")
    _ = plt.grid(True)
    plt.tight_layout()
    plt.show()
_med_slices, _max_slices = float(_counts.median()), float(_counts.max())
_coverage = min(24.0 / max(_med_slices, 1.0), 1.0)
print(f"FINDING: median {_med_slices:.0f} slices per series, max {_max_slices:.0f} in this sample.")
print(f"IMPLICATION: a FIXED, evenly spread sample of 24 slices covers {_coverage:.0%} of the median series while")
print(f"             bounding the {_max_slices:.0f}-slice tail, and keeps a constant tensor shape for batching.")

# %% [markdown]
# ### Hypothesis 4 — sagittal fluid-sensitive series exist for most studies

# Our selection rule is only viable if the preferred acquisition actually exists study by study. Radiologically, the
# sagittal fluid-sensitive (T2/PD/STIR-like) series is the workhorse for this label set: it shows the cruciate
# ligaments along their length, cuts the menisci in cross-section, and makes effusion and marrow oedema bright. The
# check measures coverage of that preference and of each fallback rung.

# %%
# Coverage of each fallback rung decides whether the selection chain needs a third rung or a fourth.
display(df_series["Anatomical_Plane"].value_counts(dropna=False).to_frame("series"))
display(df_series["Fluid_Sensitive"].value_counts(dropna=False).to_frame("series"))
_studies = df_series["StudyInstanceUID"].nunique()
_sag_fs = df_series[(df_series["Anatomical_Plane"] == "Sagittal") & (df_series["Fluid_Sensitive"] == 1)]
_sag = df_series[df_series["Anatomical_Plane"] == "Sagittal"]
_n_sag_fs = _sag_fs["StudyInstanceUID"].nunique()
_n_sag = _sag["StudyInstanceUID"].nunique()
print(f"studies with a sagittal fluid-sensitive series : {_n_sag_fs} / {_studies}")
print(f"studies with any sagittal series               : {_n_sag} / {_studies}")
print(
    f"FINDING: sagittal fluid-sensitive covers {_n_sag_fs / _studies:.1%}; any sagittal covers {_n_sag / _studies:.1%}."
)
print(
    "IMPLICATION: the fallback rungs are exercised in training data — the chain sagittal+fluid-sensitive ->"
    "\n             any sagittal -> any series keeps every study predicted."
    if _n_sag_fs < _studies
    else "IMPLICATION: the preferred acquisition is universal here — the fallback chain remains as a safety net"
    "\n             for the hidden test data."
)

# %% [markdown]
# ### Hypothesis 5 — the reports are multilingual, which bounds how far keyword rules can take us

# Weak supervision from text only works if we can read the text. The competition notes that reports may be written in
# any of several languages, and an English-only keyword rule applied to a multilingual corpus does not fail loudly — it
# fails *silently*, by labelling every non-English study negative. The check below estimates the language mix with a
# cheap marker-word heuristic and a non-ASCII character scan.

# %%
# A crude language probe is enough to decide whether keyword weak-labelling is safe to enable by default.
_LANG_MARKERS = {
    "english": [" the ", " and ", " with ", " no "],
    "german": [" der ", " und ", " mit ", " kein "],
    "spanish": [" el ", " los ", " con ", " sin "],
    "french": [" le ", " les ", " avec ", " sans "],
    "dutch": [" het ", " een ", " met ", " geen "],
}
_reports = df_train["Report"].dropna().astype(str).str.lower()
_non_ascii = float(_reports.apply(lambda t: any(ord(ch) > 127 for ch in t)).mean())
_hits = {
    _lang: int(_reports.apply(lambda t, mk=_marks: any(w in f" {t} " for w in mk)).sum())
    for _lang, _marks in _LANG_MARKERS.items()
}
display(pd.Series(_hits, name="reports_matching_markers").sort_values(ascending=False).to_frame())
print(f"reports scanned                : {len(_reports)}")
print(f"share containing non-ASCII text: {_non_ascii:.1%}")
_langs_hit = [_lang for _lang, _n in _hits.items() if _n >= max(1, len(_reports) // 100)]
if len(_langs_hit) > 1:
    print(f"FINDING: marker words fire for {len(_langs_hit)} languages ({', '.join(_langs_hit)}) — not English-only.")
    print("IMPLICATION: report-derived labels are WEAK and noisy — they stay behind a feature flag, are never used")
    print("             for validation, and the reader is told exactly which languages the keyword map misses.")
else:
    print("FINDING: marker words fire for one language only — the corpus reads as effectively monolingual.")
    print("IMPLICATION: keyword weak-labelling is less risky than feared, but stays gated until measured.")

# %% [markdown]
# ### Modality display — from DICOM files to a normalised volume

# Everything above was tabular. Now we open the imaging itself, because three properties of these DICOMs directly shape
# the Dataset we are about to write: slices arrive as one file each and must be *ordered*, intensities are not
# comparable across scanners and must be *normalised*, and the compressed transfer syntaxes must actually *decode*.

# The helpers below are the exact ones the training Dataset will reuse — defining them here means the pictures we look
# at are produced by the same code path that feeds the model, not by a throwaway viewer that might disagree with it.

# %% [markdown]
# The first helper picks one series per study. It implements the fallback chain that Hypothesis 4 justified, and it
# sorts by `SeriesInstanceUID` inside each rung so the choice is deterministic across runs — a selector that returns a
# different series each epoch would make validation scores irreproducible.


# %%
def select_one_series(df_series_all):
    """Pick one series per study: sagittal fluid-sensitive, else any sagittal, else any series."""
    df = df_series_all.copy()
    df["_rank"] = 2
    df.loc[df["Anatomical_Plane"] == "Sagittal", "_rank"] = 1
    df.loc[(df["Anatomical_Plane"] == "Sagittal") & (df["Fluid_Sensitive"] == 1), "_rank"] = 0
    df = df.sort_values(["StudyInstanceUID", "_rank", "SeriesInstanceUID"])
    return df.groupby("StudyInstanceUID", as_index=False).first().drop(columns=["_rank"])


# %% [markdown]
# Ordering slices correctly matters more than it looks: a shuffled stack destroys the through-plane continuity that the
# attention head learns to exploit. We read `InstanceNumber` from the *header only* — `stop_before_pixels=True` skips
# pixel decoding, which is roughly two orders of magnitude cheaper — and fall back to filename order when any file in
# the series lacks the tag, because a partially-sorted stack is worse than a consistently-sorted arbitrary one.


# %%
def sorted_slice_paths(dir_series):
    """Order a series' `.dcm` files by InstanceNumber, falling back to filename order when the tag is unusable."""
    paths = sorted(glob.glob(os.path.join(dir_series, "*.dcm")))
    if not paths:
        return []
    numbers = []
    for path in paths:
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
            value = getattr(header, "InstanceNumber", None)
            numbers.append(float(value) if value is not None else np.nan)
        except (OSError, ValueError, TypeError, pydicom.errors.InvalidDicomError):
            numbers.append(np.nan)
    if np.isnan(numbers).any():
        return paths
    return [path for _, path in sorted(zip(numbers, paths))]


# %% [markdown]
# The slice reader is where the compressed transfer syntaxes bite. `pixel_array` transparently decompresses JPEG
# Lossless and JPEG 2000 *only* when the pylibjpeg plugins are installed — otherwise it raises, which is exactly why
# they are in the setup cell. We catch the realistic failure modes (unreadable file, missing decoder, malformed
# dataset) and return `None` so a single corrupt file degrades one slice rather than killing an epoch.


# %%
def read_dicom_slice(path):
    """Decode one DICOM file to a 2D float32 array, or return None when the file cannot be read."""
    try:
        arr = pydicom.dcmread(path).pixel_array.astype(np.float32)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        NotImplementedError,
        AttributeError,
        pydicom.errors.InvalidDicomError,
    ):
        return None
    if arr.ndim == 3:  # rare multi-frame file — take the central frame
        arr = arr[arr.shape[0] // 2]
    return arr if arr.ndim == 2 else None


# %% [markdown]
# The volume reader turns a directory into the fixed-shape tensor the model consumes. Three decisions are encoded here.

# - **Even index sampling** via `np.linspace`: it spans the whole series rather than a contiguous chunk, so a lesion at
#   either end is still visible. When a series has fewer slices than `n_slices`, the rounded indices repeat — the stack
#   is padded by duplication, which keeps the tensor shape constant without inventing black slices.
# - **Per-volume percentile normalisation**: MRI intensities carry no absolute meaning and vary by scanner and
#   sequence, so we clip to the 1st–99th percentile of *this* volume and rescale to 0–1. Percentiles rather than
#   min/max because a single hot voxel would otherwise compress the entire useful range.
# - **Resize before stacking**: slices within a series can differ in matrix size; resizing each one individually makes
#   the function robust to that instead of failing on `torch.cat`.


# %%
def read_dicom_volume(dir_series, n_slices=24, image_size=224):
    """Read a DICOM series into a normalised (n_slices, 1, image_size, image_size) float32 tensor."""
    paths = sorted_slice_paths(dir_series)
    if not paths:
        return torch.zeros(n_slices, 1, image_size, image_size, dtype=torch.float32)
    indices = np.linspace(0, len(paths) - 1, n_slices).round().astype(int)
    cache, slices = {}, []
    for i in indices:
        if i not in cache:  # short series repeat indices — decode each file at most once
            arr = read_dicom_slice(paths[i])
            tensor = torch.zeros(1, 1, image_size, image_size) if arr is None else torch.from_numpy(arr)[None, None]
            if arr is not None:
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
                )
            cache[i] = tensor
        slices.append(cache[i])
    volume = torch.cat(slices, dim=0).float()
    # ==============================
    lo, hi = np.percentile(volume.numpy(), [1.0, 99.0])
    return ((volume - lo) / max(float(hi - lo), 1e-6)).clamp(0.0, 1.0)


# %% [markdown]
# With the helpers in place we load one real study. The figure shows the same volume cut three ways: the native
# acquisition plane, and two reconstructions through the stack. There is no mask overlay here — this is a
# classification problem, so what we are checking is anatomical legibility and normalisation quality, not annotation
# alignment.

# %%
# One concrete volume beats any amount of schema reading — it validates decoder, ordering and scaling at once.
_sel = select_one_series(df_series)
_row = _sel.iloc[0]
_dir = os.path.join(PATH_DATASET, "train_series", _row["StudyInstanceUID"], _row["SeriesInstanceUID"])
demo_volume = read_dicom_volume(_dir, n_slices=24, image_size=224)
# Fail fast: an all-zero volume means the decoders are missing, which would poison every batch silently.
assert float(demo_volume.abs().sum()) > 0, f"decoded only zeros from {_dir} — are the pylibjpeg plugins installed?"
print(f"study  : {_row['StudyInstanceUID']}")
print(f"plane  : {_row['Anatomical_Plane']} | fluid: {_row['Fluid_Sensitive']} | fat: {_row['Fat_Suppression']}")
print(f"volume : shape={tuple(demo_volume.shape)} dtype={demo_volume.dtype}")

# %%
# Three orthogonal cuts through the same array expose ordering errors that a single slice would hide.
_vol = demo_volume[:, 0].numpy()
_fig, _axes = plt.subplots(1, 3, figsize=(13, 4.5))
for _ax, _plane, _img in zip(
    _axes,
    ["acquisition plane (mid-slice)", "reconstruction along axis 1", "reconstruction along axis 2"],
    [_vol[_vol.shape[0] // 2], _vol[:, _vol.shape[1] // 2, :], _vol[:, :, _vol.shape[2] // 2]],
):
    _ = _ax.imshow(_img, cmap="gray", aspect="auto")
    _ = _ax.set_title(_plane)
    _ = _ax.set_xlabel("pixel")
    _ = _ax.set_ylabel("pixel / slice")
plt.tight_layout()
plt.show()

# %% [markdown]
# The mid-slice is legible knee anatomy and the reconstructions are continuous rather than striped, which confirms the
# `InstanceNumber` ordering is correct — a mis-ordered stack shows as banding in exactly those two panels. The
# normalisation check below confirms the value range the model will actually receive.

# %%
# Intensity statistics verify the normalisation contract the model's first BatchNorm implicitly assumes.
_flat = demo_volume.numpy().ravel()
print(f"min={_flat.min():.3f} max={_flat.max():.3f} mean={_flat.mean():.3f} std={_flat.std():.3f}")
print(f"zero-valued fraction (background): {(_flat == 0).mean():.1%}")
# Fail fast: a collapsed distribution means normalisation produced no usable contrast for the backbone.
assert float(_flat.std()) > 0.05, "intensity distribution collapsed — percentile normalisation is not working"
_interior = float(((_flat > 0.05) & (_flat < 0.95)).mean())
print(f"FINDING: {_interior:.0%} of voxels sit in the interior of [0, 1] rather than at the clipped extremes.")
print("IMPLICATION: per-volume percentile scaling is sufficient; no dataset-level mean/std statistics are needed.")

# %% [markdown]
# ### Extension — weak labels derived from the reports

# Hypothesis 2 showed that most training studies carry a report but no labels, and Hypothesis 5 showed those reports
# are multilingual. This subsection implements the obvious extension — keyword-matching the reports into weak labels —
# and then deliberately leaves it **switched off**.

# The honest framing matters more than the code:

# - The keyword map below is English-centric with a handful of German and Spanish synonyms. Every report in a language
#   it does not cover becomes an all-negative row, which is not "missing data" but *actively wrong* supervision.
# - Radiology reports are full of negation ("no evidence of ACL tear"), so a bare substring match inverts the label on
#   exactly the sentences that mention the finding. We apply a crude negation window, which reduces the error without
#   removing it.
# - Weak labels may only ever enter the *training* fold. Validation stays on human labels, or the AUC we optimise stops
#   measuring the thing the leaderboard measures.

# Turn `USE_PSEUDO_LABELS` on only after measuring the change against the clean validation fold.

# %%
# Feature flag first: the baseline result must be reproducible without any text-derived supervision.
USE_PSEUDO_LABELS = False
PSEUDO_NEGATION_WINDOW = 40
PSEUDO_LABEL_KEYWORDS = {
    "ACL": ["acl", "anterior cruciate", "vorderes kreuzband", "ligamento cruzado anterior"],
    "MCL": ["mcl", "medial collateral", "innenband", "ligamento colateral medial"],
    "Medial Meniscus": ["medial meniscus", "innenmeniskus", "menisco medial"],
    "Lateral Meniscus": ["lateral meniscus", "aussenmeniskus", "menisco lateral"],
    "Medial OA": ["medial compartment osteoarthritis", "medial osteoarthritis", "mediale gonarthrose"],
    "Lateral OA": ["lateral compartment osteoarthritis", "lateral osteoarthritis", "laterale gonarthrose"],
    "PF OA": ["patellofemoral osteoarthritis", "retropatellar", "femoropatelar"],
    "Effusion": ["effusion", "erguss", "derrame"],
    "Synovitis": ["synovitis", "synovialitis", "sinovitis"],
    "Baker's": ["baker", "popliteal cyst", "bakerzyste", "quiste de baker"],
    "Contusion": ["contusion", "bone marrow oedema", "bone marrow edema", "knochenmarksoedem"],
    "Fracture": ["fracture", "fraktur", "fractura"],
}
_NEGATIONS = ["no ", "not ", "without ", "absence of ", "kein", "keine", "sin ", "negative for "]
print(f"USE_PSEUDO_LABELS = {USE_PSEUDO_LABELS} (weak supervision is opt-in and never used for validation)")


# %% [markdown]
# The matcher below scans for each keyword and then inspects the characters immediately preceding the hit for a
# negation cue. It is intentionally simple and intentionally documented as lossy — its value is that it makes the
# noise mechanism visible to the reader rather than burying it inside a preprocessing script.


# %%
def weak_label_from_report(text, keywords, negation_window=PSEUDO_NEGATION_WINDOW):
    """Return 1 when a keyword appears without a nearby negation cue, else 0."""
    if not isinstance(text, str) or not text.strip():
        return 0
    low = text.lower()
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), low):
            prefix = low[max(0, match.start() - negation_window) : match.start()]
            if not any(neg in prefix for neg in _NEGATIONS):
                return 1
    return 0


# %%
# Building `df_labelled` is the hand-off from EDA to modelling: it is the exact table the DataModule will split.
df_labelled = df_train[df_train[TARGET_COLS].notna().all(axis=1)].copy()
df_labelled[TARGET_COLS] = df_labelled[TARGET_COLS].astype(int)
print(f"human-labelled studies used for training/validation: {len(df_labelled)}")
if USE_PSEUDO_LABELS:
    _unlabelled = df_train[~df_train[TARGET_COLS].notna().all(axis=1)].copy()
    for _col in TARGET_COLS:
        _unlabelled[_col] = _unlabelled["Report"].apply(
            lambda t, kw=PSEUDO_LABEL_KEYWORDS[_col]: weak_label_from_report(t, kw)
        )
    _unlabelled["is_weak"] = 1
    df_labelled["is_weak"] = 0
    df_labelled = pd.concat([df_labelled, _unlabelled], ignore_index=True)
    print(f"weak-labelled studies appended: {len(_unlabelled)} (training fold only — see the note above)")
else:
    df_labelled["is_weak"] = 0

# %% [markdown]
# ### EDA lens — decisions carried into the model

# Everything the modelling sections assume was measured above, not guessed. Collecting the decisions in one place makes
# the chain auditable: if a later result disappoints, this is the list of assumptions to revisit first.

# | Evidence | Decision |
# | --- | --- |
# | Twelve labels, imbalanced and correlated | One backbone, twelve logits, `BCEWithLogitsLoss`, macro AUROC |
# | All twelve labels project to <10 validation positives | Multilabel-stratified split, seeded |
# | Most studies have no labels, only reports | Train on fully labelled rows only; weak labels gated off |
# | Median ~5 series per study, ~30 slices per series | One selected series, 24 evenly sampled slices |
# | Sagittal fluid-sensitive not universal | Explicit fallback chain so every study yields a prediction |
# | Reports are multilingual | Text is a train-time labelling source only, never a test-time input |
# | Intensities vary by scanner | Per-volume percentile normalisation inside the Dataset |

# One limitation must be stated plainly rather than discovered later: **the schema exposes no `PatientID`**. Studies
# from the same patient — a follow-up scan of the same knee, or the contralateral knee — cannot be detected, so a truly
# patient-level split is impossible with the data as given. Our split is study-level and stratified; if the organisers
# later publish a patient mapping, swapping in a grouped splitter is a one-line change in the DataModule.

# %%
# A printed summary keeps the hand-off honest when the notebook is executed rather than read.
print(f"labelled studies for modelling : {len(df_labelled)}")
print(f"targets                        : {len(TARGET_COLS)}")
print(f"weak supervision enabled       : {USE_PSEUDO_LABELS}")
print(f"weak rows in table             : {int((df_labelled['is_weak'] == 1).sum())}")

# %% [markdown]
# ## 3. Dataset and DataModule

# This section turns the EDA decisions into code: a `Dataset` that maps one study to one fixed-shape tensor stack, and
# a `LightningDataModule` that owns the split. The division of labour is deliberate — the Dataset receives a table that
# has *already* been split and knows nothing about training versus validation, while the DataModule is the single place
# where any row is assigned to a fold. That way there is exactly one line of code that can leak data, and it is easy to
# find.

# The configuration below is the first place where compute budget becomes concrete. Twenty-four slices at 224×224 means
# each study costs 24 backbone forward passes, so an effective batch of 8 studies is really 192 images — which is why
# the batch size looks small for a classifier.

# %%
# Batch size counts STUDIES, not images: each study expands to N_SLICES backbone passes.
BATCH_SIZE = 8
IMAGE_SIZE = 224
N_SLICES = 24
VAL_FRACTION = 0.2
NUM_WORKERS = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
print(f"batch={BATCH_SIZE} studies | slices={N_SLICES} | px={IMAGE_SIZE} | images/batch={BATCH_SIZE * N_SLICES}")

# %% [markdown]
# ### The Dataset

# The Dataset accepts a table that already carries the chosen `SeriesInstanceUID` per study, so series selection
# happens once at setup rather than repeatedly inside workers. Passing `target_cols=None` produces a label-free
# variant for test inference — the same class, the same transforms, no hidden mode switch that could silently change
# preprocessing between training and submission.

# One domain-specific choice deserves emphasis: **we do not flip these images**. For a sagittal series the through-plane
# axis runs medial↔lateral, so reversing slice order would swap the `Medial OA` and `Lateral OA` labels, and an
# in-plane horizontal flip would exchange anterior and posterior anatomy. Both are standard augmentations that are
# quietly wrong for a laterality-specific label set, so augmentation is limited to intensity jitter.


# %%
class KneeSeriesDataset(Dataset):
    """Maps one study to a fixed-length stack of slices from its selected MRI series."""

    def __init__(
        self,
        df_studies,
        path_series_root,
        target_cols=None,
        n_slices=N_SLICES,
        image_size=IMAGE_SIZE,
        augment=False,
    ):
        """Store the pre-split study table and the decoding parameters shared by all splits."""
        self.df = df_studies.reset_index(drop=True)
        self.path_series_root = path_series_root
        self.target_cols = target_cols
        self.n_slices = n_slices
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        """Number of studies in this split."""
        return len(self.df)

    def __getitem__(self, idx):
        """Return the normalised slice stack, the optional label vector, and the study identifier."""
        row = self.df.iloc[idx]
        dir_series = os.path.join(self.path_series_root, row["StudyInstanceUID"], row["SeriesInstanceUID"])
        volume = read_dicom_volume(dir_series, n_slices=self.n_slices, image_size=self.image_size)
        if self.augment:  # intensity-only jitter — geometric flips would swap medial/lateral labels
            _scale = float(np.random.uniform(0.9, 1.1))
            volume = (volume * _scale + float(np.random.uniform(-0.05, 0.05))).clamp(0.0, 1.0)
        assert volume.shape == (self.n_slices, 1, self.image_size, self.image_size), f"bad shape {tuple(volume.shape)}"
        assert volume.dtype == torch.float32, f"bad dtype {volume.dtype}"
        sample = {"volume": volume, "StudyInstanceUID": row["StudyInstanceUID"]}
        if self.target_cols is not None:
            sample["labels"] = torch.tensor(row[self.target_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
        return sample


# %% [markdown]
# ### The split

# The splitter implements Hypothesis 1's conclusion. `MultilabelStratifiedKFold` balances label *combinations* across
# folds, which is what the co-occurrence heatmap told us we need — plain per-label stratification cannot represent "ACL
# tear together with effusion" as a stratum. The package is not part of the standard Kaggle image, so the import is
# guarded and a label-count-stratified split takes over when it is missing, with the chosen strategy printed rather
# than assumed.

# Note the fallback's own edge case: stratifying on a label count that occurs only once raises inside scikit-learn, so
# singleton counts are merged into a shared bucket first.


# %%
def stratified_split_indices(df, target_cols, val_fraction=VAL_FRACTION, seed=SEED):
    """Seeded multilabel-stratified train/validation indices, with a label-count fallback."""
    labels = df[target_cols].to_numpy(dtype=int)
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        n_splits = max(2, int(round(1.0 / val_fraction)))
        splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(df)), labels))
        strategy = f"MultilabelStratifiedKFold(n_splits={n_splits})"
    except ImportError:
        from sklearn.model_selection import train_test_split

        counts = pd.Series(labels.sum(axis=1))
        strata = counts.where(counts.map(counts.value_counts()) >= 2, -1)  # singleton strata break scikit-learn
        train_idx, val_idx = train_test_split(
            np.arange(len(df)), test_size=val_fraction, random_state=seed, stratify=strata
        )
        strategy = "label-count stratified (iterative-stratification unavailable)"
    return np.asarray(train_idx), np.asarray(val_idx), strategy


# %% [markdown]
# ### The DataModule

# The DataModule is the only component that sees all four tables at once, and it is the only one allowed to decide who
# goes where. It runs series selection for train and test, performs the seeded split on the labelled table, and builds
# loaders with explicit shuffling: on for training, off everywhere else so that validation scores and submission rows
# stay in a stable order.

# When weak labels are enabled they are filtered out of the validation fold after splitting, keeping the evaluation
# signal purely human-labelled — the invariant the extension section promised.


# %%
class KneeDataModule(pl.LightningDataModule):
    """Owns series selection, the seeded leakage-aware split, and all three dataloaders."""

    def __init__(
        self,
        df_studies,
        df_series_train,
        df_test,
        df_series_test,
        path_dataset=PATH_DATASET,
        target_cols=TARGET_COLS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    ):
        """Store the grounded tables and loader configuration without touching the filesystem."""
        super().__init__()
        self.df_studies = df_studies
        self.df_series_train = df_series_train
        self.df_test = df_test
        self.df_series_test = df_series_test
        self.path_dataset = path_dataset
        self.target_cols = list(target_cols)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.split_strategy = None

    def setup(self, stage=None):
        """Attach one series per study, split the labelled table, and instantiate the three datasets."""
        sel_train = select_one_series(self.df_series_train)[["StudyInstanceUID", "SeriesInstanceUID"]]
        table = self.df_studies.merge(sel_train, on="StudyInstanceUID", how="inner")
        train_idx, val_idx, self.split_strategy = stratified_split_indices(
            table, self.target_cols, self.val_fraction, self.seed
        )
        df_tr = table.iloc[train_idx].reset_index(drop=True)
        df_va = table.iloc[val_idx].reset_index(drop=True)
        if "is_weak" in df_va.columns:  # validation must stay human-labelled even when weak supervision is on
            df_va = df_va[df_va["is_weak"] == 0].reset_index(drop=True)
        root_train = os.path.join(self.path_dataset, "train_series")
        self.ds_train = KneeSeriesDataset(df_tr, root_train, target_cols=self.target_cols, augment=True)
        self.ds_val = KneeSeriesDataset(df_va, root_train, target_cols=self.target_cols, augment=False)
        # ==============================
        sel_test = select_one_series(self.df_series_test)[["StudyInstanceUID", "SeriesInstanceUID"]]
        table_test = self.df_test.merge(sel_test, on="StudyInstanceUID", how="left")
        # A test study absent from test_series.csv leaves NaN here; os.path.join would raise inside a worker and kill
        # the whole inference run. An empty string degrades to an empty glob -> zero volume -> a row still predicted.
        table_test["SeriesInstanceUID"] = table_test["SeriesInstanceUID"].fillna("")
        self.ds_test = KneeSeriesDataset(table_test, os.path.join(self.path_dataset, "test_series"), target_cols=None)

    def _loader(self, dataset, shuffle):
        """Build a dataloader with worker settings shared by every split."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def train_dataloader(self):
        """Shuffled loader over the training fold."""
        return self._loader(self.ds_train, shuffle=True)

    def val_dataloader(self):
        """Deterministic loader over the human-labelled validation fold."""
        return self._loader(self.ds_val, shuffle=False)

    def test_dataloader(self):
        """Deterministic, label-free loader over the test studies."""
        return self._loader(self.ds_test, shuffle=False)


# %% [markdown]
# ### Data lens — one real batch

# Nothing validates a data pipeline like pulling a batch through it. We check the tensor contract the model depends on:
# five dimensions ordered `(batch, slices, channel, height, width)`, float32 in `[0, 1]`, twelve float labels, and the
# study identifiers that the submission join will need.

# %%
# Building the DataModule prints which split strategy actually ran — assumed stratification is not stratification.
df_test = pd.read_csv(os.path.join(PATH_DATASET, "test.csv"))
df_series_test = pd.read_csv(os.path.join(PATH_DATASET, "test_series.csv"))
dm = KneeDataModule(df_labelled, df_series, df_test, df_series_test)
dm.setup()
print(f"split strategy : {dm.split_strategy}")
print(f"train studies  : {len(dm.ds_train)}")
print(f"val studies    : {len(dm.ds_val)}")
print(f"test studies   : {len(dm.ds_test)}")

# %%
# Reading one batch surfaces shape and dtype errors now, rather than as a cryptic failure inside the first epoch.
demo_batch = next(iter(dm.train_dataloader()))
print(f"volume : {tuple(demo_batch['volume'].shape)} {demo_batch['volume'].dtype}")
print(f"labels : {tuple(demo_batch['labels'].shape)} {demo_batch['labels'].dtype}")
print(f"ids    : {len(demo_batch['StudyInstanceUID'])} e.g. {demo_batch['StudyInstanceUID'][0]}")
print(f"range  : [{demo_batch['volume'].min():.3f}, {demo_batch['volume'].max():.3f}]")

# %% [markdown]
# The figure below shows consecutive slices of the first study in the batch — the exact pixels the backbone will see,
# after sampling, resizing, normalisation and jitter. If augmentation were corrupting the images, it would be visible
# here before a single gradient step is taken.

# %%
# Visualising post-transform slices is the last checkpoint before compute is spent on training.
_vol = demo_batch["volume"][0, :, 0].numpy()
_n_show = min(8, _vol.shape[0])
_fig, _axes = plt.subplots(1, _n_show, figsize=(2 * _n_show, 2.6))
for _i, _ax in enumerate(_axes):
    _ = _ax.imshow(_vol[_i * (_vol.shape[0] // _n_show)], cmap="gray")
    _ = _ax.set_title(f"slice {_i * (_vol.shape[0] // _n_show)}", fontsize=9)
    _ = _ax.axis("off")
_ = plt.suptitle(f"Post-transform slices — {demo_batch['StudyInstanceUID'][0][:24]}...")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Model — 2D backbone with attention-MIL pooling

# The modelling question for a slice stack is how to get from twenty-four per-slice embeddings to one study-level
# prediction. Three answers are common, and the trade-offs decide the architecture:

# | Approach | Strength | Why not here |
# | --- | --- | --- |
# | True 3D CNN | Through-plane context is native | No large pretrained weights; anisotropic spacing; heavy for 9 h |
# | 2D slices + mean/max pool | Trivial, uses ImageNet weights | Mean dilutes a two-slice lesion; max is noisy |
# | 2D slices + **attention MIL** | Pretrained weights, learns *which* slices matter | Slightly more parameters |

# Attention-based multiple-instance learning is the right inductive bias for this problem: an ACL tear is visible on a
# handful of slices out of twenty-four, and the gated attention head (Ilse et al., and the lineage of MRNet-style knee
# MRI models) learns to concentrate weight on them while remaining differentiable end-to-end. The attention weights are
# also returned, which makes the model inspectable — a rare property in medical imaging baselines.

# The head is a single linear layer producing twelve logits from the pooled embedding, matching the twelve independent
# binary decisions the metric scores.

# %%
# `efficientnet_b0` is the budget default: strong ImageNet features at ~5M params, so 24 slices/study fit.
MODEL_NAME = "efficientnet_b0"
MAX_EPOCHS = 30
LEARNING_RATE = 3e-4
PRETRAINED = True
print(f"backbone={MODEL_NAME} | epochs={MAX_EPOCHS} | lr={LEARNING_RATE} | pretrained={PRETRAINED}")

# %% [markdown]
# ### The model definition

# For this early EDA + baseline notebook the model lives inline — one cell, two classes, nothing emitted to disk. The
# eventual submission workflow (a code competition with internet disabled at scoring time) will need this class
# definition copied into the offline inference notebook alongside the trained checkpoint, but that packaging step is a
# concern for the submission notebook, not for the baseline.

# One consequence of that future split is worth encoding now: `PRETRAINED = True` is safe *here* because this notebook
# trains with internet on, so timm may fetch ImageNet weights. The offline inference notebook must construct the model
# with `pretrained=False` and let the checkpoint supply every weight. `load_from_checkpoint` does **not** do this by
# itself — it replays the saved hyperparameters, `pretrained=True` included, and timm would try to download ImageNet
# weights at reload time. The override has to be passed explicitly, which the inference section below demonstrates.

# A note on one detail in the cell below: the metric update casts targets to `long` — `MultilabelAUROC` rejects float
# targets outright, and our labels arrive as floats from NaN-capable pandas columns.

# %%
# Imports repeated here so the model cell stays self-contained and copy-pastable into the offline inference notebook.
import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torchmetrics.classification import MultilabelAUROC


class AttentionMILPooling(nn.Module):
    """Gated attention pooling over the slice axis of a per-slice embedding tensor.

    Implements the gated variant of attention-based multiple-instance learning: two parallel projections (a `tanh`
    value branch and a `sigmoid` gate branch) are combined and scored, then softmax-normalised across slices so the
    pooled embedding is a convex combination of slice embeddings.

    Args:
        in_features: Width of each slice embedding produced by the backbone.
        hidden_features: Width of the internal attention projection.
    """

    def __init__(self, in_features: int, hidden_features: int = 128) -> None:
        super().__init__()
        self.attn_v = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Tanh())
        self.attn_u = nn.Sequential(nn.Linear(in_features, hidden_features), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden_features, 1)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool slice embeddings into one study embedding.

        Args:
            feats: Tensor of shape `(batch, slices, in_features)`.

        Returns:
            Tuple of the pooled `(batch, in_features)` embedding and the `(batch, slices)` attention weights.
        """
        scores = self.attn_w(self.attn_v(feats) * self.attn_u(feats))
        weights = torch.softmax(scores, dim=1)
        return (feats * weights).sum(dim=1), weights.squeeze(-1)


class KneeMILClassifier(pl.LightningModule):
    """Per-slice timm backbone with attention-MIL pooling and a twelve-way multi-label head.

    Args:
        model_name: Any timm architecture name; instantiated with `num_classes=0` as a feature extractor.
        num_labels: Number of binary findings predicted per study.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: Decoupled weight decay for AdamW.
        pretrained: Whether timm should download ImageNet weights; must be False without internet access.
        max_epochs: Horizon for the cosine schedule.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_labels: int = 12,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        max_epochs: int = 8,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # `in_chans=1` adapts the ImageNet stem to single-channel MRI by summing the RGB filters.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
        self.pooling = AttentionMILPooling(self.backbone.num_features)
        self.head = nn.Linear(self.backbone.num_features, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_auroc = MultilabelAUROC(num_labels=num_labels, average="macro")
        self.val_auroc = MultilabelAUROC(num_labels=num_labels, average="macro")

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Map a batch of slice stacks to per-study logits.

        Args:
            volume: Tensor of shape `(batch, slices, 1, height, width)`.

        Returns:
            Logit tensor of shape `(batch, num_labels)`; apply `sigmoid` for probabilities.
        """
        batch, slices, channels, height, width = volume.shape
        feats = self.backbone(volume.view(batch * slices, channels, height, width)).view(batch, slices, -1)
        pooled, _ = self.pooling(feats)
        return self.head(pooled)

    def attention_weights(self, volume: torch.Tensor) -> torch.Tensor:
        """Return the per-slice attention weights for inspection.

        Args:
            volume: Tensor of shape `(batch, slices, 1, height, width)`.

        Returns:
            Attention tensor of shape `(batch, slices)` summing to one per study.
        """
        batch, slices, channels, height, width = volume.shape
        feats = self.backbone(volume.view(batch * slices, channels, height, width)).view(batch, slices, -1)
        return self.pooling(feats)[1]

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        """Shared train/validation step computing loss and updating the macro AUROC."""
        logits = self(batch["volume"])
        labels = batch["labels"]
        loss = self.loss_fn(logits, labels.float())
        metric = self.train_auroc if stage == "train" else self.val_auroc
        # MultilabelAUROC rejects float targets; pandas hands us floats because the columns are NaN-capable.
        metric.update(torch.sigmoid(logits), labels.long())
        # Epoch-level loss logging: with only a handful of batches per epoch, step-level logging thinned by
        # `log_every_n_steps` leaves holes in the training curve; one value per epoch is what the lens plots.
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=logits.size(0))
        self.log(f"{stage}/auroc", metric, on_step=False, on_epoch=True, prog_bar=True, batch_size=logits.size(0))
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step logging `train/loss` and `train/auroc`."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step logging `val/loss` and `val/auroc`, the checkpoint-selection metric."""
        return self._step(batch, "val")

    def configure_optimizers(self) -> dict:
        """AdamW with a cosine schedule annealed over the full training horizon."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# %% [markdown]
# ### Model lens — shape contract on synthetic input

# We verify the model's input/output contract with a synthetic tensor before wiring it to real data. A synthetic check
# isolates architecture bugs from data bugs: if this passes and training still fails, the fault is in the pipeline, not
# the network. It costs one forward pass on random tensors, so it runs before a single DICOM is decoded.

# %%
# A two-study synthetic batch confirms the (B,S,C,H,W) -> (B,12) contract and that attention normalises across slices.
model = KneeMILClassifier(
    model_name=MODEL_NAME,
    num_labels=len(TARGET_COLS),
    learning_rate=LEARNING_RATE,
    pretrained=PRETRAINED,
    max_epochs=MAX_EPOCHS,
)
_probe = torch.rand(2, N_SLICES, 1, IMAGE_SIZE, IMAGE_SIZE)
with torch.no_grad():
    _logits = model(_probe)
    _attn = model.attention_weights(_probe)
print(f"input   : {tuple(_probe.shape)}")
print(f"logits  : {tuple(_logits.shape)} (expected (2, {len(TARGET_COLS)}))")
print(f"attention sums to 1 per study: {torch.allclose(_attn.sum(dim=1), torch.ones(2), atol=1e-5)}")
print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 5. Training

# Training is where the metric choice becomes operational. Everything selects on `val/auroc` with `mode="max"`:
# checkpointing keeps the best-scoring epoch, early stopping waits for that same score to stall, and the cosine
# schedule anneals across the full horizon. Getting the direction wrong here is a silent failure — the run completes
# and the saved checkpoint is simply the worst one.

# Three practical settings for the 9 h budget:

# - **Mixed precision** on GPU roughly doubles throughput at this batch shape, which matters when each batch is 192
#   backbone passes; it falls back to full precision on CPU where fp16 would be slower, not faster.
# - **`FAST_DEV_RUN`** runs a single batch through train, validation and the callbacks. It is the cheapest way to catch
#   a pipeline error before committing hours of compute. It is exposed as a flag here and left off — a fast-dev run
#   writes neither a checkpoint nor `metrics.csv`, and the lens cells below assert both, so a flipped flag stops the
#   notebook instead of reporting on a run that never happened.
# - **Early stopping** on a patience of five epochs: with a small labelled subset, the backbone starts memorising
#   before the schedule ends, and the epochs saved are better spent on a larger backbone or more slices.

# %%
# Directions are set once and reused by both callbacks — `val/auroc` is the leaderboard metric, so higher is better.
FAST_DEV_RUN = False
MONITOR_METRIC = "val/auroc"
MONITOR_MODE = "max"
EARLY_STOP_PATIENCE = 5
PRECISION = "16-mixed" if torch.cuda.is_available() else "32-true"
print(f"monitor={MONITOR_METRIC} ({MONITOR_MODE}) | precision={PRECISION} | fast_dev_run={FAST_DEV_RUN}")

# %%
# An explicit filename matters: the monitored metric contains a slash, which Lightning would otherwise interpolate
# into the checkpoint name and turn into a nested directory.
logger = pl.loggers.CSVLogger(save_dir=PATH_OUTPUT, name="knee_mil_logs")
checkpoint_cb = pl.callbacks.ModelCheckpoint(
    dirpath=os.path.join(PATH_OUTPUT, "checkpoints"),
    filename="knee-mil-best",
    monitor=MONITOR_METRIC,
    mode=MONITOR_MODE,
    save_top_k=1,
    auto_insert_metric_name=False,
)
callbacks = [
    checkpoint_cb,
    pl.callbacks.EarlyStopping(monitor=MONITOR_METRIC, mode=MONITOR_MODE, patience=EARLY_STOP_PATIENCE),
    pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
]
print(f"logging to: {logger.log_dir}")

# %%
# `accelerator="auto"` keeps the same notebook runnable on Kaggle's GPU and on a CPU-only machine without edits.
trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",
    devices="auto",
    precision=PRECISION,
    logger=logger,
    callbacks=callbacks,
    fast_dev_run=FAST_DEV_RUN,
    log_every_n_steps=10,
)
print(f"trainer ready | max_epochs={MAX_EPOCHS} | accelerator={trainer.accelerator.__class__.__name__}")

# %%
# The single expensive call in the notebook — its inputs were asserted upstream, so any failure here is a real one.
trainer.fit(model, datamodule=dm)

# %%
# Printing the actual best path (not a guessed filename) is what the inference section and the checkpoint dataset need.
_best = checkpoint_cb.best_model_path
# Fail fast: the inference section reloads this exact path, so an empty one must stop the notebook here.
assert _best, "no checkpoint written — run training with FAST_DEV_RUN=False before the inference section"
print(f"best checkpoint : {_best}")
print(f"best {MONITOR_METRIC} : {float(checkpoint_cb.best_model_score):.4f}")

# %% [markdown]
# ### Training lens — reading the curves

# `CSVLogger` writes a plain `metrics.csv`, which we read back rather than trusting the progress bar. The plot shows
# training and validation loss together with validation AUROC: loss curves diagnose optimisation, and the AUROC curve
# is the only one that predicts leaderboard movement. A validation loss that improves while AUROC stalls means the
# model is sharpening probabilities it already ranks correctly — worth knowing before tuning further.

# %%
# Reading the logged CSV keeps the diagnosis reproducible after the session that produced it has ended.
_metrics_path = os.path.join(logger.log_dir, "metrics.csv")
assert os.path.isfile(_metrics_path), f"no metrics at {_metrics_path} — training must run with FAST_DEV_RUN=False"
df_metrics = pd.read_csv(_metrics_path)
print(f"logged columns: {list(df_metrics.columns)}")
display(df_metrics.dropna(axis=1, how="all").tail())
# Naming the best epoch here explains the in-memory-vs-reloaded gap the inference lens measures later.
_best_epoch = int(df_metrics.loc[df_metrics["val/auroc"].idxmax(), "epoch"])
_last_epoch = int(df_metrics["epoch"].max())
print(f"best val/auroc epoch : {_best_epoch} (training ran to epoch {_last_epoch})")

# %%
# Curves are plotted per epoch because AUROC is only defined once a full epoch of predictions has accumulated.
_agg = df_metrics.groupby("epoch").mean(numeric_only=True)
_fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
for _col in [c for c in ["train/loss", "val/loss"] if c in _agg.columns]:
    _ = sns.lineplot(x=_agg.index, y=_agg[_col], marker="o", ax=_axes[0], label=_col)
_ = _axes[0].set_xlabel("epoch")
_ = _axes[0].set_ylabel("BCE loss")
_ = _axes[0].set_title("Optimisation")
_ = _axes[0].grid(True)
_ = _axes[0].legend()
for _col in [c for c in ["train/auroc", "val/auroc"] if c in _agg.columns]:
    _ = sns.lineplot(x=_agg.index, y=_agg[_col], marker="o", ax=_axes[1], label=_col)
_ = _axes[1].set_xlabel("epoch")
_ = _axes[1].set_ylabel("macro ROC AUC")
_ = _axes[1].set_title("Competition metric")
_ = _axes[1].grid(True)
_ = _axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Inference

# Inference runs the twelve-logit model over the test studies and converts logits to probabilities with a sigmoid —
# and *only* a sigmoid. There is no argmax and no thresholding: ROC AUC is computed from the ranking of scores, so any
# hard decision we make discards exactly the information the metric rewards.

# The one thing that must not appear anywhere in this section is the report text. Reports exist in `train.csv` only;
# `test.csv` carries `StudyInstanceUID` alone. A model that consumed text would train beautifully and then have nothing
# to consume at scoring time. This is the practical meaning of the multimodal framing used earlier: **text is a
# train-time labelling source, images are the only test-time input**.

# We run both prediction paths the competition workflow needs — the in-memory model, and the model restored from the
# saved checkpoint — because the second is what the offline inference notebook will actually do. The two carry
# *different* weights whenever the best epoch is not the last one: `EarlyStopping` halts training but does not restore
# the best weights, so the in-memory model holds last-epoch weights while the checkpoint holds the best-`val/auroc`
# epoch. The reloaded model is therefore the one whose predictions we submit.


# %%
# Predictions carry their study identifier so the submission can join by key rather than trusting loader order.
def predict_studies(trained_model, loader, device):
    """Run sigmoid inference over a label-free loader, returning study identifiers and probabilities."""
    trained_model = trained_model.to(device).eval()
    uids, probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference"):
            logits = trained_model(batch["volume"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())  # only outputs return to CPU
            uids.extend(list(batch["StudyInstanceUID"]))
    stacked = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(TARGET_COLS)), dtype=np.float32)
    return uids, stacked


# %%
# Path 1 — the model already in memory; this is what a single-notebook submission would use.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
uids_mem, probs_mem = predict_studies(model, dm.test_dataloader(), DEVICE)
print(f"in-memory predictions: {probs_mem.shape} over {len(uids_mem)} studies")

# %%
# Path 2 — restore from disk exactly as the offline inference notebook must, from the path asserted above.
# `pretrained=False` overrides the saved hyperparameter: without it, `load_from_checkpoint` replays `pretrained=True`
# and timm re-downloads ImageNet weights — pointless here, a hard failure in the internet-disabled scoring run.
reloaded = KneeMILClassifier.load_from_checkpoint(_best, map_location=DEVICE, pretrained=False)
print(f"reloaded {type(reloaded).__name__} | params: {sum(p.numel() for p in reloaded.parameters()):,}")
uids_ckpt, probs_ckpt = predict_studies(reloaded, dm.test_dataloader(), DEVICE)
print(f"reloaded predictions: {probs_ckpt.shape}")

# %% [markdown]
# ### Inference lens

# Before anything is written to disk we check the properties a valid submission needs: the right number of rows, and
# values that are finite and inside `[0, 1]`. We also measure the gap between the in-memory (last-epoch) and reloaded
# (best-epoch) predictions: a near-zero gap means the best epoch was the final one, a large gap means early stopping
# ran past the best epoch — both are legitimate outcomes, and the print below says which one this run produced.

# %%
# These assertions are cheap; a malformed submission costs a day of leaderboard time.
# The reloaded model is the one the offline inference notebook runs, so its predictions are the ones we submit.
_uids, _probs = uids_ckpt, probs_ckpt
print(f"predictions : {_probs.shape} | ids: {len(_uids)} | unique ids: {len(set(_uids))}")
print(f"finite      : {bool(np.isfinite(_probs).all())}")
_in_range = bool((_probs >= 0).all() and (_probs <= 1).all())
print(f"range       : [{_probs.min():.4f}, {_probs.max():.4f}] within [0,1]: {_in_range}")
display(pd.DataFrame(_probs[: min(5, len(_probs))], columns=TARGET_COLS).round(4))
_delta = float(np.abs(probs_mem - probs_ckpt).max())
print(f"max |in-memory - reloaded| : {_delta:.2e}")
print(
    "-> matches: the best-val/auroc epoch was the final epoch, so both paths carry the same weights."
    if _delta < 1e-4
    else "-> differs: early stopping ran past the best epoch (EarlyStopping does not restore best weights);"
    "\n   the reloaded best-epoch predictions are the ones submitted."
)

# %% [markdown]
# ## 7. Submission

# The final step is mechanical but unforgiving. We start from `sample_submission.csv` rather than from our own
# predictions, because that file defines the authoritative row set and column order — including the twelve label names
# exactly as the grader expects them, spaces and the apostrophe in `Baker's` included. Predictions are joined onto it by
# `StudyInstanceUID`; any study our pipeline could not score (an unreadable series, a missing test series entry) keeps
# the neutral 0.5 default rather than vanishing from the file, because a missing row is a hard submission error while a
# neutral score merely contributes an uninformative ranking for that study.

# %%
# The sample submission is the contract: its rows and column order define what the grader accepts.
_path_sample = os.path.join(PATH_DATASET, "sample_submission.csv")
assert os.path.isfile(_path_sample), f"missing {_path_sample} — without the contract no submission can be assembled"
df_sample = pd.read_csv(_path_sample)
print(f"sample_submission shape: {df_sample.shape}")
display(df_sample.head())

# %%
# Joining by key (never by row order) is what makes the file correct even if the loader reorders studies.
df_pred = pd.DataFrame(_probs, columns=TARGET_COLS)
df_pred.insert(0, "StudyInstanceUID", _uids)
df_submission = df_sample[["StudyInstanceUID"]].merge(df_pred, on="StudyInstanceUID", how="left")
# An unscored study keeps the neutral 0.5 by design: a missing row is a hard error, an uninformative one is not.
_n_missing = int(df_submission[TARGET_COLS].isna().any(axis=1).sum())
df_submission[TARGET_COLS] = df_submission[TARGET_COLS].fillna(0.5)
df_submission = df_submission[["StudyInstanceUID", *TARGET_COLS]]
print(f"joined rows: {len(df_submission)} | rows falling back to 0.5: {_n_missing}")
display(df_submission.head())

# %%
# `index=False` — an extra unnamed index column is the single most common cause of a rejected submission.
df_submission.to_csv("submission.csv", index=False)
print(f"written: submission.csv ({len(df_submission)} rows)")

# %% [markdown]
# ### Submission lens

# The last gate re-reads the file from disk — not the DataFrame in memory — and checks it against every constraint the
# grader enforces: row count matching the sample, exact column names and order, unique identifiers covering the
# expected set, no NaN or infinity, and all values inside `[0, 1]`. Checking the written artifact rather than the
# in-memory object is the point: serialisation is where index columns and dtype surprises appear.

# %%
# Re-reading from disk verifies the artifact that will actually be graded.
_check = pd.read_csv("submission.csv")
_expected = ["StudyInstanceUID", *TARGET_COLS]
_rows_ok = "OK" if len(_check) == len(df_sample) else "MISMATCH"
print(f"rows            : {len(_check)} (sample: {len(df_sample)}) -> {_rows_ok}")
print(f"columns exact   : {list(_check.columns) == _expected}")
print(f"ids unique      : {_check['StudyInstanceUID'].is_unique}")
print(f"ids cover sample: {set(_check['StudyInstanceUID']) == set(df_sample['StudyInstanceUID'])}")
print(f"no NaN          : {not bool(_check[TARGET_COLS].isna().any().any())}")
print(f"all finite      : {bool(np.isfinite(_check[TARGET_COLS].to_numpy()).all())}")
_vals = _check[TARGET_COLS].to_numpy()
print(f"within [0,1]    : {bool((_vals >= 0).all() and (_vals <= 1).all())}")
display(_check.head())

# %%
# Final proof on the artifact itself: the header must read exactly as the grader expects, apostrophe included.
# ! head submission.csv
