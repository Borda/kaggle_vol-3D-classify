# Notebooks

Notebooks live here as **jupytext `py:percent` scripts** (`.py`), not `.ipynb`.
`.ipynb` is gitignored — generate it locally when needed (Kaggle/Colab upload, local Jupyter editing).

## Why

- diffable/reviewable in PR like normal code
- no stripped-output noise, no merge-conflict-prone JSON
- round-trips losslessly to `.ipynb` via jupytext

## Rules for writing/editing these scripts

1. **Keep the jupytext YAML header** at top of file (`# ---` ... `# ---` block with `jupytext:` and
   `kernelspec:` keys). Required for round-trip conversion — never hand-edit or delete it.
2. **Cell markers**: `# %%` for code cells, `# %% [markdown]` for markdown cells. One blank line
   after the marker before content. Don't hand-roll other jupytext styles (`light`, `sphinx`) in
   this repo — stick to `percent`.
3. **First cell = markdown title cell** (`# # Title` + short description). Kaggle requires a title;
   Colab shows it as the notebook header.
4. **Shell/magic commands** stay commented with a leading `# !` (e.g. `# ! pip install ...`) or
   `# %%bash` cell — this is jupytext's encoding of Jupyter `!`/`%` syntax so the file stays valid
   Python. Don't uncomment them for local runs; use a separate `if __name__ == "__main__"` path or
   run the underlying command directly in your shell instead.
5. **No absolute local paths.** Use relative paths or platform-detected roots so the same file runs
   on Kaggle (`/kaggle/input/...`, `/kaggle/working/...`), Colab (`/content/...` or mounted Drive),
   and locally.
6. **Install/setup cells** (pip installs, package uninstalls) go in their own leading code cell(s),
   kept idempotent and commented (`# !`) — platform-specific, not meant to execute during CI/lint.
7. **Don't commit outputs or execution counts** — there are none in `.py`, keep it that way; never
   paste raw notebook JSON into this directory.
8. `notebooks/**` is exempt from `F401`/`F811` in ruff (`pyproject.toml`) — unused imports and
   redefinitions across cells are expected in notebook-style code, don't fight the linter here.

## Converting

```bash
# .py -> .ipynb, e.g. to upload to Kaggle/Colab or open in classic Jupyter
python -m jupytext --to ipynb notebooks/your_script.py

# .ipynb -> .py (percent), e.g. after editing in Jupyter/Colab and downloading back
python -m jupytext --to py:percent your_script.ipynb

# open directly as a notebook (Jupyter, with jupytext installed as a server extension)
jupyter notebook notebooks/your_script.py
```

Pre-commit runs `jupytext --test-strict --to ipynb` on every `notebooks/*.py` to verify the file
still round-trips cleanly — a failure there means the percent-format markers got corrupted and the
file won't convert cleanly for Kaggle/Colab upload.
