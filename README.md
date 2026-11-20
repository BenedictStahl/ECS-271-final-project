# ECS-271-final-project
Final project for ECS 271, FQ 2026.

## Cameron's data exploration branch

Branch: `cam-data-exploration` (created from `main`). This workflow downloads and
profiles Defactify/MS COCOAI for pretrained-model input compatibility. It does
not train a detector or download model weights.

### Run on another Windows computer

Install Git and 64-bit Python 3.12. Open PowerShell and use a reasonably short
checkout path, for example `C:\src`. No CUDA installation is needed for this audit.
Allow about 10 GB for raw data and the environment; additional space is needed
if you also build Hugging Face Arrow caches or run the original notebook.

```powershell
git clone --branch cam-data-exploration https://github.com/BenedictStahl/ECS-271-final-project.git
cd ECS-271-final-project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\profile_dataset.py
```

For an existing clone, fetch and switch to the branch instead of cloning again:

```powershell
git fetch origin
git switch cam-data-exploration
```

Calling the venv's Python directly avoids PowerShell activation-policy issues.
`requirements-data.txt` pins the three direct dependencies used by these scripts;
it is not a complete lockfile for every transitive dependency. The existing
`requirements.txt` is for the broader notebook environment.

### Files and outputs

- `scripts/download_dataset.py`: downloads the dataset card and Parquet shards
  into `data/raw`, using immutable dataset revision
  `1c3b714c9691eb6dd11c2c1e92eab1ac29584fa0` by default. Writes a download manifest.
- `scripts/profile_dataset.py`: reads local Parquet batches directly with
  PyArrow, without network access or another full Arrow cache. Writes
  `data/dataset_compatibility_profile.json`.
- `scripts/dataset_io.py`: repository-relative defaults and shard completeness
  validation shared by the scripts.
- `scripts/dataset_overview.ipynb`: original team notebook, preserved unchanged.
- `tests/test_dataset_workflow.py`: small local-fixture regression tests; no
  dataset download required.

The data, generated reports, caches and `.venv` are Git-ignored. The original
one-off `data/profile_dataset.py` is superseded by the tracked script in
`scripts/`; it is not needed on the new machine. Research documents are supplied
separately in the handoff ZIP, not in this branch.

Both scripts resolve default paths relative to the repository, regardless of
the current working directory. Explicit relative option paths are relative to
your shell's current directory:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py --data-dir D:\datasets\defactify
.\.venv\Scripts\python.exe scripts\profile_dataset.py --data-dir D:\datasets\defactify --output data\my_profile.json
```

The profiler accepts either a download root containing `data/*.parquet` or the
shard directory itself. An older Hugging Face snapshot can be inspected by
passing its snapshot directory using `--data-dir`.

### What the audit establishes

It inspects every row's dimensions, aspect ratios, image format/mode, label pairs,
and source/split counts. By default it also fully decodes, converts to RGB, and
resizes the first five successful images per source per split (90 total for
this dataset). The JSON includes all size counts, schema, versions, errors,
runtime, and download provenance when a manifest is available.

This is not a full corruption scan, duplicate/leakage audit, or model inference
benchmark. Header readability does not prove every image fully decodes. The
profiler exits nonzero for image/label errors and retains the report for review.

Previously verified: 96,000 RGB JPEG rows, splits 42K/9K/45K; SD2.1 at 768 square,
SDXL/SD3 at 1024 square, DALL-E 3 mostly 270 square, Midjourney mostly 436 square,
and variable-sized real photos. Binary labels are imbalanced 5:1 fake:real.

### Troubleshooting and checks

- Internet access to Hugging Face is required for downloading. The dataset is
  public; an account token is generally unnecessary. Interrupted downloads can
  be resumed by rerunning the same command.
- The old downloader called `datasets.load_dataset(repo_id)` and failed on long
  Windows Arrow-cache filenames. This version downloads raw Parquet directly and
  the profiler creates no additional Arrow copy. A short checkout/data path is
  still advisable because Hub metadata has nested paths.
- `--revision main` intentionally fetches newer data. Omit it to reproduce the
  audited revision. The manifest records the resolved commit. Use a new data
  destination when switching revisions so leftover shards cannot be mixed.
- `--local-files-only` requests a download using only local files/cache. The
  profiler is always local-only and reports missing/incomplete shard sets.
- The original notebook has separate cache and working-directory assumptions;
  it will not automatically discover `data/raw`. These two Python scripts are
  the portable compatibility workflow.

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
