Historical working notes. Statements below describe earlier snapshots and are superseded by the current README and docs/STATUS.md.

# ECS-271-final-project
Final project for ECS 271, FQ 2026.

## Training environment

Use the existing Python 3.12 `.venv`, or create it as shown below. Install the
training dependencies and check the GPUs:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\check_training_env.py
.\.venv\Scripts\python.exe scripts\check_training_env.py --models
```

The requirements select matching PyTorch 2.11 / torchvision 0.26 CUDA 12.8 wheels
from the official PyTorch index, plus timm, Transformers and scikit-learn. An
NVIDIA driver is required; a separate CUDA toolkit installation is not required
for these wheel-based checks. These Windows CUDA pins are not a CPU/Mac setup.

`scripts/check_training_env.py` is a standalone setup check. It imports the
libraries/model classes and runs a small classifier forward/backward on every
visible GPU. `--models` additionally runs random-weight DeiT III, DINOv3 and
SigLIP vision Base/16 encoders at 224 pixels using FP16 on GPU 0. Use `--device 1`
to run that optional encoder check on GPU 1. Each should return two 768-dimensional
mean-patch feature vectors and pass the patch-score/logit identity check. This
verifies architecture/API operation, not the exact
pretrained checkpoints, normalization, accuracy, or throughput. No image data or
pretrained weights are downloaded by the checker.

The selected experiment uses DeiT III Base/16, DINOv3 Base/16, and SigLIP 2 Base/16
as frozen image encoders, with separately trained binary linear heads. The group's
replacement dataset is primary; original Defactify is an optional comparison. The fresh Python entry points below prepare
features and fit a head from an explicit split manifest. DeiT III and SigLIP 2
pretrained weights have passed real-image inference locally; Meta's DINOv3
checkpoint still requires access approval on Hugging Face.

## Cameron's data exploration branch

Branch: `cam-data-exploration` (created from `main`). This workflow downloads and
profiles Defactify/MS COCOAI for pretrained-model input compatibility. The newer
training entry points also download encoder weights and prepare a linear probe;
the original download/profile scripts remain data-only.

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

### Mean-patch features and the replacement dataset

All encoders now use final LayerNorm-normalized image patch tokens, followed by
mean pooling and one separately trained 768-to-1 linear head. DeiT excludes its
CLS token; DINOv3 excludes CLS and register tokens; SigLIP bypasses attention
pooling. Each image has 196 patch vectors (14x14) at 224 pixels. There is no
image-vector L2 normalization. Native normalization remains checkpoint-specific.

`train_probe.patch_scores` gives signed fake-evidence scores `w @ patch`, in
row-major 14x14 order on the CENTER-CROPPED image. Their mean plus the separate
bias equals the image logit; sigmoid is applied afterward. Scores are contextual
feature contributions, not causal pixel explanations. Train-only standardization
in `fit_probe.py` is folded into one affine layer, exported as `linear_head.npz`.

Image geometry is shared: EXIF orientation, RGB conversion, bicubic resize of the
shorter edge to 224, and center crop to 224. Heatmaps describe that crop, not the
uncropped photo. There are no random augmentations. This deliberately differs
from SigLIP's native square/bilinear resize. Different pretraining datasets and
recipes still prevent causal claims about pretraining objectives alone.

#### Prepare the local replacement snapshot

The loader expects `root/{train,validation,test}/{real,generator}/filename.jpg`
plus `metadata.csv`. It uses source names, not local numeric generator labels.
The audited `reference_map.jsonl` pairs each real reference with its Defactify
row ID and original caption. Future generators use the same reference filenames.
Keep the verified map with the dataset when moving to another computer.

```powershell
.\.venv\Scripts\python.exe scripts\replacement_dataset.py --data-dir ..\partial_new_image_set\extracted --reference-map ..\partial_new_image_set\inspection\reference_map.jsonl --output data\replacement-inventory-2026-11-23
```

This writes `index.jsonl` and `dataset_summary.json` under a NEW output directory.
The current snapshot has 48,000 rows: 16,000 real, 16,000 flux3, and 16,000
sdxl_turbo. Real images occur once. Do not concatenate Defactify and replacement
caches. All 32,000 generated files have matching metadata. The 4,168 metadata
records without delivered files are counted explicitly and are not indexed.

Each row records its supplied split, binary target, content hash, stable ID,
reference ID, original reference caption, conservative caption-group ID, and
assumed seed. Shared original captions flag 55 cross-split groups even where new
captions differ. These flags are not unique-photo IDs or automatic exclusions.
No held-out generator or final split manifest has been chosen. The index reader
verifies file hashes so modified images cannot silently reuse an old index.

`scripts/generation_provenance.json` records the supplied model repositories,
checkpoint filenames, ComfyUI generation-code commit, and assumed defaults.
Seed `42 + index` was reported by the operator; `--only-matched` OFF was confirmed.
Other committed defaults are assumptions, not recovered per-image run logs.
Descriptions use Qwen3-VL by default; image generation is text-only.

#### Apply the visual review and build shared holdouts

All 55 cross-split caption flags were visually reviewed. They link different
photographs across splits, so no image exclusions are required for those flags.
The review links the 34 inspected test-only augmentation families and assigns
photo-family IDs to the 265 reviewed references and their generated counterparts.
This is not an exhaustive near-duplicate audit of the dataset.

```powershell
.\.venv\Scripts\python.exe scripts\build_holdout_manifests.py --index data\replacement-inventory-2026-11-23\index.jsonl --review ..\partial_new_image_set\inspection\group_review\review_decisions.json --output data\replacement-holdouts-2026-11-23
```

The output contains one reviewed `index.jsonl`, its `dataset_summary.json`, the
hash-bound review, `manifest_summary.json`, and five `holdout-<generator>.json`
files. All encoders use these SAME files. Raw caption-group flags are preserved;
reviewed photo-family IDs replace caption identity only where the recorded visual
review authorizes it. Every generator sharing a reference inherits its family ID.
File hashes and review/index checksums prevent stale grouping decisions.

Each usable manifest excludes the held-out generator from train AND validation.
Test retains seen generators, the unseen generator, and the same real images,
with explicit disjoint cohort ID lists. Original split boundaries stay intact.
Current FLUX and SDXL holdouts have 14,000 train / 3,000 validation / 22,500 test
rows, and are marked `pilot_only`: each trains on just one available generator.
The other three holdouts have empty selections and a blocked status because their
images are unavailable. This is NOT yet the four-seen/one-unseen experiment.
`fit_probe.py` refuses pilot manifests unless `--allow-pilot` is explicitly given;
blocked manifests are always refused. Four explicit two-generator pilot heads have
now been fitted; see the pilot results below.

When all five generators cover the real reference set, regenerating the manifests
makes all five ready for the final protocol. Prepare a fresh inventory for the
later delivery, use a new output folder, and supply `--review-base-index` pointing
to the ORIGINAL inventory above. The builder verifies that all earlier image
identities, hashes, splits, and reference relationships remain unchanged before
carrying the review forward. Replacements/deletions require a new review binding.

Feature caches bind image IDs to content hashes and encoder/pooling settings.
The fitter may adopt newly reviewed grouping metadata without recomputing features
for unchanged images. It reports explicit held-out-versus-real and seen-versus-real
metrics, including real false-positive rate, alongside per-generator results.

#### Verify encoders and the XAI identity

```powershell
# All three architectures, random weights; no pretrained downloads.
.\.venv\Scripts\python.exe scripts\check_training_env.py --models
# Real pretrained checkpoints on TRAINING images from the prepared index.
.\.venv\Scripts\python.exe scripts\train_probe.py check --model deit3 --offline --replacement data\replacement-holdouts-2026-11-23\index.jsonl
.\.venv\Scripts\python.exe scripts\train_probe.py check --model siglip2 --offline --replacement data\replacement-holdouts-2026-11-23\index.jsonl
```

`check` uses a deterministic diagnostic linear head to test the score/logit math;
it does not fit a classifier. DINOv3's real weights still need approved Hub access
and authentication; its random-weight architecture check does not establish
pretrained accuracy or successful access. Remove `--offline` for first downloads.
If Hub transfer stalls, retry with `$env:HF_HUB_DISABLE_XET='1'` in PowerShell.

#### Later: cache and fit, after split review

Completed on 23 November 2026: `data/features/deit3-replacement-mean` and
`data/features/siglip2-replacement-mean` each contain 48,000 x 768 float32
mean-patch features. Each includes 16,000 real, 16,000 FLUX, and 16,000 SDXL-Turbo
images. All values are finite; row metadata/order exactly match the shared
reviewed index, and both available pilot manifests validate against each cache.
`data/features/replacement-mean-verification.json` records checksums and checks.
Four pilot heads were subsequently fitted. DINOv3 download access returned HTTP 401 with
no local Hub login; this does not establish the account's approval status.

```powershell
.\.venv\Scripts\python.exe scripts\train_probe.py cache --model deit3 --replacement data\replacement-holdouts-2026-11-23\index.jsonl --output data\features\deit3-replacement-mean
```

Use one process on one 3090 (`--device cuda:0`, default). Repeat with `siglip2` and
`dinov3`. Output directories must be new. `--limit 64` creates a diagnostic cache;
limited caches cannot train heads. Full caches retain mean vectors only; recompute
patch tokens for selected XAI examples rather than storing every token array.
Extraction restarts after interruption; there is no resume machinery.

The previous `data/features/deit3-defactify` and `siglip2-defactify` caches contain
CLS/native attention-pooled features. Preserve them as historical baselines;
they CANNOT be converted to mean-patch features, even for unchanged real images.
New cache provenance includes `pooling: mean_patch_v1`; fitting rejects old caches.

Each explicit JSON manifest contains `held_out_generator` and ID lists in
`splits.train`, `splits.validation`, and `splits.test`. Use the same manifest for
all encoders. The fitting script rejects cross-split byte/caption/reference/content
groups, moved published rows, and held-out generator contamination. It uses fixed
balanced logistic regression C=1 and threshold 0.5, with AUROC and balanced accuracy
reported overall and per generator versus real. The primary training selection
must contain new synthetic rows. Treat explicit pilot runs as debugging only; final evaluation waits for the complete snapshot and protocol. Partial fine-tuning remains optional future work.

### Two-generator debugging pilot (23 November 2026)

Four frozen-encoder heads completed under
`data/runs/pilot-two-generators-2026-11-23`: DeiT III and SigLIP 2, each trained
on SDXL-Turbo while holding out FLUX, then the reverse. All four converged;
train-only scaler statistics, saved predictions/metrics, exported affine heads,
and training-image patch-score reconstruction checks passed. Settings stayed
fixed at C=1 and threshold 0.5. No encoder fine-tuning was performed.

| Encoder | Training generator | Held-out generator | Seen balanced accuracy | Held-out balanced accuracy | Held-out AUROC |
|---|---|---|---:|---:|---:|
| DeiT III | SDXL-Turbo | FLUX | 94.54% | 74.52% | 0.9013 |
| DeiT III | FLUX | SDXL-Turbo | 94.23% | 79.55% | 0.9331 |
| SigLIP 2 | SDXL-Turbo | FLUX | 96.97% | 89.98% | 0.9757 |
| SigLIP 2 | FLUX | SDXL-Turbo | 97.03% | 88.75% | 0.9762 |

`PILOT_REPORT.txt` and `pilot_summary.json` in the run folder summarize results
and checks. Each run saves `metrics.json`, `head.pkl`, `linear_head.npz`,
`predictions.jsonl`, and `verification.json`. These are preliminary image-level
results, not the final five-generator ranking. Do not tune the final protocol
on pilot held-out scores. Test families are correlated; no confidence intervals
are claimed. DINOv3 and the other three generators are not included.

### Deterministic training augmentation (integrated 25 November 2026)

Benedict's `scripts/augment_image.py` was copied unchanged from origin/main commit
`4ebd24098f3f8f76fb63483c7cb26bafd316aef6`. It supplies scale/resampling,
contrast, Gaussian blur, and JPEG compression. `training_augmentation.py` handles
our view identities and provenance; `train_probe.training_view` is the shared
preprocessing callable for cached features and later image-based fine-tuning.

Order: EXIF/RGB -> common 224 center crop -> Ben augmentation -> restore 224x224
with bicubic if scaling changed dimensions -> checkpoint-specific normalization.
Thus blur is measured at a common working resolution. Scaling is a resampling
perturbation, not a zoom or a different field of view. Model code stays unchanged.

The default remains augmentation OFF. Existing images, original feature caches,
and validation/test preprocessing are unchanged. With `--augment-views N`, the
extractor writes ONLY N extra views per supplied TRAINING image, in a separate
cache. It does not re-encode original views or produce augmented image files.
Use a new folder per cache and the same recipe/view numbers across encoders.

```powershell
# Example for the existing snapshot; use the final reviewed index when ready.
.\.venv\Scripts\python.exe scripts\train_probe.py cache --model deit3 --offline --replacement data\replacement-holdouts-2026-11-23\index.jsonl --augment-views 1 --output data\features\deit3-replacement-aug1
# Combine original and extra-view caches; the shared manifest is unchanged.
.\.venv\Scripts\python.exe scripts\fit_probe.py --cache data\features\deit3-replacement-mean --cache data\features\deit3-replacement-aug1 --manifest data\replacement-holdouts-2026-11-23\holdout-flux3.json --output data\runs\deit3-aug1-pilot --allow-pilot
```

Frozen encoders can precompute views for all supplied training sources once.
The head fitter adds ONLY views whose original parent is in that holdout's
training list. Cached views from the held-out source are ignored. Each selected
parent must have original features and equal extra-view coverage. Validation
and test retain exactly their original IDs/features. References/families are
inherited, so views cannot become independent examples in another split.

Each parent retains total loss weight one: with one extra view, original and
augmented view each have weight 0.5. Both scaler fitting and classifier fitting
use these weights. This avoids silently doubling the image contribution merely
because it has two views. Class balancing and fixed C=1 remain in place.

Provenance records the imported module hash, settings, pipeline/recipe ID, view
count, stable full-parent-ID/view-number seed, transformation choices, and pixel
hash per view. Original-image SHA remains separate from augmented pixel SHA.
The fitter rejects incompatible recipes, forged identity/labels, missing parent
features, incomplete view coverage, and augmented validation/test rows. `--limit`
counts original source images, so N views and limit K produce N*K vectors; these
small diagnostic caches still cannot be used for project head training.

For fine-tuning, call `training_view(raw_pil_image, encoder_name, original_row,
view_number)` and pass its tensor to the encoder. The existing unaugmented
`image_transform` is used for original views and evaluation. Use the same fixed
views as the frozen comparison first; epoch-varying view numbers would be a
separate protocol decision. A partial-fine-tuning trainer is not implemented yet.

Verified: 31 tests pass, including an augmented fitting-CLI fixture, and both
pretrained available encoders generated matching view metadata/pixel hashes and
finite 4x768 diagnostic feature arrays. Those limited caches are under
`data/features/augmentation-smoke-2026-11-25`. No full augmented caches or new
project classifier fits have been run. Wait for the new snapshot integration
before choosing the full extraction jobs.


### Complete replacement dataset and local DINO (25 November 2026)

This supersedes the earlier incomplete-snapshot instructions. The active sources
are real, FLUX, Z-Image-Turbo, SDXL-Turbo, Qwen-Image-2.1, and `mage_flow`.
MiniMax was not delivered. Mage's exact checkpoint/settings remain unresolved;
its provenance and seed are unknown, rather than inherited from MiniMax.

The full index uses three extraction roots without copying images. Source names
control binary labels; the updated CSV's numeric source IDs are not reused as
binary labels. Supplied `raw/` and `generated/` prefixes are stripped only in the
index. Existing image IDs and hashes remain stable, permitting old mean-patch
features to be reused for unchanged images via the canonical reviewed index.
The root paths in dataset_summary.json are local: rebuild the index after moving
the dataset to another computer.

```powershell
.\.venv\Scripts\python.exe scripts\replacement_dataset.py --data-dir ..\partial_new_image_set --metadata ..\partial_new_image_set\metadata_updated.csv --image-root ..\partial_new_image_set\extracted --image-root ..\partial_new_image_set\z_image_turbo\extracted --image-root ..\partial_new_image_set\additional_delivery_2026-11-25\extracted --reference-map ..\partial_new_image_set\inspection\reference_map.jsonl --output data\replacement-inventory-2026-11-25
.\.venv\Scripts\python.exe scripts\build_holdout_manifests.py --index data\replacement-inventory-2026-11-25\index.jsonl --review data\review-decisions-2026-11-25.json --review-base-index data\replacement-inventory-2026-11-23\index.jsonl --output data\replacement-holdouts-2026-11-25
```

These commands create new snapshots and refuse to overwrite existing outputs.
The original 55 caption-group review decisions are preserved. Two extra caption
pairs became equal after whitespace trimming. Visual review of the four real
photos established distinct photographs; the added hash-bound decisions and
contact sheet are in data/review-decisions-2026-11-25.json and
data/additional-caption-review-2026-11-25.jpg. No images were excluded.
This is not an exhaustive perceptual-duplicate audit.

All five candidate holdouts are ready: each has 35,000 training images, 7,500
validation images, and 45,000 test images. The held-out generator is absent
from training and validation. Test cohorts contain 7,500 real, 7,500 unseen fake,
and 30,000 seen fake images. No final held-out generator has been selected.

The verified DINO delivery can be used directly without a Hugging Face login:

```powershell
.\.venv\Scripts\python.exe scripts\train_probe.py check --model dinov3 --offline --local-checkpoint ..\dino_inspection_2026-11-25\dinov3-base --replacement data\replacement-holdouts-2026-11-25\index.jsonl
```

`--local-checkpoint` verifies the selected repository/revision plus config,
weights, and license hashes against MANIFEST.json before an offline strict load.
The same option works for cache extraction and augmentation. This validates
against the supplied manifest; it does not independently query remote hashes.

To repeat the tiny end-to-end check, choose a NEW output directory:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_training.py --manifests data\replacement-holdouts-2026-11-25 --local-dino ..\dino_inspection_2026-11-25\dinov3-base --output data\runs\full-delivery-smoke-2026-11-25
```

The smoke script uses four references per published split and all six sources:
72 original images total plus 24 deterministic training views per encoder.
For every encoder it checks all five candidate holdouts. Each tiny head trains
on 20 original images plus 20 views, validates on 20, and tests on 24. Four
cached views belonging to each held-out generator must be excluded from fitting.
Validation/test have no augmentation. Augmented pixel hashes must agree across
all three encoders. Runs are explicitly marked pilot_only and are pipeline
checks, not scientific model comparisons. Check verification.json for completion;
no full-dataset extraction or encoder fine-tuning is performed by this script.

Completed verification: 33 unit tests passed. All three encoders produced finite
72x768 original features and 24x768 training-view features, with identical view
pixel hashes across models. All 15 tiny heads passed holdout/view-selection and
export checks. Completion reports: data/training-setup-verification-2026-11-25.json
and data/runs/full-delivery-smoke-2026-11-25/verification.json. No full extraction
or fine-tuning was started.


### Completing and reusing frozen features

`train_probe.py cache --reuse-cache EXISTING_FOLDER` validates the encoder,
pooling, transform, full-cache status, image IDs/labels/hashes and feature array,
then encodes only original IDs absent from that cache. It requires --replacement
and cannot be combined with --augment-views. The new cache holds only missing
rows; keep the old and new folders and pass both to fit_probe.py. Existing mean
features remain valid when metadata grouping is refreshed through the canonical
index. Old CLS/native-pooling caches remain incompatible.

The full snapshot extraction destination is data/features/full-replacement-2026-11-25.
For DeiT III and SigLIP 2, combine the existing MODEL-replacement-mean folder
(48,000 rows) with MODEL-original in that destination (48,000 new rows).
DINO's dinov3-original folder contains all 96,000 original rows. MODEL-aug1
contains 42,000 extra training views, including views that the selected holdout
manifest will exclude from fitting. These are vectors, not augmented image files.

After verification.json reports passed, fit a selected head using three caches
for DeiT III/SigLIP 2 (old originals, missing originals, augmented views), or two
for DINO (all originals, augmented views). Example only; select the final holdout
before executing a scientific training run:

```powershell
.\.venv\Scripts\python.exe scripts\fit_probe.py --cache data\features\deit3-replacement-mean --cache data\features\full-replacement-2026-11-25\deit3-original --cache data\features\full-replacement-2026-11-25\deit3-aug1 --manifest data\replacement-holdouts-2026-11-25\holdout-CHOSEN_GENERATOR.json --output data\runs\deit3-CHOSEN_GENERATOR-aug1
```

Do not choose the holdout by comparing its test performance. Feature extraction
uses frozen inference only and does not choose a holdout or train a classifier.

Full extraction completed and verified on 25 November 2026. Every encoder now
has 96,000 original-image vectors plus 42,000 deterministic training-view
vectors (all float32, 768 dimensions, finite). All five holdout manifests pass
against each cache set: 35,000 training originals plus 35,000 extra views, with
7,000 held-out-source views excluded. Validation/test stay at 7,500/45,000.
The 42,000 augmented pixel hashes agree exactly across all three encoders.
Fresh inference on one training image per source per encoder (18 checks)
reproduced the saved original vectors exactly, including reused vectors; patch
score reconstruction also passed. See verification.json and raw-image-spot-check.json
in the extraction destination. No classifier training or fine-tuning was run.

### Primary holdout selected: Qwen-Image-2.1-Turbo

On 25 November 2026, Cameron selected `qwen_image_2_1` as the primary held-out
generator for all three encoders and both frozen-head and fine-tuned comparisons.
The authoritative selection is training_protocol.json; use
data/replacement-holdouts-2026-11-25/holdout-qwen_image_2_1.json. Training uses real
images plus FLUX, Z-Image-Turbo, SDXL-Turbo and mage_flow. Qwen is absent from
training and validation, including augmented views. The prepared manifest has
35,000 training originals, 7,500 validation images and 45,000 test images.
One extra view doubles training feature rows to 70,000 while preserving each
original parent's total weight. The 7,000 cached Qwen training views are excluded.
Other candidate manifests do not change this primary selection. Earlier text
saying the final holdout is undecided is superseded by this decision.

### Frozen Qwen-holdout heads trained (25 November 2026)

All three frozen heads completed sequentially with four CPU threads and the
fixed C=1 balanced logistic-regression configuration. Each fit used 35,000
original images plus 35,000 augmented views; Qwen is absent from training and
validation. The runs are under data/runs/frozen-qwen-holdout-2026-11-25.
Training converged in 92/132/97 iterations for DeiT III/SigLIP 2/DINOv3. Fits
took roughly 3–4 seconds each after loading/validation of the cached features.
Seen-generator validation BA/AUROC: DeiT III 0.90750/0.963956, SigLIP 2
0.969833/0.995343, DINOv3 0.97050/0.995945. These do not measure Qwen
generalization. Test evaluation was deferred with --defer-test; no test
predictions were computed. All 33 tests pass, including deferred-test output.

Each encoder folder contains head.pkl (scaler+classifier), linear_head.npz
(folded raw-feature coefficients for the common head/XAI), validation predictions,
metrics, and verification. Train-only weighted scaling, exported logits,
serialized predictions and holdout exclusions passed independent checks.
No encoder fine-tuning has run. Josh’s fine-tuning configuration is still pending.

To reproduce a head, use the earlier cache arguments with the selected Qwen
manifest and add --defer-test --cpu-threads 4. Choose a fresh output directory.
The --defer-test option still validates test split identities but never computes
test predictions or metrics. Convergence warnings stop a run instead of saving
a nonconverged head as successful.
