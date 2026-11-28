# Training workflow

Run commands from the repository root in PowerShell. Use fresh output folders.
`training_protocol.json` records Qwen as primary holdout and binds the existing
manifest/index hashes. Rebuilding indexes on another computer changes local path
metadata and may change those hashes; review and update bindings explicitly.

## Exact encoders and common head

| Encoder | Repository | Pinned revision |
|---|---|---|
| DeiT III Base/16 | timm/deit3_base_patch16_224.fb_in22k_ft_in1k | b5791d4983082b0c680da556954beb5b1962e85f |
| DINOv3 Base/16 | facebook/dinov3-vitb16-pretrain-lvd1689m | 5931719e67bbdb9737e363e781fb0c67687896bc |
| SigLIP 2 Base/16 | google/siglip2-base-patch16-224 | 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 |

General-purpose pretrained weights, not pretrained forensic classifiers. The full
encoder produces final LayerNorm-normalized patch vectors. Exclude DeiT's CLS and
DINO's CLS/register tokens; bypass SigLIP attention pooling. Average 196 x 768
patch vectors into one 768-vector. No image-vector L2 normalization. Each encoder
has its own trained linear weights; architecture is shared, weights are not.

Common geometry: EXIF orientation -> RGB -> short edge 224, bicubic/antialias ->
224 center crop. DeiT/DINO use ImageNet mean/std; SigLIP uses 0.5/0.5.
Common geometry deliberately differs from SigLIP's native square/bilinear resizing.

StandardScaler and balanced LogisticRegression(C=1, max_iter=2000, solver=lbfgs)
fit training features only. Prediction threshold stays 0.5. Train-only scaling is
folded into linear_head.npz (weight[768], scalar bias). Apply that head directly
to raw mean-patch vectors; do not standardize again. Keep head.pkl for the complete
scaler/classifier pipeline. Convergence warnings stop fitting as an error.

## Dataset index and reviewed splits

The original reference map, review decisions and original reviewed inventory
must be transferred with the dataset; they are not in Git. Images stay in three
roots without copying. Source names determine binary labels. Updated raw/ and
generated/ prefixes are normalized only in the index; supplied CSV stays unchanged.

```powershell
.\.venv\Scripts\python.exe scripts\replacement_dataset.py --data-dir ..\partial_new_image_set --metadata ..\partial_new_image_set\metadata_updated.csv --image-root ..\partial_new_image_set\extracted --image-root ..\partial_new_image_set\z_image_turbo\extracted --image-root ..\partial_new_image_set\additional_delivery_2026-11-25\extracted --reference-map ..\partial_new_image_set\inspection\reference_map.jsonl --output data\replacement-inventory-NEW
.\.venv\Scripts\python.exe scripts\build_holdout_manifests.py --index data\replacement-inventory-NEW\index.jsonl --review data\review-decisions-2026-11-25.json --review-base-index data\replacement-inventory-2026-11-23\index.jsonl --output data\replacement-holdouts-NEW
```

The additive review checks earlier image IDs, hashes, splits and reference
relationships. Real images occur once. Do not concatenate original Defactify
features to replacement features. The primary existing manifest is
`data/replacement-holdouts-2026-11-25/holdout-qwen_image_2_1.json`.

## Local DINO and feature extraction

The verified local DINO folder contains config.json, model.safetensors, LICENSE.md
and MANIFEST.json. --local-checkpoint checks the repository/revision, file sizes
and hashes before strict offline loading; no Hub login is needed. Hashes are
verified against the supplied manifest, not independently fetched remote hashes.
For first DeiT/SigLIP downloads omit --offline; later runs can use their local cache.

```powershell
.\.venv\Scripts\python.exe scripts\train_probe.py check --model dinov3 --offline --local-checkpoint ..\dino_inspection_2026-11-25\dinov3-base --replacement data\replacement-holdouts-2026-11-25\index.jsonl
.\.venv\Scripts\python.exe scripts\train_probe.py cache --model deit3 --offline --replacement data\replacement-holdouts-2026-11-25\index.jsonl --reuse-cache data\features\deit3-replacement-mean --output data\features\deit3-missing-NEW
.\.venv\Scripts\python.exe scripts\train_probe.py cache --model dinov3 --offline --local-checkpoint ..\dino_inspection_2026-11-25\dinov3-base --replacement data\replacement-holdouts-2026-11-25\index.jsonl --output data\features\dinov3-original-NEW
.\.venv\Scripts\python.exe scripts\train_probe.py cache --model deit3 --offline --replacement data\replacement-holdouts-2026-11-25\index.jsonl --augment-views 1 --output data\features\deit3-aug1-NEW
```

--reuse-cache validates compatibility/IDs/hashes and skips existing original IDs.
Its output contains missing rows only; pass both old and new caches to fitting.
It cannot be combined with augmentation. Old CLS/native-pooling caches cannot be
converted or reused. --limit creates diagnostic caches that fitting refuses.
Use --device cuda:0 or cuda:1 for independent extraction jobs. Extraction saves
mean vectors only; patch tokens can be recomputed for XAI. There is no mid-cache resume.

## Deterministic train-only augmentation

Ben's augment_image.py was copied unchanged from origin/main commit
4ebd24098f3f8f76fb63483c7cb26bafd316aef6. It selects scale/resampling, contrast,
Gaussian blur and JPEG compression; at least one is applied. Training order:
common crop -> Ben augmentation -> restore 224 square with bicubic -> native normalization.
Scaling is a resampling perturbation, not a zoom into a different field of view.

training_augmentation.py derives stable seeds from full parent ID/view number and
records module/recipe hashes, choices and augmented pixel hashes. All encoders use
the same view pixels. Validation/test use original deterministic preprocessing.
--augment-views 1 writes only additional views for published training images.
The fitter selects only views whose parents are in the chosen training manifest;
it excludes Qwen views, requires uniform coverage and original parent vectors,
and gives original/view weight 0.5 each (total one per parent) in scaler and head fitting.

For fine-tuning reuse training_view(image, encoder_name, original_row, view_number)
and the same fixed views first. Any epoch-varying policy is a separate decision.

## Reproduce a frozen head

The completed feature caches are listed in DATA_LAYOUT.md. Example with the existing
DeiT caches; repeat for SigLIP, substituting its cache names. DINO uses its single
original cache plus dinov3-aug1, without an older replacement-mean cache.

```powershell
.\.venv\Scripts\python.exe scripts\fit_probe.py --cache data\features\deit3-replacement-mean --cache data\features\full-replacement-2026-11-25\deit3-original --cache data\features\full-replacement-2026-11-25\deit3-aug1 --manifest data\replacement-holdouts-2026-11-25\holdout-qwen_image_2_1.json --output data\runs\deit3-qwen-NEW --defer-test --cpu-threads 4
```

--defer-test is essential for the current stage: test identities are validated,
but no test predictions/metrics are computed. Without this flag the script also
evaluates test. Frozen-head fitting is CPU-only; four threads avoids contention.
The script saves metrics/config, head.pkl, linear_head.npz and predictions.jsonl.
Existing primary runs also have independent verification.json files and a shared
summary.json. Do not overwrite them or tune against their eventual test results.

## Pipeline checks

```powershell
.\.venv\Scripts\python.exe scripts\check_training_env.py --models
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\smoke_training.py --manifests data\replacement-holdouts-2026-11-25 --local-dino ..\dino_inspection_2026-11-25\dinov3-base --output data\runs\smoke-NEW
```

The environment model check uses random weights, not pretrained checkpoints.
The smoke script uses 72 originals plus 24 training views per encoder and fits
all five candidate heads; it evaluates tiny validation/test subsets. It is a
pipeline diagnostic, not a primary benchmark or a holdout-selection procedure.

## XAI interface

For a crop's final patch tokens, patch score = weight dot patch. The mean of the
14x14 score grid plus bias equals the image logit before sigmoid. The bias is
separate. Heatmaps cover the processed crop; tokens incorporate image-wide context,
so scores are not causal pixel explanations. Keep the same pooling/head for the
first fine-tuning comparison to isolate adaptation effects.


## Approved initial fine-tuning recipe (27 November 2026)

Configuration: finetuning_initial.json. Unfreeze the last two transformer blocks,
final normalization, and the existing raw-feature linear head. Encoder/head LRs
are 1e-5/1e-4; AdamW decay is 0.01 on multidimensional weights and zero on
biases/one-dimensional parameters (including normalization and layer scales).
Batch 32; up to five epochs; two epochs without seen-validation AUROC improvement
stops training. Warmup is ceil(5% of the planned optimizer steps), then cosine
decay. Clip global gradient norm to 1. FP16 encoder autocast with gradient scaling;
head/loss stay FP32. Seed 42. These settings are initial choices, not tuned optima.

All frozen prefix/embedding modules remain in eval mode. Only the chosen final
blocks, final norm and head enter training mode. Frozen parameters receive no
gradients. Each of the 35,000 approved parents has an original and the same fixed
view 1 used by the frozen baseline: 70,000 views/epoch. Loss gives each view
weight 0.5 and normalizes by the sum of parent weights, with class weights
N/(2*n_class) from original training counts (real=2.5, fake=0.625). No sampler
rebalancing is added. Validation uses 7,500 original images; Qwen/test are never
opened or scored by this trainer. Index/review/manifest and baseline head hashes
are checked. Each image is hash-checked as it is loaded.

Smoke and full-training actions are explicit and require NEW output folders:

```powershell
.\.venv\Scripts\python.exe scripts\finetune.py smoke --model deit3 --device cuda:0 --output data\runs\finetune-smoke-NEW\deit3
.\.venv\Scripts\python.exe scripts\finetune.py smoke --model dinov3 --device cuda:1 --local-dino ..\dino_inspection_2026-11-25\dinov3-base --output data\runs\finetune-smoke-NEW\dinov3
# Full-run examples. Current runs use run_initial_finetunes.py with snapshot inputs.
.\.venv\Scripts\python.exe scripts\finetune.py train --model deit3 --device cuda:0 --output data\runs\finetune-initial-NEW\deit3
.\.venv\Scripts\python.exe scripts\finetune.py train --model siglip2 --device cuda:1 --output data\runs\finetune-initial-NEW\siglip2
# Start DINO once a GPU is free.
.\.venv\Scripts\python.exe scripts\finetune.py train --model dinov3 --device cuda:0 --local-dino ..\dino_inspection_2026-11-25\dinov3-base --output data\runs\finetune-initial-NEW\dinov3
```

Defaults: two Windows-safe DataLoader workers, four main-process CPU threads,
o network downloads (offline checkpoint loading). --workers 0 can help debug
worker issues. --config selects a separately named recipe for future approved
experiments. --protocol and --baseline-root allow explicit transferred inputs.
There is no CLI resume. Do not overwrite or train from smoke diagnostics.

Completed smoke tests selected 32 training references across five included
sources (160 parents, 320 available views), shuffled and processed six batches
of 32, and evaluated eight matched validation references (40 images). The smoke
warmup/cosine schedule is compressed into six steps to exercise the schedule;
its trajectory is not the first six steps of a scientific five-epoch run.
Initial validation predictions matched the frozen baselines within 8e-7;
frozen parameter hashes stayed unchanged; both selected blocks, final norm and
head changed; gradients/loss stayed finite; reload into fresh bases reproduced
predictions; and XAI patch-score identities passed. Timing extrapolates only
four updates after two warmups and can be optimistic because of prefetch/cache
and short validation samples. Allocated tensor peaks around 1 GiB do not include
all reserved/driver/desktop memory. Measure the first full epoch for scheduling.

Full runs save configuration.json, history.json, best.pt and last.pt. Best is
selected strictly by seen-validation AUROC; best validation predictions and
linear_head.npz are saved when best improves. Best/last checkpoints contain only
trainable parameter deltas (plus optimizer/scheduler/scaler state and metadata),
not duplicate frozen encoder weights. Use the exact original base to reconstruct.
The standalone head must be paired with the corresponding fine-tuned encoder;
it is not interchangeable with the original frozen encoder.

For inference/XAI from a fine-tuned checkpoint:

```python
import torch
from finetune import load_finetuned  # add scripts/ to your import path
model = load_finetuned("path/to/best.pt", torch.device("cuda:0"),
                       local_dino="path/to/dinov3-base")  # DINO only; omit for other encoders
model.eval()
# Use train_probe.image_transform(model.name), then model(image_batch) inside
# inference_mode + CUDA FP16 autocast as in finetune.evaluate.
# For patch maps use model.encoder and model.head with the existing patch_scores.
```

load_finetuned validates checkpoint format, selected base revision, parameter
names and shapes. It needs the original checkpoint cache/local DINO folder but
does not need the old baseline head or full feature caches for inference.

## Active run launcher and live timing

run_initial_finetunes.py starts DeiT on GPU0 and SigLIP on GPU1, then automatically
queues DINO on the first free GPU. It snapshots the recipe/protocol and records
commands, PIDs and exit status. The completed initial output is
data/runs/finetune-initial-2026-11-27-r2; see INITIAL_FINETUNING_RUN_2026-11-27.md.
finetune.py writes progress.json every 100 successful updates with recent
throughput and a five-epoch remaining-time estimate, and per-epoch timing in history.json.

FP16 gradient overflow is recoverable: GradScaler skips that update, reduces
loss scale and the same batch is retried (up to eight attempts). No batch is
dropped, and the scheduler advances only after success. Missing gradients,
nonfinite loss, nonfinite gradients without AMP, or repeated failed recovery
still stop explicitly. Retry events are written to amp_overflow_retries.json.
The first full-run attempt exposed an overly strict fatal-gradient guard; its
logs were retained and both first jobs were restarted cleanly after this correction.

## Completed grid and test handoff

All 12 grid runs are complete. See FINETUNING_GRID_2026-11-27.md for the
configuration matrix and results/finetuning-grid-2026-11-27 for all epoch histories
and locked selections. The original live run launcher descriptions above describe
execution, not currently running jobs. EVALUATION_HANDOFF.md explains portable
verification and explicit test evaluation. Fine-tuned inference files exclude
optimizer state; original best.pt/last.pt training artifacts remain local.
