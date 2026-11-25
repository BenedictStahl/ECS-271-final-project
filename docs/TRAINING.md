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
