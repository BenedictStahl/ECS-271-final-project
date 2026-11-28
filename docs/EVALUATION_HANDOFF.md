# Locked-model test evaluation handoff

The 12-run grid is complete. For each encoder, the highest seen-generator validation
AUROC selects configuration C: final four blocks, encoder LR 1e-5, head LR 1e-4.
The best epoch is selected independently per encoder. This choice was made before
full test evaluation. Frozen-head baselines are included for comparison.

## Files to send privately

Clone the `cam-finetuning-evaluation` branch and obtain both transfer archives:

- `evaluation-model-assets.zip`: three selected inference checkpoints, frozen heads,
  exact original encoder weights, selection records, test index and hashes.
- `evaluation-test-images.zip`: all 45,000 original test JPEGs, with their original
  bytes. No augmentation and no training images are required.
- `SHA256SUMS.txt`: archive checksums; `README_FIRST.md`: these instructions.

The archives are intentionally outside Git. They are several GB, exceeding normal
GitHub file limits. They contain no credentials. Share within the project team;
retain the supplied DINO license and original model manifests.

The dataset archive is optional only if the recipient reconstructs the exact
`data/evaluation-handoff/images/test/<source>/<filename>.jpg` layout from their own
copy and passes verification. A public Drive folder alone is not a local dataset.

## Windows setup

From PowerShell in the cloned repository, with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
Expand-Archive -LiteralPath C:\path\to\evaluation-model-assets.zip -DestinationPath .
Expand-Archive -LiteralPath C:\path\to\evaluation-test-images.zip -DestinationPath .
.\.venv\Scripts\python.exe scripts/evaluate_handoff.py verify
```

Use a fresh clone/destination; the archives extract paths relative to the repository
root. The pinned Windows CUDA wheels also execute on CPU without an NVIDIA GPU.
The archives include every encoder base, including DINO; no Hugging Face login,
token, download, training cache, or original training machine path is needed.
Allow roughly 8 GB for the archives plus another 8 GB extracted, before installing
the environment. Keep the computer awake during evaluation.

## Explicit test evaluation

Verification hashes all inputs but does not run inference or compute test scores.
The following command deliberately opens the test set for evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_handoff.py run --model all --kind both --device cpu --batch-size 16 --output data/evaluation-results
```

For an NVIDIA GPU, use `--device cuda:0` instead. Models execute sequentially.
CPU evaluation is supported but may take many hours; GPU timing depends on the
hardware. To divide the work, use `--model deit3`, `siglip2`, or `dinov3`, each with
a separate new output directory. `--kind finetuned` evaluates only fine-tuned models;
`--kind frozen` evaluates only the existing frozen baselines. The default `both`
performs six passes of 45,000 images. No model fitting occurs.

Each model/kind exports per-image probabilities and reference/content family IDs,
plus AUROC, balanced accuracy, real false-positive rate and confusion matrices for:
all generators, seen generators, held-out Qwen, and each individual generator.
Threshold is fixed at 0.5. The same real images are reused in each generator
comparison; these comparisons are not independent. Use content-family grouped
resampling for uncertainty. Confidence intervals are left to the evaluation owner.

Send back the output directory containing `COMPLETE.json`, the evaluation plan,
predictions and metrics. A partial directory without `COMPLETE.json` is incomplete.
Do not select checkpoints, thresholds, hyperparameters or augmentations from test
performance. Record unexpected failures and any rerun reason.

## Interpretation and limitations

Validation uses the four training generators; Qwen is absent from training and
validation. Nearly perfect seen-validation AUROC is not proof of transfer to Qwen.
Real training images use JPEG-quality-75-equivalent tables; generated images use
quality-92-equivalent tables. The fixed JPEG-75 frozen-model validation diagnostic
showed measurable sensitivity. Compression robustness of the fine-tuned models
has not yet been evaluated. Original test images remain the primary test condition;
do not silently recompress them or substitute a robustness condition.

Split IDs, byte hashes and reference/content groups were checked, but this is not
an exhaustive perceptual duplicate audit. Earlier tiny pipeline smoke tests used
some test examples; do not describe the project as having zero historical test
exposure. No full trained-model test evaluation was performed on Cameron's machine
while preparing this package.

## Rebuilding the archives on the original training machine

```powershell
.\.venv\Scripts\python.exe scripts/package_evaluation_handoff.py --output ..\ECS271_evaluation_handoff_2026-11-27
```

This one-time builder expects the original local run/data layout and fresh output,
`data/evaluation-handoff`, and `data/models/dinov3` directories. The evaluator itself
uses portable paths. Selected inference files omit optimizer/scheduler/scaler state;
full resumable training artifacts remain on the original machine.

## Package verification completed

All 45,000 archived JPEG byte hashes and all model/metadata hashes passed.
The three selected models loaded from an isolated extraction of the delivered
base weights. Five validation predictions per encoder matched saved predictions
exactly on GPU; no test inference was performed. The full unit suite passed
45 tests. Evidence: results/finetuning-grid-2026-11-27/package_verification.json.
