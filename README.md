# ECS 271: pretrained encoders for AI-image detection

Compare DeiT III Base/16, DINOv3 Base/16, and SigLIP 2 Base/16 using the group's
96,000-image replacement dataset. Each encoder uses mean patch pooling and its
own 768-to-1 linear head. **Qwen-Image-2.1-Turbo is the primary held-out generator.**

## Current state â€” 25 November 2026

- Complete replacement dataset inspected and indexed; splits and visual review applied.
- All three encoders have 96,000 original feature vectors and 42,000 augmented training-view vectors.
- Three frozen-head baselines trained and verified; all 12 encoder fine-tuning grid runs are complete.
- **Primary full-dataset test evaluation is deferred.** Earlier pilots/debug checks evaluated their own test subsets.
- Approved fine-tuning trainer/configuration implemented; technical smoke tests passed for all three encoders. The 12-run grid completed successfully; see [grid results](docs/FINETUNING_GRID_2026-11-27.md).
- Local DINO weights are verified and usable without a Hub login.

Frozen-head baselines:

| Encoder | Seen-generator validation balanced accuracy | Validation AUROC |
|---|---:|---:|
| DeiT III | 90.75% | 0.9640 |
| SigLIP 2 | 96.98% | 0.9953 |
| DINOv3 | 97.05% | 0.9959 |

These are validation results on 7,500 images from real + four seen generators.
They do not measure Qwen generalization or establish a statistically significant ranking.

## Windows setup

Use 64-bit Python 3.12 and a reasonably short checkout path.

```powershell
git clone --branch cam-finetuning-evaluation https://github.com/BenedictStahl/ECS-271-final-project.git
cd ECS-271-final-project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\check_training_env.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The pinned Windows wheels include the CUDA runtime; an NVIDIA driver is required,
but a separate CUDA toolkit is not. Frozen-head fitting uses the CPU. Feature
extraction and later fine-tuning use the GPUs. Calling the venv's Python directly
avoids PowerShell activation-policy issues. The requirements pin direct dependencies,
not every transitive package. `requirements.txt` is the upstream notebook/data-generation dependency list;
use requirements-training.txt for this Windows training pipeline. Do not combine
the two pin sets in the same environment.

Cloning this branch supplies code and documentation. Replacement images, review/index inputs,
model weights, features, and trained checkpoints are transferred separately and
remain Git-ignored. Research documents and their handoff ZIP remain outside this repo.

## Documentation

- [Current results, pending work, and team handoff](docs/STATUS.md)
- [Training protocol, exact checkpoints, augmentation, and commands](docs/TRAINING.md)
- [Local artifact locations, transfer inputs, and cleanup record](docs/DATA_LAYOUT.md)
- [Original Defactify download/profile workflow](docs/DEFACTIFY.md)
- [Primary experiment selection](training_protocol.json)
- [Generator provenance and unresolved assumptions](scripts/generation_provenance.json)
- [Historical working notes](docs/history/WORKING_NOTES_THROUGH_2026-11-25.md) â€” superseded snapshots, not current instructions

## Main entry points

| Script | Purpose |
|---|---|
| `replacement_dataset.py` | Index multiple image roots against supplied metadata and the reference map |
| `build_holdout_manifests.py` | Apply hash-bound visual decisions; construct shared holdout manifests |
| `train_probe.py` | Check pretrained encoders; extract original/missing or augmented features |
| `finetune.py` | Image-based last-block fine-tuning; explicit smoke/train actions; no test scoring |
| `fit_probe.py` | Fit a frozen linear head; export its affine coefficients and predictions |
| `training_augmentation.py` | Bind Ben's augmentation to deterministic training-only views |
| `smoke_training.py` | Tiny all-encoder/all-holdout pipeline check, separate from scientific runs |
| `check_training_env.py` | Dependency/GPU checks; optional random-weight architecture checks |
| `download_dataset.py`, `profile_dataset.py` | Original Defactify compatibility audit |

Scripts live in `scripts/`. Inventory, manifest, feature-cache, smoke-run and
head-fit output directories must be new; these entry points refuse existing
destinations. The original downloader supports reruns and the profiler can replace
its report. Extraction is not resumable mid-cache. No new training is
needed to resume Ben's XAI work with the saved frozen heads.

Accuracy audit completed 25 November 2026 against current code, filesystem,
metadata, manifest/review bindings, encoder/cache provenance, and saved validation
outputs. All 96,000 current image hashes match the indexed full-decode evidence.
The local audit report is `data/maintenance/2026-11-25/documentation_layout_audit.json`.
No primary test predictions were computed during this check.

## Team image-generation tools

The upstream `dataset-gen/` ComfyUI generation/description/metadata tools and
`local_img_gen/` caption-based generation workflow are retained. See
[local generation setup](local_img_gen/README.md) for that separate environment.
The committed dataset-gen workflow still lists MiniMax; delivered mage_flow
provenance remains unresolved. These tools are not required to fit the frozen heads.

The downloader conflict with main was resolved in favor of direct, revision-pinned
Parquet download, preserving the portable audit and avoiding full Arrow cache creation.
Python files use LF checkouts via .gitattributes so augmentation source hashes and
recipe identities stay portable across operating systems.

## Fine-tuning readiness — 27 November

See finetuning_initial.json and docs/TRAINING.md. The approved initial recipe
unfreezes the last two blocks, final norm and existing linear head, with encoder
LR 1e-5 and head LR 1e-4. All three real-checkpoint smoke runs passed at batch 32
(six optimizer updates each, 40 validation images). 42 unit tests pass.
Smoke artifacts live in data/runs/finetune-smoke-2026-11-27; they are technical
diagnostics, not primary experiment results. Original frozen baselines are preserved.
The fine-tuning additions are currently local working-tree changes, not yet pushed.

## Validation-only JPEG diagnostic

The fixed JPEG75 control is complete for all three frozen models on the exact
7,500 seen-generator validation images. Balanced accuracy changed from
90.75% to 86.87% (DeiT III), 96.98% to 96.29% (SigLIP 2), and 97.05% to
94.79% (DINOv3). No Qwen/test images, fitting, threshold changes or source-image
rewrites were involved. This shows processing sensitivity, not a causal estimate
of compression-shortcut reliance. The approved initial training recipe is unchanged.
See [paired diagnostic and implications](docs/JPEG_VALIDATION_CONTROL_2026-11-27.md).

## Active initial runs

DeiT III and SigLIP 2 run concurrently, one per RTX 3090. DINO is queued on the
first free GPU. Active artifacts: data/runs/finetune-initial-2026-11-27-r2.
See [recipe, execution and timing record](docs/INITIAL_FINETUNING_RUN_2026-11-27.md).
The first attempt's FP16 recovery issue and restart are documented there.

## Evaluation handoff

See [evaluation instructions](docs/EVALUATION_HANDOFF.md). The new branch contains
training/evaluation code, all 12 validation histories and locked model selections.
Large image/model transfer archives stay outside Git. All three encoders selected
four final blocks at encoder LR 1e-5 by seen-validation AUROC. Primary full test
evaluation remains pending; preparing the handoff does not run test inference.
