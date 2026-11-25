# ECS 271: pretrained encoders for AI-image detection

Compare DeiT III Base/16, DINOv3 Base/16, and SigLIP 2 Base/16 using the group's
96,000-image replacement dataset. Each encoder uses mean patch pooling and its
own 768-to-1 linear head. **Qwen-Image-2.1-Turbo is the primary held-out generator.**

## Current state â€” 25 November 2026

- Complete replacement dataset inspected and indexed; splits and visual review applied.
- All three encoders have 96,000 original feature vectors and 42,000 augmented training-view vectors.
- Three frozen heads trained and verified. Encoder weights remain unchanged.
- **Primary full-dataset test evaluation is deferred.** Earlier pilots/debug checks evaluated their own test subsets. Fine-tuning configuration is pending Josh's proposal; no fine-tuning trainer is implemented.
- Local DINO weights are verified and usable without a Hub login.

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
git clone --branch cam-training-pipeline https://github.com/BenedictStahl/ECS-271-final-project.git
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
not every transitive package. `requirements.txt` is the older notebook environment.

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
