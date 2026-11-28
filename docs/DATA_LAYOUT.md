# Local data and artifact map

All paths below are relative to the repo root unless prefixed `../`.
`data/*` is Git-ignored. Dataset and model folder locations are intentionally
unchanged to preserve saved provenance. Root-level reports stay where existing
bindings and earlier handoffs expect them. No private replacement dataset is
fetched by a Git clone.

## Active artifacts

| Path | Contents |
|---|---|
| `../partial_new_image_set/metadata_updated.csv` | Supplied full 96,000-row metadata |
| `../partial_new_image_set/extracted` | Real, FLUX, SDXL-Turbo: 48,000 images |
| `../partial_new_image_set/z_image_turbo/extracted` | Z-Image-Turbo: 16,000 images |
| `../partial_new_image_set/additional_delivery_2026-11-25/extracted` | mage_flow + Qwen: 32,000 images; mage_flow's exact checkpoint remains unconfirmed |
| `../partial_new_image_set/inspection/reference_map.jsonl` | Original real-reference identities/splits/captions |
| `../partial_new_image_set/inspection/group_review` | Original 55-flag visual evidence and decisions |
| `../partial_new_image_set/inspection/full_delivery_2026-11-25` | Full delivery audit, image ledger, sample sheet |
| `data/review-decisions-2026-11-25.json` | Preserved original review + four additional reference decisions |
| `data/additional-caption-review-2026-11-25.jpg` | Visual evidence for two whitespace-induced caption matches |
| `data/full-delivery-split-conflicts-2026-11-25.json` | Records identifying those two subsequently cleared pairs |
| `data/replacement-inventory-2026-11-25` | Full unreviewed inventory + source/root metadata |
| `data/replacement-holdouts-2026-11-25` | Canonical reviewed index + five candidate manifests; Qwen is primary |
| `data/models/deit3`, `data/models/siglip2` | Complete locally cached pretrained checkpoints |
| `../dino_inspection_2026-11-25/dinov3-base` | DINO config, weights, license and manifest |
| `data/runs/frozen-qwen-holdout-2026-11-25` | Three trained frozen heads, validation outputs, checks and summary |

## Feature caches used to train/reproduce current heads

These caches are needed to reproduce fitting and cached-feature evaluation.
For inference/XAI on a new image, use the appropriate pretrained encoder,
preprocessing and trained head; existing dataset-wide feature caches are not needed.

| Encoder | Original feature folders | Extra training views |
|---|---|---|
| DeiT III | `data/features/deit3-replacement-mean` (48K) + `data/features/full-replacement-2026-11-25/deit3-original` (48K) | `data/features/full-replacement-2026-11-25/deit3-aug1` (42K) |
| SigLIP 2 | `data/features/siglip2-replacement-mean` (48K) + `data/features/full-replacement-2026-11-25/siglip2-original` (48K) | `data/features/full-replacement-2026-11-25/siglip2-aug1` (42K) |
| DINOv3 | `data/features/full-replacement-2026-11-25/dinov3-original` (96K) | `data/features/full-replacement-2026-11-25/dinov3-aug1` (42K) |

Keep both old and new DeiT/SigLIP original caches: they are complementary, not
duplicates. Each folder has features.npy, rows.jsonl and provenance.json. Full
verification and raw-image spot checks are in the full-replacement folder.
Cached Qwen training views are not selected when fitting the primary heads.

## Historical/reproducibility inputs retained

- `data/replacement-inventory-2026-11-23`: original hash-bound input required for
  additive review verification. Do not delete simply because it is older.
- `data/replacement-holdouts-2026-11-23`: pilot reviewed index/manifests referenced
  by old caches and pilot runs.
- `data/features/deit3-defactify`, `data/features/siglip2-defactify`: original native-pooling
  caches, incompatible with mean-patch training; retained as historical evidence.
- `data/runs/pilot-two-generators-2026-11-23`: previous two-generator experiments.
- `data/runs/full-delivery-smoke-2026-11-25` and
  `data/features/augmentation-smoke-2026-11-25`: tiny implementation checks.
- `data/augmentation-review-2026-11-25`: augmentation review examples.
- `data/hf/hub/datasets--Rajarshi-Roy-research--Defactify_Image_Dataset/snapshots/1c3b714c9691eb6dd11c2c1e92eab1ac29584fa0/data`:
  complete original 17-shard Parquet source. Pass this explicitly with --data-dir;
  the portable downloader's default is data/raw.
- `data/datasets`: tiny historical download metadata; left untouched.
- Supplied ZIPs and original metadata beside the extracted images remain untouched.
  Research handoff ZIP/documents and the DINO delivery ZIP remain outside this repo.

## Moving to another computer

Transfer the three image roots, updated CSV, reference map, review files/evidence,
and original 23-Nov inventory needed for additive review verification. Transfer
model folders/licenses and feature caches if avoiding downloads/re-extraction.
Transfer trained heads/run metadata if continuing from the frozen baseline.

Saved dataset_summary.json files contain absolute local root paths. Rebuild an
inventory/manifests against the destination roots, reusing the transferred review
and original base inventory. Keep image IDs/hashes/splits intact. Review the new
manifest/index checksums and update training_protocol.json explicitly before
new runs. Old canonical_cache_rows compatibility allows unchanged vectors to
adopt the newly reviewed canonical index. Do not silently edit historic run files.

## Cleanup performed 25 November 2026

- Removed data/a (7,521,905,027 bytes), a regenerable Arrow-format copy of original
  Defactify. All 96,000 image bytes, captions and labels were compared against the
  retained Parquet source as a multiset before deletion; every record matched.
- Removed a stale 737,929,479-byte SigLIP .incomplete download. The complete
  1,500,800,904-byte checkpoint remains; it passed loading/inference and safetensors checks.
- Freed 8,259,834,506 bytes (about 7.69 GiB), excluding filesystem allocation effects.
- Moved six one-off maintenance/verification scripts out of data's top level into
  `data/maintenance/2026-11-25`. Run them from the repo root if needed; the extraction
  and training launchers refuse existing output folders, while verification scripts
  can refresh their reports. The redundant-Arrow verifier
  describes a removed input and is kept as historical evidence only.
- Preserved all supplied archives, image roots, metadata, review evidence, model
  weights, feature arrays, historical pilots, trained heads and reports.

Detailed removal/move ledger, original Arrow metadata and record-equivalence proof:
`data/maintenance/2026-11-25/cleanup.json`,
`data/maintenance/2026-11-25/removed_arrow_dataset_info.json`,
and `data/maintenance/2026-11-25/arrow_cache_equivalence.json`. No active paths changed.

Accuracy audit completed 25 November 2026 against current code, filesystem,
metadata, manifest/review bindings, encoder/cache provenance, and saved validation
outputs. All 96,000 current image hashes match the indexed full-decode evidence.
The local audit report is `data/maintenance/2026-11-25/documentation_layout_audit.json`.
No primary test predictions were computed during this check.

## Fine-tuning smoke artifacts (27 November 2026)

`data/runs/finetune-smoke-2026-11-27/{deit3,siglip2,dinov3}` contains each
configuration, six-step history, smoke.pt trainable-delta checkpoint, validation
predictions, verification.json and a separate checkpoint-loading verification.
`data/runs/finetune-smoke-2026-11-27/summary.json` summarizes readiness. These
are diagnostics, not scientific training results; do not use them to initialize
the full runs. The original frozen-head runs and feature caches are unchanged.
Logs are data/finetune-MODEL-smoke-2026-11-27.log.

## Validation JPEG diagnostic artifacts

`data/runs/jpeg75-validation-control-2026-11-27/{deit3,siglip2,dinov3}` contains
plan.json (fixed settings and hashes written before inference), paired_predictions.jsonl
and results.json. The parent folder has summary.json and REPORT.txt. These are
validation-only diagnostic outputs, not training data or primary test results.
The training-only header audits are in data/maintenance/2026-11-27/; no source
JPEGs or caches were rewritten.

## Initial full fine-tuning runs

Completed: data/runs/finetune-initial-2026-11-27-r2. Top level has immutable
recipe.json/protocol.json, source_snapshot/, environment_and_sources.json,
run_status.json, per-model logs and runtime_estimate.json. Model directories
have configuration.json, progress.json, per-epoch history/checkpoints and any
AMP recovery events. All three model directories are present and complete.
The unsuffixed finetune-initial-2026-11-27 folder is the interrupted first attempt;
its status explains recovery. Its partial weights are not used by r2.

## Completed grid and portable evaluation assets

`data/runs/finetuning-grid-2026-11-27/grid_status.json` records all twelve
successful runs. A runs remain in the initial r2 root; B-DeiT is under
`data/runs/finetune-last2-lr3e-6-2026-11-27`; the other eight model runs are
inside the grid root. Every run retains full best/last training checkpoints.

`results/finetuning-grid-2026-11-27/` contains compact Git-tracked validation
histories and selected-model records. `data/evaluation-handoff/` holds portable
inference deltas, frozen heads and the test index. The test images are delivered
in a separate archive, extracting to its images/test/ directory. No local copy
of those extracted test images is required on the original machine to build it.

Transfer archives live outside the repository in
`../ECS271_evaluation_handoff_2026-11-27/`. See EVALUATION_HANDOFF.md. The evaluator
uses only portable package paths and checks image/model hashes. The original
machine's absolute dataset roots in historical run artifacts are not required.
