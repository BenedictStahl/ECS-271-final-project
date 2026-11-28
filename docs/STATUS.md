# Project status — 27 November 2026

## Completed

The replacement delivery has 96,000 RGB JPEGs: train 42,000, validation 9,000,
test 45,000. Each source contributes 7,000/1,500/7,500. Sources: real, flux3,
sdxl_turbo, z_image_turbo, qwen_image_2_1, mage_flow. MiniMax was not delivered.

Metadata matches every image. No missing/extra images, decoding failures, exact
byte duplicates, or exact decoded-pixel duplicates were found. Previous 64,000
image hashes are unchanged. New images were fully decoded; unchanged images
were rehashed against their earlier full-decode ledgers. Real-caption changes
are whitespace trimming. This is not an exhaustive perceptual-duplicate audit.

All 55 original cross-split caption flags were visually cleared as distinct
photographs. Two extra pairs exposed by trimmed captions were also cleared.
Original decisions and 34 reviewed test-only photo-variant families are retained;
269 real references have explicit review decisions. No images were excluded.

All encoders have 96,000 original vectors and 42,000 deterministic training views,
768 dimensions, float32, finite. Existing 48,000-row DeiT/SigLIP caches were reused.
Augmented pixel hashes match across all three encoders. Fresh inference reproduced
18 sampled original vectors exactly; patch-score reconstruction checks passed.

The primary Qwen holdout has 35,000 training originals, 7,500 validation images,
and 45,000 test images. One additional training view produces 70,000 fit rows,
with each original parent's total weight still one. Qwen originals and its
7,000 cached training views are excluded from fitting; Qwen is absent from validation.

All three frozen heads converged with fixed balanced logistic regression C=1 and
threshold 0.5. No hyperparameter or threshold search was performed. Fits took
2.8/3.5/3.0 seconds after feature loading, with 92/132/97 iterations respectively.

| Encoder | Validation balanced accuracy | AUROC | Real false-positive rate |
|---|---:|---:|---:|
| DeiT III | 90.75% | 0.963956 | 10.80% |
| SigLIP 2 | 96.98% | 0.995343 | 3.60% |
| DINOv3 | 97.05% | 0.995945 | 4.13% |

Validation contains only seen generators. Test predictions and metrics for the
primary full-dataset runs have not been computed. Earlier two-generator pilots
and the tiny pipeline smoke tests did evaluate their test subsets; their outputs
are historical/debugging evidence, not the primary full-dataset results.

The frozen-baseline stage passed 33 unit tests. Saved-head reload, predictions, train-only weighted scaler
statistics, folded head equivalence, and generator/view exclusions were verified.

## Fine-tuning readiness and next work

The group approved the initial configuration in finetuning_initial.json. A simple
image-based trainer is implemented in scripts/finetune.py. Real-checkpoint smoke
runs passed on both RTX 3090s for all three encoders: six updates at batch 32,
with 160 selected training parents (320 possible original/augmented views),
192 views actually processed, and 40 unaugmented validation images each.
Frozen weights were unchanged; selected blocks/norm/head updated with finite
gradients. Fresh-base/delta reloads reproduced predictions, and patch-score
math passed. Per-source augmented pixels matched the existing baseline caches.
42 unit tests pass, including the validation-only JPEG diagnostic. No primary test images were read or scored.

1. Start the three full initial-config runs, two concurrently on separate GPUs,
   then the third. Each starts from the original encoder and frozen baseline head,
   not from smoke-test weights. Full initial-config runs are active; see the execution record below.
2. Compare seen-generator validation and available time; optional next experiments
   are encoder LR 3e-6, then four unfrozen blocks. Weight decay stays fixed.
3. Freeze configurations before primary test evaluation and XAI comparison.
   Report held-out Qwen versus real separately from seen-generator results;
   prefer family-aware uncertainty because reviewed test variants correlate.

Smoke checkpoints are trainable deltas plus optimizer/scheduler/scaler state;
they require the exact original pretrained encoder. No CLI resume is implemented.
The saved protocol now points to the approved fine-tuning configuration. Earlier
frozen runs bind the prior protocol file hash as historical provenance; their
manifest/index bindings and trained artifacts are unchanged.

Ben can already use each encoder's linear_head.npz for XAI. Recompute final patch
tokens on selected images; mean(score grid) + bias equals the image logit.
The scores are contextual feature contributions, not causal pixel explanations.
Sudarsan can prepare evaluation/report infrastructure without opening primary test scores.

## Unresolved provenance and scope

- Josh must identify mage_flow's exact checkpoint/variant and settings. Do not
  inherit MiniMax or the other generators' assumptions. This does not block loading.
- FLUX's committed workflow uses four steps; the official documentation inspected
  on 25 November lists sixteen for distilled Klein. Clarify revision/settings
  before making claims about representative FLUX image quality.
- Preserve the distinction between controlled encoder comparisons and causal
  claims about pretraining objectives: datasets and recipes differ too.
- CNN baseline, generator attribution, and Josh's KEV/DPO side task are separate
  scope decisions, not completed components of this binary training pipeline.

The primary holdout was explicitly selected by Cameron after protocol analysis,
not from the full baseline test scores. Other holdout manifests remain available;
no full five-holdout benchmark has been run.

## JPEG validation check completed

One fixed quality-75 re-encoding condition was compared with fresh original
inference on the exact 7,500 seen-generator validation images. No Qwen/test
images were opened, no model was fitted and threshold stayed at 0.5. BA changes
were -3.88/-0.69/-2.26 percentage points for DeiT/SigLIP/DINO respectively;
real FPR stayed essentially unchanged. This supports reporting compression
sensitivity, especially for DeiT. It does not isolate shortcut reliance from
destruction of genuine forensic cues or erase historical compression. The
approved initial recipe and all existing artifacts remain unchanged. Preserve
this condition as a secondary comparison for fine-tuned checkpoints. See
JPEG_VALIDATION_CONTROL_2026-11-27.md for protocol, intervals and implications.

## Full fine-tuning grid completed

All 12 runs completed successfully: two encoder learning rates (1e-5, 3e-6),
two unfreezing depths (2, 4), and three encoders. Each ran five epochs; no run
failed. The initial failed AMP attempt is historical and excluded from the grid.
All three selected four final blocks and encoder LR 1e-5 using seen-validation
AUROC. Selected epochs: DeiT 5, SigLIP 3, DINO 5. Primary full test evaluation
has not been run. See FINETUNING_GRID_2026-11-27.md and EVALUATION_HANDOFF.md.

Git contains code, recipes and compact validation results under results/.
Model weights and 45,000 test images are supplied in separate private archives.
The frozen baseline encoder weights remain unchanged; fine-tuned models require
both the original encoder and the selected trained delta.
