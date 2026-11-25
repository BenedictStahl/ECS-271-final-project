# Project status — 25 November 2026

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

All 33 unit tests pass. Saved-head reload, predictions, train-only weighted scaler
statistics, folded head equivalence, and generator/view exclusions were verified.

## Next work

1. Josh supplies the fine-tuning configuration; agree on layers, learning rates,
   optimizer, epochs, early stopping and checkpoint selection using seen validation.
2. Implement a simple image-based trainer using the same Qwen holdout, pooling,
   linear head, and baseline augmentation first. Initialize from the exact original
   encoder and the compatible trained head. Changing encoder weights invalidates
   frozen feature caches for that fine-tuned model.
3. Run a short memory/speed/checkpoint check, then run two independent models on
   the two RTX 3090s. Keep the frozen and fine-tuned checkpoints for Ben.
4. Freeze the training/evaluation configuration before primary test evaluation.
   Report held-out Qwen versus real separately from seen-generator results;
   prefer family-aware uncertainty estimates because reviewed test variants correlate.

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
