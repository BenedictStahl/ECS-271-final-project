# JPEG-75 validation diagnostic - 27 November 2026

All 7,500 primary seen-generator validation images (1,500 real + 1,500 per
FLUX/Z-Image/SDXL/mage_flow) were evaluated in two paired conditions using each
original frozen encoder and its saved linear head. No Qwen or test images were
opened, no weights were fitted, no threshold changed, and no source image was
replaced. The exact approved validation ID list was enforced by the dataset.

The single predefined control was: EXIF-orient and RGB-decode -> in-memory JPEG
quality 75, chroma subsampling 4:2:0, no optimize/progressive -> the existing
224 resize/crop and native normalization. JPEG processing occurred at the native
oriented image dimensions. Current quantization tables became identical for all
control images; past compression histories were not erased. There was no quality
grid/search. Per-model plan.json records settings and bindings before inference.

| Encoder | Original BA | JPEG75 BA | Change (percentage points) | Original AUROC | JPEG75 AUROC | Real FPR original -> JPEG75 |
|---|---:|---:|---:|---:|---:|---:|
| deit3 | 90.75% | 86.87% | -3.88 | 0.9640 | 0.9399 | 10.80% -> 10.53% |
| siglip2 | 96.98% | 96.29% | -0.69 | 0.9953 | 0.9934 | 3.60% -> 3.60% |
| dinov3 | 97.05% | 94.79% | -2.26 | 0.9959 | 0.9890 | 4.13% -> 4.07% |

BA is balanced accuracy; FPR is the fraction of real photos called fake. Threshold
stayed at 0.5. The fresh original inference reproduced saved BA/FPR exactly and
AUROC within 0.000001; maximum individual probability differences were below
0.00085 (FP16 batch-shape arithmetic). All findings compare fresh paired inference.

## Paired changes and uncertainty

500 seeded percentile bootstrap resamples over 1,499 indexed content groups were
used with identical resampling in both conditions. Groups keep related source
images together according to the existing review/index; they do not constitute
an exhaustive perceptual-family audit. These are exploratory diagnostic intervals,
not multiple-comparison-adjusted tests or new-generator performance estimates.

- deit3: BA change 95% interval [-4.28, -3.48] percentage points.
- siglip2: BA change 95% interval [-0.84, -0.54] percentage points.
- dinov3: BA change 95% interval [-2.56, -1.95] percentage points.

Most performance loss came from fake images being called real, not additional
false alarms on real photos:

- deit3: fake false negatives 462 -> 944 out of 6,000; real false positives 162 -> 158 out of 1,500.
- siglip2: fake false negatives 146 -> 229 out of 6,000; real false positives 54 -> 54 out of 1,500.
- dinov3: fake false negatives 106 -> 381 out of 6,000; real false positives 62 -> 61 out of 1,500.

By-source results are in each results.json. SDXL-Turbo had the largest fake-vs-real
BA decrease in all three models. SigLIP 2 was least sensitive in this one condition;
this is not a universal robustness or held-out-generator ranking.

## Implications

1. The training-header confound is real; all real training quantization tables
   matched Pillow quality 75 and all seen fake training tables matched quality 92.
   This diagnostic demonstrates measurable processing sensitivity, especially
   for DeiT III. Its BA drop is around 3.9 points; DINO drops 2.3; SigLIP drops 0.7.
2. No model collapsed under this particular recompression. Substantial discrimination
   remains, but that does not establish compression independence: prior lossy
   histories remain and JPEG can destroy genuine synthesis traces too. This is
   not a causal estimate of how much original accuracy came from a shortcut.
3. These findings do not justify discarding the dataset, rebuilding existing
   caches, or changing learning rate/unfreezing depth by themselves. Keep original
   baselines and the approved initial recipe as the controlled main comparison.
4. Preserve this same fixed JPEG75 condition as a secondary robustness comparison
   for the resulting fine-tuned checkpoints. Keep original seen-validation AUROC
   as the already agreed selection rule for the initial runs; do not retrospectively
   choose a condition/threshold/checkpoint based on Qwen or primary test scores.
5. Disclose compression history and unequal source resolutions as limitations.
   If a stronger debiasing experiment is later adopted, apply the protocol to
   both frozen and fine-tuned comparisons; uniform re-encoding alone is not a
   guarantee that all historical processing cues are removed.

Full fine-tuning has not started. This check did not alter the approved training
configuration, primary manifests, cached features, heads, or original images.

## Reproduction and artifacts

scripts/check_jpeg_control.py is the diagnostic entry point. It requires a NEW
--output directory and --model deit3/siglip2/dinov3; use --local-dino for DINO.
Each model output has plan.json, paired_predictions.jsonl, results.json. All 41
unit tests pass, including strict split/ID guards and in-memory re-encoding checks.
The local aggregate report is data/runs/jpeg75-validation-control-2026-11-27/summary.json.
No artifacts from this diagnostic have been committed or pushed.
