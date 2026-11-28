# Fine-tuning grid — 27 November 2026

Approved: two encoder learning rates × two unfreezing depths × three encoders = 12 runs.
All other settings match the initial recipe. Same splits and seed; select using seen-validation AUROC only.
Checkpoints and predictions stay under data/runs. No test or Qwen evaluation is performed.
The scheduler checks progress and process exit codes every 180 seconds, with at most one training job per GPU.
Any failed job stops new launches; an already-running job is allowed to finish.

Last check (UTC): 2026-11-28T03:02:07.787467+00:00

| Config | Model | Blocks | Encoder LR | Status | Best val AUROC |
|---|---|---:|---:|---|---:|
| A | deit3 | 2 | 1e-05 | complete | 0.9942430000000001 |
| A | siglip2 | 2 | 1e-05 | complete | 0.9999912222222223 |
| A | dinov3 | 2 | 1e-05 | complete | 0.9998238888888888 |
| B | deit3 | 2 | 3e-06 | complete | 0.9869202222222223 |
| B | siglip2 | 2 | 3e-06 | complete | 0.9999766666666666 |
| B | dinov3 | 2 | 3e-06 | complete | 0.9994252222222222 |
| C | deit3 | 4 | 1e-05 | complete | 0.9984782222222223 |
| C | siglip2 | 4 | 1e-05 | complete | 0.9999941111111111 |
| C | dinov3 | 4 | 1e-05 | complete | 0.999985 |
| D | deit3 | 4 | 3e-06 | complete | 0.9966243888888888 |
| D | siglip2 | 4 | 3e-06 | complete | 0.9999925555555556 |
| D | dinov3 | 4 | 3e-06 | complete | 0.9998469999999999 |

Live machine-readable record: data/runs/finetuning-grid-2026-11-27/grid_status.json.
Recipes, protocol, source snapshot, commands, timestamps, logs, and per-run histories are retained with the runs.

## Locked selections and interpretation

All twelve runs completed with exit code 0, five epochs each. For every encoder,
configuration C (last four blocks, encoder LR 1e-5) had the highest seen-validation
AUROC. Selected checkpoint epochs: DeiT 5, SigLIP 3, DINO 5. All other settings
were fixed, including seed 42; this is a single-seed search, not a variance study.

| Encoder | Selected validation AUROC | Balanced accuracy | Real FPR |
|---|---:|---:|---:|
| DeiT III | 0.998478 | 97.30% | 4.80% |
| SigLIP 2 | 0.999994 | 99.86% | 0.20% |
| DINOv3 | 0.999985 | 99.68% | 0.53% |

Independent pair-count AUROC and confusion-matrix checks reproduced saved scores;
validation IDs and labels match the approved manifest. Train/validation byte
hashes, reference IDs and content-group IDs do not overlap. This is not an
exhaustive perceptual duplicate check or proof that models ignore compression.
Same-generator validation is nearly saturated; tiny AUROC differences should not
be described as statistically established superiority or unseen-generator transfer.
The known real/fake JPEG history mismatch remains a limitation.

All epoch histories are tracked in results/finetuning-grid-2026-11-27/. The test
handoff is described in EVALUATION_HANDOFF.md. No full trained-model test scoring
was performed while packaging. Full checkpoints/optimizer states remain local;
the teammate package contains inference-only selected deltas and frozen baselines.
