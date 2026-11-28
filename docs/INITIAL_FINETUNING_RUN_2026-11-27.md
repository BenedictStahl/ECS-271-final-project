# Completion update — 27 November 2026

All three initial runs and all twelve grid runs completed successfully. No jobs
remain active. The launch/PID statements below are historical. See
FINETUNING_GRID_2026-11-27.md and EVALUATION_HANDOFF.md for current results and handoff.

---

# Active initial fine-tuning runs - 27 November 2026

**Active run root: data/runs/finetune-initial-2026-11-27-r2.**
DeiT III uses GPU 0 (PID 6904), SigLIP 2 uses GPU 1 (PID 24312); DINO is queued
on the first free GPU. See that folder's run_status.json for authoritative live
PIDs/states and each model's progress.json/history.json for progress.

The first attempt below was interrupted after SigLIP encountered recoverable FP16
backward overflow that the strict guard treated as fatal. Both first jobs were
stopped for a clean restart from the original encoders and frozen heads. No first-
attempt partial state is used by r2. All first-attempt logs/status are preserved.

The corrected implementation lets GradScaler skip the overflowed update, reduce
its loss scale and retry the exact same batch. The scheduler advances only on a
successful update. Retry events are recorded in amp_overflow_retries.json; after
eight consecutive failed attempts on a batch, training still fails explicitly.
This is a numerical recovery correction, not a new hyperparameter configuration.
A forced-overflow CUDA regression test verified unchanged weights on overflow,
scale reduction, and successful retry. Main recipe/data/selection are unchanged.

The active folder has immutable recipe.json/protocol.json input copies,
environment_and_sources.json and source_snapshot/ for the corrected launch.
The original 4:08 PM launch details and full recipe are retained below as history;
its old live-PID statements are superseded by this update.

---

# Initial fine-tuning runs - 27 November 2026

Launched at **4:08 PM PST** (2026-11-28 00:08 UTC) from the local working tree on
cam-training-pipeline. Run root: data/runs/finetune-initial-2026-11-27.

## Execution

- DeiT III: GPU 0, PID 51076.
- SigLIP 2: GPU 1, PID 47468.
- DINOv3: queued automatically on whichever GPU finishes its first run successfully.
- Launcher PID 25772. One training process per GPU. A failure stops further queued
  launches; the other already-running process is allowed to finish. See each log.

run_status.json records actual PIDs, commands, times and completion/failure states.
Each model writes progress.json every 100 updates, history.json per epoch, and
best.pt/last.pt. Do not infer completion solely from a checkpoint's existence.
The launcher can continue independently of the chat; these are local processes,
so the computer must remain running. No new reminder/automation was created.

## Fixed recipe

- Original pretrained Base/16 encoder plus its own trained frozen linear head.
- Last 2 transformer blocks, final normalization and head trainable; frozen prefix
  stays in eval mode. Approximately 14.2 million trainable parameters per model.
- Common mean-patch pooling and existing single 768-to-1 head.
- AdamW: encoder LR 1e-5, head LR 1e-4; decay 0.01 on multidimensional weights,
  zero on biases/one-dimensional parameters.
- Batch 32, seed 42, 5% warmup then cosine decay, FP16 encoder + scaled gradients,
  FP32 head/loss, clip global norm at 1.0.
- Maximum 5 epochs; stop after 2 epochs without improved original seen-validation
  AUROC. Best checkpoint selected by that metric, not JPEG control or test scores.
- 35,000 training parents: 7,000 real + 7,000 each FLUX, Z-Image, SDXL and mage_flow.
  Original plus one fixed augmented view per parent: 70,000 views/epoch, 2,188
  optimizer updates/epoch; up to 10,940 updates/model. Parent weights total one;
  class weights real=2.5 and fake=0.625. No class-balanced sampler is added.
- 7,500 unaugmented seen-generator validation images. Qwen is absent from training
  and validation. No primary test evaluation is performed by this trainer.
- JPEG75 remains a separately documented diagnostic; it does not alter this recipe.
- Two data workers per model; four main CPU threads. Windows spawn was smoke-tested.

## Inputs, provenance and outputs

The launcher copied finetuning_initial.json into recipe.json and the approved
training_protocol.json into protocol.json before launch. It passes these snapshots
to all three processes. The Qwen manifest/index checksums and reviewed group
bindings are validated; baseline head identities and augmented pixel hashes are
checked. The image reader checks source SHA256 on every read. All three complete
an initial full validation pass to verify frozen-head initialization before updates.

source_snapshot/ and environment_and_sources.json record the exact launch-time
sources and library versions. Each model configuration.json records trainable
names/counts, recipe and input hashes. Checkpoints store trainable deltas and
optimizer/scheduler/scaler state; reconstruct using the exact original base.
No CLI resume is implemented. An interrupted run retains any already-written
best/last checkpoints, but rerunning into that folder is refused.

## Runtime estimates

An informed estimate will be appended after observing sustained full-data updates.
progress.json estimates remaining runtime assuming all five epochs, using a rolling
window of up to 200 updates plus measured full-validation time. It excludes future
checkpoint-write overhead; early stopping can shorten the run. Startup includes
model/data loading, worker spawning and initial validation, which are not typical
per-epoch training costs. Do not use the six-step smoke extrapolation as a deadline.

## Measured estimate for active r2, 04:15 PM PST

Both first jobs have passed the earlier failure point. SigLIP's first recovered
overflow reduced the scale from 1024 to 512 and the same batch then completed;
no learning rate, data, batch size or optimizer hyperparameter changed.
42 tests pass, including a real CUDA forced-overflow regression test.

Current progress snapshots: 1100 DeiT updates and
1100 SigLIP updates; recent throughput
252 and
248 training views/second.
The first pair has roughly 24-24 minutes
remaining at the five-epoch ceiling. Allow **50-65 minutes from this estimate**
for all three, including queued DINO. That corresponds approximately to
05:05 PM-05:20 PM PST.
Early stopping can finish sooner. DINO's duration remains an inference until
it starts. The first complete epoch will provide a stronger estimate. See the
active run root's runtime_estimate.json and live progress.json files.
