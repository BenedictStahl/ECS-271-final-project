"""Fine-tune selected final blocks using reviewed image splits; never score test.

Actions are explicit: smoke runs a tiny diagnostic; train runs the full recipe.
Checkpoints contain trainable deltas, so loading requires the exact pretrained base.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from dataset_io import REPO_ROOT
from fit_probe import validate_splits, metrics
from train_probe import CHECKPOINTS, load_encoder, image_transform, training_view, patch_tokens, patch_scores
from training_augmentation import augmentation_config


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_progress(folder, values):
    temporary = folder / "progress.tmp"
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(folder / "progress.json")


def encoder_parts(encoder, name):
    if name == "deit3":
        return encoder.blocks, encoder.norm
    if name == "dinov3":
        return encoder.model.layer, encoder.norm
    if name == "siglip2":
        return encoder.encoder.layers, encoder.post_layernorm
    raise ValueError(name)


class Detector(nn.Module):
    def __init__(self, encoder, name, blocks, head_path):
        super().__init__()
        self.encoder, self.name = encoder, name
        layers, norm = encoder_parts(encoder, name)
        if not 1 <= blocks <= len(layers):
            raise ValueError("Unfreezing depth outside encoder block count")
        encoder.requires_grad_(False)
        self.selected_blocks = list(layers[-blocks:])  # already registered through encoder
        for layer in self.selected_blocks:
            layer.requires_grad_(True)
        norm.requires_grad_(True)
        self.head = nn.Linear(768, 1)
        if head_path is not None:
            with np.load(head_path, allow_pickle=False) as head, torch.no_grad():
                if str(head["pooling"]) != "mean_patch_v1":
                    raise ValueError("Head requires mean-patch pooling")
                self.head.weight.copy_(torch.as_tensor(head["weight"]).reshape(1, 768))
                self.head.bias.copy_(torch.as_tensor(head["bias"]).reshape(1))

    def training_mode(self):
        # Keep prefix/embeddings and global encoder stochastic operations in eval.
        self.eval()
        for block in self.selected_blocks:
            block.train()
        encoder_parts(self.encoder, self.name)[1].train()
        self.head.train()

    def forward(self, pixels):
        tokens = patch_tokens(self.encoder, self.name, pixels).float()
        # Head and loss remain float32 even when encoder inference uses autocast.
        with torch.autocast(pixels.device.type, enabled=False):
            return self.head(tokens.mean(1)).squeeze(1)


def optimizer_groups(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = config["head_lr"] if name.startswith("head.") else config["encoder_lr"]
        decay = config["weight_decay"] if parameter.ndim >= 2 and not name.endswith(".bias") else 0.
        groups.setdefault((lr, decay), []).append(parameter)
    return [dict(params=params, lr=lr, weight_decay=decay) for (lr, decay), params in groups.items()]


def balanced_loss(logits, targets, parent_weights, class_weights):
    losses = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    return (losses * class_weights[targets.long()] * parent_weights).sum() / parent_weights.sum()


def backward_step(model, optimizer, scaler, loss, clip):
    """Return (norm, None) on update, or (None, old_scale) on recoverable AMP overflow.

    Caller retries the same batch after overflow; optimizer/scheduler must not advance.
    """
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError(f"Missing trainable gradient: {name}")
            parameters.append(parameter)
        elif parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter has gradient: {name}")
    finite = torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all()
    if not finite:
        if not scaler.is_enabled():
            raise RuntimeError("Nonfinite gradient without AMP recovery")
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)  # unscale_ recorded overflow: this skips the update
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            raise RuntimeError("AMP overflow did not reduce the loss scale")
        return None, previous_scale
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return norm, None


def schedule_factor(step, total, fraction):
    warmup = max(1, math.ceil(total * fraction))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return .5 * (1 + math.cos(math.pi * progress))


def load_inputs(protocol_path):
    protocol = read_json(protocol_path)
    path = REPO_ROOT / protocol["manifest"]
    if sha(path) != protocol["manifest_sha256"]:
        raise ValueError("Protocol manifest checksum changed")
    manifest = read_json(path)
    index = path.parent / manifest["index_file"]
    if sha(index) != manifest["index_sha256"] or sha(index) != protocol["index_sha256"]:
        raise ValueError("Canonical index checksum changed")
    review_path = path.parent / manifest["review_file"]
    if sha(review_path) != manifest["review_sha256"]:
        raise ValueError("Review checksum changed")
    if read_json(review_path)["reference_family_overrides"] != manifest["reference_family_overrides"]:
        raise ValueError("Review overrides differ")
    if manifest["status"] != "ready" or manifest["held_out_generator"] != protocol["primary_held_out_generator"]:
        raise ValueError("Manifest is not the approved primary experiment")
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    validate_splits(rows, manifest, {r["id"] for r in rows if r["target"]})
    by_id = {r["id"]: r for r in rows}
    selected = {s: [by_id[i] for i in manifest["splits"][s]] for s in ("train", "validation")}
    return selected, read_json(index.with_name("dataset_summary.json")), manifest, protocol


def small_subset(rows, references, seed):
    real = sorted((r for r in rows if r["source"] == "real"), key=lambda r: r["id"])
    ids = {r["reference_id"] for r in random.Random(seed).sample(real, min(references, len(real)))}
    return [r for r in rows if r["reference_id"] in ids]


class ImageViews(Dataset):
    def __init__(self, rows, summary, name, augment=False):
        if any(r["split"] != ("train" if augment else "validation") for r in rows):
            raise ValueError("Image dataset only accepts selected train or validation rows")
        self.rows, self.summary, self.name, self.augment = rows, summary, name, augment
        self.transform = image_transform(name)

    def __len__(self):
        return len(self.rows) * (2 if self.augment else 1)

    def __getitem__(self, index):
        row = self.rows[index // 2 if self.augment else index]
        view = index % 2 if self.augment else 0
        root = Path(self.summary["image_roots"][row["image_root"]] if "image_roots" in self.summary else self.summary["dataset_root"]).resolve()
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or sha(path) != row["sha256"]:
            raise ValueError(f"Image path/hash changed: {row['id']}")
        with Image.open(path) as image:
            if view:
                pixels, record = training_view(image, self.name, row, 1)
            else:
                pixels, record = self.transform(image), row
        return dict(pixels=pixels, target=float(row["target"]), weight=.5 if self.augment else 1.,
                    id=record["id"], source=row["source"], view=view,
                    pixel_sha256=record.get("view_pixel_sha256", ""))


def worker_init(_):
    torch.set_num_threads(1)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    records = []
    started = time.perf_counter()
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(batch["pixels"].to(device))
        probabilities = logits.sigmoid().cpu().tolist()
        records.extend(dict(id=i, source=s, target=int(y), probability=p) for i, s, y, p in
                       zip(batch["id"], batch["source"], batch["target"], probabilities))
    result = metrics(np.array([r["target"] for r in records]), np.array([r["probability"] for r in records]))
    return result, records, time.perf_counter() - started


def parameter_hashes(model):
    return {n: hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, p in model.named_parameters()}


def save_checkpoint(path, model, optimizer, scheduler, scaler, metadata):
    delta = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    payload = dict(format_version=1, trainable_state=delta, optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), metadata=metadata)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_delta(model, checkpoint):
    metadata = checkpoint.get("metadata")
    if metadata and (metadata["model"] != model.name
                     or tuple(metadata["checkpoint"]) != CHECKPOINTS[model.name]
                     or metadata["config"]["unfrozen_blocks"] != len(model.selected_blocks)):
        raise ValueError("Checkpoint model/revision/unfreezing configuration differs")
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    state = checkpoint["trainable_state"]
    if set(state) != expected:
        raise ValueError("Checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                if parameter.shape != state[name].shape:
                    raise ValueError("Checkpoint parameter shape differs")
                parameter.copy_(state[name])


def load_finetuned(path, device, local_dino=None):
    """Reconstruct exact base + trained delta for inference/XAI; no baseline head needed."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    name = metadata["model"]
    if checkpoint["format_version"] != 1 or tuple(metadata["checkpoint"]) != CHECKPOINTS[name]:
        raise ValueError("Unsupported checkpoint format or base revision")
    encoder = load_encoder(name, device, offline=True, local_checkpoint=local_dino)
    model = Detector(encoder, name, metadata["config"]["unfrozen_blocks"], None).to(device)
    restore_delta(model, checkpoint)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "train"])
    parser.add_argument("--model", choices=CHECKPOINTS, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "finetuning_initial.json")
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "training_protocol.json")
    parser.add_argument("--baseline-root", type=Path, default=REPO_ROOT / "data/runs/frozen-qwen-holdout-2026-11-25")
    parser.add_argument("--local-dino", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    config = read_json(args.config)
    if args.output.exists() or args.workers < 0:
        parser.error("Choose a new output directory and a nonnegative worker count")
    if config["augmentation_views"] != 1 or config["selection_metric"] != "seen_validation_auroc":
        parser.error("This trainer implements one fixed extra view and validation AUROC selection")
    if (min(config[k] for k in ("batch_size", "epochs", "patience", "encoder_lr", "head_lr", "gradient_clip")) <= 0
            or not 0 < config["warmup_fraction"] < 1 or config["weight_decay"] < 0):
        parser.error("Invalid training configuration")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    selected, summary, manifest, protocol = load_inputs(args.protocol)
    full_counts = Counter(r["target"] for r in selected["train"])
    class_weights = torch.tensor([len(selected["train"]) / (2 * full_counts[i]) for i in (0, 1)], device=device)
    if args.action == "smoke":
        selected = {s: small_subset(v, 32 if s == "train" else 8, config["seed"])
                    for s, v in selected.items()}
    head_path = args.baseline_root / args.model / "linear_head.npz"
    baseline_metrics = read_json(head_path.with_name("metrics.json"))
    baseline_verification = read_json(head_path.with_name("verification.json"))
    if (baseline_metrics["checkpoint_and_transform"][0] != list(CHECKPOINTS[args.model])
            or baseline_metrics["manifest"] != manifest
            or sha(head_path) != baseline_verification["artifact_hashes"]["linear_head.npz"]):
        raise ValueError("Baseline head/model/manifest binding changed")
    encoder = load_encoder(args.model, device, offline=True, local_checkpoint=args.local_dino)
    model = Detector(encoder, args.model, config["unfrozen_blocks"], head_path).to(device)
    train_data = ImageViews(selected["train"], summary, args.model, augment=True)
    val_data = ImageViews(selected["validation"], summary, args.model)
    generator = torch.Generator().manual_seed(config["seed"])
    loader_options = dict(batch_size=config["batch_size"], num_workers=args.workers,
                          pin_memory=device.type == "cuda", worker_init_fn=worker_init,
                          persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    # Verify fixed view pixels against the existing frozen-feature cache per source.
    augmentation_file = REPO_ROOT / f"data/features/full-replacement-2026-11-25/{args.model}-aug1"
    cached_recipe = read_json(augmentation_file / "provenance.json")["augmentation"]
    if cached_recipe["recipe_id"] != augmentation_config()["recipe_id"]:
        raise ValueError("Augmentation recipe changed since frozen baseline")
    cached_views = {r["id"]: r["view_pixel_sha256"] for r in
                    map(json.loads, (augmentation_file / "rows.jsonl").read_text().splitlines())}
    seen = set()
    for i, row in enumerate(train_data.rows):
        if row["source"] not in seen:
            item = train_data[2 * i + 1]
            if cached_views[item["id"]] != item["pixel_sha256"]:
                raise ValueError("Augmented pixels differ from frozen baseline")
            seen.add(row["source"])
    print(f"{args.model}: verifying frozen-head initialization on {len(val_data)} validation images", flush=True)
    initial_metrics, initial_records, initial_val_seconds = evaluate(model, val_loader, device)
    baseline_predictions = {r["id"]: r["probability"] for r in
                            map(json.loads, head_path.with_name("predictions.jsonl").read_text().splitlines())}
    initial_error = max(abs(r["probability"] - baseline_predictions[r["id"]]) for r in initial_records)
    if initial_error > (.002 if device.type == "cuda" else .01):
        raise ValueError(f"Initial model differs from frozen baseline: {initial_error}")
    optimizer = torch.optim.AdamW(optimizer_groups(model, config))
    total_steps = 6 if args.action == "smoke" else config["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_factor(step, total_steps, config["warmup_fraction"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    before = parameter_hashes(model) if args.action == "smoke" else None
    metadata = dict(model=args.model, checkpoint=CHECKPOINTS[args.model], config=config,
                    action=args.action, manifest_sha256=protocol["manifest_sha256"], index_sha256=protocol["index_sha256"],
                    config_sha256=sha(args.config), baseline_head_sha256=sha(head_path),
                    augmentation=augmentation_config(), trainable_names=sorted(trainable),
                    parameter_counts={"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                      "total": sum(p.numel() for p in model.parameters())},
                    training_parents=len(train_data.rows), training_views=len(train_data), validation_images=len(val_data),
                    class_weights=class_weights.tolist(), device=str(device), torch_version=str(torch.__version__),
                    workers=args.workers, started_utc=started_utc, initial_validation_seconds=initial_val_seconds,
                    trainer_sha256=sha(__file__), test_evaluation="never performed by this trainer",
                    checkpoint_format="trainable delta; exact pretrained base required; no CLI resume implemented")
    args.output.mkdir(parents=True)
    (args.output / "configuration.json").write_text(json.dumps(metadata, indent=2))
    best_auc, stale, global_step, history = -math.inf, 0, 0, []
    step_times = []
    overflow_retries = []
    processed_views = 0
    latest_val_seconds = initial_val_seconds
    for epoch in range(1 if args.action == "smoke" else config["epochs"]):
        epoch_started = time.perf_counter()
        model.training_mode()
        losses = []
        tick = time.perf_counter()
        for batch in train_loader:
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(batch["pixels"].to(device))
                loss = balanced_loss(logits, batch["target"].to(device).float(), batch["weight"].to(device).float(), class_weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                norm, previous_scale = backward_step(model, optimizer, scaler, loss, config["gradient_clip"])
                if norm is not None:
                    break
                overflow_retries.append(dict(epoch=epoch + 1, completed_optimizer_steps=global_step,
                    attempt=attempt + 1, old_scale=previous_scale, new_scale=scaler.get_scale()))
                (args.output / "amp_overflow_retries.json").write_text(json.dumps(overflow_retries, indent=2))
                print(f"{args.model}: AMP overflow; retry same batch, scale {previous_scale} -> {scaler.get_scale()}", flush=True)
            else:
                raise RuntimeError("Repeated AMP overflow after eight attempts on one batch")
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            if global_step >= 2:
                step_times.append(elapsed / len(batch["id"]))
            losses.append(float(loss.detach()))
            global_step += 1
            processed_views += len(batch["id"])
            if args.action == "smoke" or global_step % 100 == 0:
                print(f"{args.model} epoch={epoch+1} step={global_step} loss={float(loss):.5f} gradient_norm={float(norm):.4f}", flush=True)
                if step_times:
                    seconds_per_view = float(np.mean(step_times[-200:]))
                    remaining_views = max(0, config["epochs"] * len(train_data) - processed_views)
                    write_progress(args.output, dict(phase="training", model=args.model, device=str(device),
                        epoch=epoch + 1, max_epochs=config["epochs"], optimizer_steps=global_step,
                        steps_per_epoch=len(train_loader), processed_training_views=processed_views,
                        amp_overflow_retries=len(overflow_retries), loss_scale=scaler.get_scale(),
                        recent_training_views_per_second=1 / seconds_per_view,
                        elapsed_seconds=time.perf_counter() - run_started,
                        remaining_seconds_if_all_epochs=remaining_views * seconds_per_view + (config["epochs"] - epoch) * latest_val_seconds,
                        estimate_note="Recent update throughput plus measured validation; excludes future checkpoint writes. Early stopping can shorten run.",
                        updated_utc=datetime.now(timezone.utc).isoformat()))
            tick = time.perf_counter()
            if args.action == "smoke" and global_step == total_steps:
                break
        training_seconds = time.perf_counter() - epoch_started
        write_progress(args.output, dict(phase="validation", model=args.model, epoch=epoch + 1,
                                         optimizer_steps=global_step, elapsed_seconds=time.perf_counter() - run_started))
        val_metrics, predictions, val_seconds = evaluate(model, val_loader, device)
        latest_val_seconds = val_seconds
        history.append(dict(epoch=epoch + 1, mean_training_loss=float(np.mean(losses)), validation=val_metrics,
                            training_seconds=training_seconds, validation_seconds=val_seconds))
        improved = val_metrics["auroc"] > best_auc
        stale = 0 if improved else stale + 1
        checkpoint_path = args.output / ("smoke.pt" if args.action == "smoke" else "last.pt")
        state_metadata = dict(metadata, epoch=epoch + 1, optimizer_steps=global_step, validation=val_metrics)
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, state_metadata)
        if args.action == "smoke":
            (args.output / "validation_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
        if improved:
            best_auc = val_metrics["auroc"]
            if args.action == "train":
                save_checkpoint(args.output / "best.pt", model, optimizer, scheduler, scaler, state_metadata)
                (args.output / "validation_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
                np.savez(args.output / "linear_head.npz", weight=model.head.weight.detach().cpu().numpy().reshape(768),
                         bias=model.head.bias.detach().cpu().numpy()[0], pooling="mean_patch_v1")
        history[-1]["epoch_total_seconds"] = time.perf_counter() - epoch_started
        history[-1]["amp_overflow_retries_total"] = len(overflow_retries)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"{args.model}: validation AUROC={val_metrics['auroc']:.6f}; epoch {epoch+1} complete", flush=True)
        if stale >= config["patience"]:
            break
    if args.action == "smoke":
        training_peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        after = parameter_hashes(model)
        changed = {n for n in before if before[n] != after[n]}
        if not changed or changed - trainable:
            raise RuntimeError("Unexpected/frozen parameter updates")
        for block in model.selected_blocks + [encoder_parts(model.encoder, model.name)[1], model.head]:
            names = {n for n, p in model.named_parameters() if any(p is q for q in block.parameters())}
            if not changed.intersection(names):
                raise RuntimeError("A selected block/norm/head did not update")
        # Reload into a fresh original encoder, not merely back into the same model.
        saved = torch.load(args.output / "smoke.pt", map_location="cpu", weights_only=True)
        reloaded = load_finetuned(args.output / "smoke.pt", device, args.local_dino)
        for name, parameter in reloaded.named_parameters():
            if name in saved["trainable_state"]:
                torch.testing.assert_close(parameter.cpu(), saved["trainable_state"][name], rtol=0, atol=0)
        _, restored_records, _ = evaluate(reloaded, val_loader, device)
        np.testing.assert_allclose([r["probability"] for r in predictions], [r["probability"] for r in restored_records], rtol=0, atol=1e-7)
        batch = next(iter(val_loader))["pixels"].to(device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            tokens = patch_tokens(reloaded.encoder, args.model, batch).float()
        with torch.inference_mode():
            patch_scores(tokens, reloaded.head.weight.reshape(768), reloaded.head.bias[0])
        verification = dict(status="passed", scope="Technical smoke only; not a scientific fine-tuning run",
                            optimizer_steps=global_step, frozen_parameters_unchanged=True,
                            changed_parameter_names=sorted(changed), finite_gradients_and_loss=True,
                            each_selected_block_norm_head_updated=True, fresh_base_checkpoint_reload_predictions_match=True,
                            fixed_augmented_pixels_match_baseline=True, initial_baseline_probability_max_difference=initial_error,
                            patch_score_identity_passed=True, test_images_read=0,
                            peak_allocated_training_gib=training_peak,
                            measured_training_views_per_second=1 / np.mean(step_times),
                            rough_full_epoch_minutes=(70000 * np.mean(step_times) + val_seconds * 7500 / len(val_data)) / 60,
                            timing_note="Four timed updates after two warmup updates; small-run estimate, not a full-run benchmark. Smoke schedule compressed to six steps.")
        (args.output / "verification.json").write_text(json.dumps(verification, indent=2))
        print(json.dumps({k: v for k, v in verification.items() if k != "changed_parameter_names"}, indent=2), flush=True)
    write_progress(args.output, dict(phase="complete", model=args.model, epochs_completed=len(history),
        optimizer_steps=global_step, best_validation_auroc=best_auc,
        early_stopped=stale >= config["patience"], elapsed_seconds=time.perf_counter() - run_started,
        test_evaluation="not performed", finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == "__main__":
    main()
