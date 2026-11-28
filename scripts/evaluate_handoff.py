"""Verify a portable handoff or explicitly evaluate locked models on its test set."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from dataset_io import REPO_ROOT
from finetune import load_finetuned, worker_init
from fit_probe import metrics
from train_probe import load_encoder, image_transform, encode


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes handoff directory")
    return path


def validate_rows(rows):
    if len(rows) != 45000 or len({r["id"] for r in rows}) != 45000:
        raise ValueError("Expected exactly 45,000 unique test images")
    expected = {s: 7500 for s in ["real", "flux3", "sdxl_turbo", "z_image_turbo", "mage_flow", "qwen_image_2_1"]}
    if Counter(r["source"] for r in rows) != expected:
        raise ValueError("Test source counts differ")
    if any(r["split"] != "test" or r["target"] != int(r["source"] != "real") for r in rows):
        raise ValueError("Invalid test split or binary label")


def cohort_metrics(records):
    result = {}
    sources = sorted({r["source"] for r in records} - {"real"})
    cohorts = {"all_generators": sources,
               "seen_generators": [s for s in sources if s != "qwen_image_2_1"],
               "held_out_qwen": ["qwen_image_2_1"]}
    cohorts.update({s: [s] for s in sources})
    for name, fake_sources in cohorts.items():
        selected = [r for r in records if r["source"] == "real" or r["source"] in fake_sources]
        result[name] = metrics(np.array([r["target"] for r in selected]),
                               np.array([r["probability"] for r in selected]))
    return result


class TestImages(Dataset):
    def __init__(self, root, rows, model):
        self.root, self.rows, self.transform = root, rows, image_transform(model)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        path = checked_path(self.root, row["image_file"])
        if checksum(path) != row["sha256"]:
            raise ValueError(f"Image hash changed: {row['id']}")
        with Image.open(path) as image:
            return self.transform(image), i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify", "run"])
    parser.add_argument("--handoff", type=Path, default=REPO_ROOT / "data/evaluation-handoff")
    parser.add_argument("--model", choices=["all", "deit3", "siglip2", "dinov3"], default="all")
    parser.add_argument("--kind", choices=["both", "frozen", "finetuned"], default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive; workers nonnegative")
    root = args.handoff.resolve()
    manifest_path = root / "handoff.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["selection_metric"] != "seen_validation_auroc" or manifest["threshold"] != .5:
        raise ValueError("Unexpected selection/threshold protocol")
    for entry in manifest["files"]:
        if checksum(checked_path(root, entry["path"])) != entry["sha256"]:
            raise ValueError(f"Handoff checksum mismatch: {entry['path']}")
    for entry in manifest["base_files"]:
        if checksum(checked_path(REPO_ROOT, entry["path"])) != entry["sha256"]:
            raise ValueError(f"Base checkpoint checksum mismatch: {entry['path']}")
    rows = [json.loads(line) for line in (root / "test_rows.jsonl").read_text().splitlines()]
    validate_rows(rows)
    if args.action == "verify":
        for i, row in enumerate(rows):
            if checksum(checked_path(root, row["image_file"])) != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {row['id']}")
            if (i + 1) % 5000 == 0:
                print(f"Verified {i+1}/{len(rows)} images", flush=True)
        print("Handoff verified; no model inference or test metrics computed.")
        return
    if args.output is None or args.output.exists():
        parser.error("run requires --output pointing to a new directory")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    models = list(manifest["models"]) if args.model == "all" else [args.model]
    kinds = ["frozen", "finetuned"] if args.kind == "both" else [args.kind]
    plan = dict(started_utc=datetime.now(timezone.utc).isoformat(), handoff_sha256=checksum(manifest_path),
                device=str(device), batch_size=args.batch_size, threshold=.5, models=models, kinds=kinds,
                selection="Locked before test; no training, checkpoint selection, or threshold tuning here")
    (args.output / "evaluation_plan.json").write_text(json.dumps(plan, indent=2))
    for name in models:
        model_entry = manifest["models"][name]
        local_dino = REPO_ROOT / "data/models/dinov3" if name == "dinov3" else None
        loader = DataLoader(TestImages(root, rows, name), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, worker_init_fn=worker_init,
                            pin_memory=device.type == "cuda")
        for kind in kinds:
            out = args.output / f"{name}-{kind}"
            out.mkdir()
            if kind == "finetuned":
                model = load_finetuned(checked_path(root, model_entry["checkpoint"]), device, local_dino)
            else:
                model = load_encoder(name, device, offline=True, local_checkpoint=local_dino)
                with np.load(checked_path(root, model_entry["frozen_head"]), allow_pickle=False) as head:
                    weight, bias = head["weight"].copy(), float(head["bias"])
            started = time.perf_counter()
            records = []
            with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream, torch.inference_mode():
                for pixels, indices in loader:
                    if kind == "finetuned":
                        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                            probabilities = model(pixels.to(device)).sigmoid().cpu().numpy()
                    else:
                        logits = encode(model, name, pixels, device) @ weight + bias
                        probabilities = torch.from_numpy(logits.astype(np.float64)).sigmoid().numpy()
                    if not np.isfinite(probabilities).all():
                        raise ValueError("Nonfinite model probabilities")
                    for i, probability in zip(indices.tolist(), probabilities.tolist()):
                        row = rows[i]
                        record = {k: row[k] for k in ["id", "source", "target", "reference_id", "content_group_id"]}
                        record["probability"] = probability
                        records.append(record)
                        stream.write(json.dumps(record) + "\n")
                    if len(records) % 1000 < args.batch_size:
                        print(f"{name}-{kind}: {len(records)}/{len(rows)}", flush=True)
                stream.flush()
            report = dict(model=name, kind=kind, n=len(records), seconds=time.perf_counter()-started,
                          metrics=cohort_metrics(records), checkpoint_selection=model_entry["selection"],
                          uncertainty="Rows share reference/content families. Use grouped resampling for confidence intervals; do not treat all rows as independent.")
            (out / "metrics.json").write_text(json.dumps(report, indent=2))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Completed {name}-{kind}", flush=True)
    (args.output / "COMPLETE.json").write_text(json.dumps(dict(finished_utc=datetime.now(timezone.utc).isoformat()), indent=2))


if __name__ == "__main__":
    main()
