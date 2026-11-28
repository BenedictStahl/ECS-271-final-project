"""One predeclared validation-only JPEG-75 diagnostic; frozen models, no fitting.

Decode/EXIF-orient/RGB -> in-memory JPEG75, 4:2:0 -> existing resize/crop/normalization.
Re-run originals in the same inference path and compare with saved baseline outputs.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader

from dataset_io import REPO_ROOT
from finetune import load_inputs, sha, worker_init
from fit_probe import metrics
from train_probe import load_encoder, encode, image_transform, CHECKPOINTS

PLAN = dict(quality=75, subsampling=2, optimize=False, progressive=False,
            transform_order="decode -> EXIF/RGB -> JPEG75 at native oriented dimensions -> existing encoder preprocessing",
            threshold=.5, bootstrap_draws=500, bootstrap_seed=42,
            scope="All primary seen-generator validation images; no Qwen/test images, no fitting or threshold search")


def jpeg_control(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=75, subsampling=2, optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB"), table_hash(decoded)


def table_hash(image):
    return hashlib.sha256(json.dumps(image.quantization, sort_keys=True).encode()).hexdigest()


class ValidationPairs(Dataset):
    def __init__(self, rows, summary, name, approved_ids, held_out):
        if (len(rows) != len(approved_ids) or {r["id"] for r in rows} != set(approved_ids)
                or any(r["split"] != "validation" or r["source"] == held_out for r in rows)):
            raise ValueError("Only the exact approved seen-generator validation selection is allowed")
        self.rows, self.summary, self.transform = rows, summary, image_transform(name)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        root = Path(self.summary["image_roots"][row["image_root"]]).resolve()
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image outside approved root")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Original image hash changed")
        with Image.open(io.BytesIO(content)) as image:
            original_table = table_hash(image)
            original = self.transform(image)
            controlled, control_table = jpeg_control(image)
            controlled = self.transform(controlled)
        return dict(original=original, control=controlled, id=row["id"], original_table=original_table,
                    control_table=control_table)


def deltas(y, original, control, weights=None):
    real, fake = y == 0, y == 1
    w = np.ones(len(y)) if weights is None else weights
    def stats(p):
        fpr = np.average(p[real] >= .5, weights=w[real])
        tpr = np.average(p[fake] >= .5, weights=w[fake])
        return np.array([roc_auc_score(y, p, sample_weight=w), (.5 * (1 - fpr + tpr)), fpr])
    return stats(control) - stats(original)


def paired_intervals(rows, y, original, control):
    groups = sorted({r["content_group_id"] for r in rows})
    group_index = {g: i for i, g in enumerate(groups)}
    assigned = np.array([group_index[r["content_group_id"]] for r in rows])
    rng = np.random.default_rng(PLAN["bootstrap_seed"])
    changes = []
    for _ in range(PLAN["bootstrap_draws"]):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        changes.append(deltas(y, original, control, counts[assigned]))
    intervals = np.quantile(changes, [.025, .975], axis=0).T
    return dict(group_count=len(groups), grouping="indexed content_group_id; same resampling for both conditions",
                draws=PLAN["bootstrap_draws"], delta_95_percentile_intervals={
                    name: interval.tolist() for name, interval in zip(("auroc", "balanced_accuracy", "real_false_positive_rate"), intervals)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-dino", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; select a new folder")
    selected, summary, manifest, protocol = load_inputs(REPO_ROOT / "training_protocol.json")
    rows = selected["validation"]
    assert len(rows) == 7500
    baseline = REPO_ROOT / "data/runs/frozen-qwen-holdout-2026-11-25" / args.model
    verification = json.loads((baseline / "verification.json").read_text())
    for filename, digest in verification["artifact_hashes"].items():
        if sha(baseline / filename) != digest:
            raise ValueError("Baseline artifact hash changed")
    saved = {r["id"]: r["probability"] for r in
             map(json.loads, (baseline / "predictions.jsonl").read_text().splitlines())}
    args.output.mkdir(parents=True)
    plan = dict(PLAN, model=args.model, checkpoint=CHECKPOINTS[args.model],
                manifest_sha256=protocol["manifest_sha256"], index_sha256=protocol["index_sha256"],
                validation_ids_sha256=hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(),
                head_sha256=sha(baseline / "linear_head.npz"), script_sha256=sha(__file__))
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = load_encoder(args.model, device, offline=True, local_checkpoint=args.local_dino)
    assert not any(p.requires_grad for p in model.parameters())
    data = ValidationPairs(rows, summary, args.model, manifest["splits"]["validation"], manifest["held_out_generator"])
    loader = DataLoader(data, batch_size=32, shuffle=False, num_workers=2, worker_init_fn=worker_init,
                        pin_memory=device.type == "cuda")
    head = np.load(baseline / "linear_head.npz", allow_pickle=False)
    original_probabilities, control_probabilities, seen_ids, original_tables, control_tables = [], [], [], [], []
    started = time.perf_counter()
    for batch in loader:
        n = len(batch["id"])
        features = encode(model, args.model, torch.cat([batch["original"], batch["control"]]), device)
        probabilities = expit(features @ head["weight"] + head["bias"])
        original_probabilities.extend(probabilities[:n].tolist())
        control_probabilities.extend(probabilities[n:].tolist())
        seen_ids.extend(batch["id"])
        original_tables.extend(batch["original_table"])
        control_tables.extend(batch["control_table"])
        if len(seen_ids) % 1024 == 0:
            print(f"{args.model}: {len(seen_ids)} paired validation images", flush=True)
    assert seen_ids == [r["id"] for r in rows]
    assert len(set(control_tables)) == 1
    assert all(p.grad is None for p in model.parameters())
    y = np.array([r["target"] for r in rows])
    original, control = np.array(original_probabilities), np.array(control_probabilities)
    error = float(np.max(np.abs(original - [saved[i] for i in seen_ids])))
    if error > .002:
        raise RuntimeError(f"Fresh original predictions do not reproduce baseline: {error}")
    result = dict(status="passed", model=args.model, plan=PLAN, images=len(rows),
                  original=metrics(y, original), jpeg75=metrics(y, control),
                  fresh_original_max_difference_from_saved=error,
                  delta=dict(zip(("auroc", "balanced_accuracy", "real_false_positive_rate"), deltas(y, original, control).tolist())),
                  paired_group_bootstrap=paired_intervals(rows, y, original, control), by_source={},
                  test_and_qwen_images_read=0, weight_updates=0,
                  original_images_preserved=True, elapsed_seconds=time.perf_counter() - started)
    sources = np.array([r["source"] for r in rows])
    for source in sorted(set(sources)):
        selected_mask = (sources == source) | (sources == "real")
        source_mask = sources == source
        result["by_source"][source] = dict(n=int(source_mask.sum()),
            original_mean_probability=float(original[source_mask].mean()), jpeg75_mean_probability=float(control[source_mask].mean()),
            threshold_flips=int(np.sum((original[source_mask] >= .5) != (control[source_mask] >= .5))),
            original_quantization_tables=dict(Counter(np.array(original_tables)[source_mask])))
        if source != "real":
            result["by_source"][source].update(original=metrics(y[selected_mask], original[selected_mask]),
                                              jpeg75=metrics(y[selected_mask], control[selected_mask]))
    with (args.output / "paired_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, p, q in zip(rows, original, control):
            handle.write(json.dumps(dict(id=row["id"], split="validation", source=row["source"], target=row["target"],
                content_group_id=row["content_group_id"], original_probability=float(p), jpeg75_probability=float(q))) + "\n")
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"by_source", "plan"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
