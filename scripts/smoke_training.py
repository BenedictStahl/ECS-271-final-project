"""Exercise every encoder and candidate holdout on a tiny, separate real-image subset.

These runs check the pipeline, not model quality or the final holdout choice.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(*args):
    output = Path(args[args.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run([sys.executable, *map(str, args)], stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed; inspect {log}\n{log.read_text(encoding='utf-8')[-4000:]}")
    print(f"Passed {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--local-dino", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = defaultdict(set)
    full = read_rows(args.manifests / "index.jsonl")
    for row in full:
        if row["source"] == "real" and len(selected[row["split"]]) < 4:
            selected[row["split"]].add(row["reference_id"])
    rows = [r for r in full if r["reference_id"] in selected[r["split"]]]
    ids = {r["id"] for r in rows}
    index = args.output / "index.jsonl"
    index.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = json.loads((args.manifests / "dataset_summary.json").read_text(encoding="utf-8"))
    summary.update(rows=len(rows), status="Tiny pipeline smoke subset; not scientific results")
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "review_decisions.json").write_bytes((args.manifests / "review_decisions.json").read_bytes())
    manifests = []
    for source in sorted(args.manifests.glob("holdout-*.json")):
        manifest = json.loads(source.read_text(encoding="utf-8"))
        for key in ("splits", "test_cohorts"):
            manifest[key] = {s: [i for i in values if i in ids] for s, values in manifest[key].items()}
        manifest.update(index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                        status="pilot_only", scope="Pipeline smoke test only; not a final holdout decision")
        target = args.output / source.name
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(target)
    scripts = Path(__file__).resolve().parent
    report = {"scope": "Tiny smoke test only; no scientific model comparison", "original_rows": len(rows), "models": {}}
    view_identity = None
    for model in ("deit3", "siglip2", "dinov3"):
        extra = ["--local-checkpoint", args.local_dino] if model == "dinov3" else []
        base, views = args.output / model / "base", args.output / model / "views"
        for folder, augmentation in ((base, []), (views, ["--augment-views", "1"])):
            run(scripts / "train_probe.py", "cache", "--model", model, "--replacement", index,
                "--offline", "--batch-size", "8", "--output", folder, *extra, *augmentation)
        view_rows = read_rows(views / "rows.jsonl")
        assert all(r["split"] == "train" for r in view_rows)
        identity = [(r["id"], r["view_pixel_sha256"]) for r in view_rows]
        if view_identity is None:
            view_identity = identity
        assert identity == view_identity, "Augmented pixels differ across encoders"
        result = {"original_features": list(np.load(base / "features.npy").shape),
                  "augmented_features": list(np.load(views / "features.npy").shape), "holdouts": {}}
        for manifest in manifests:
            output = args.output / model / manifest.stem
            run(scripts / "fit_probe.py", "--cache", base, "--cache", views,
                "--manifest", manifest, "--allow-pilot", "--output", output)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            augmentation = metrics["augmentation"]
            assert augmentation["original_training_rows"] == 20
            assert augmentation["augmented_training_rows"] == 20
            assert augmentation["cached_views_not_selected"] == 4
            assert metrics["validation"]["overall"]["n"] == 20
            assert metrics["test"]["overall"]["n"] == 24
            result["holdouts"][manifest.stem] = {"head_fit_passed": True, **augmentation}
        report["models"][model] = result
    report["same_augmented_pixels_across_models"] = True
    report["status"] = "passed"
    (args.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("All three encoders and five candidate holdouts passed the tiny pipeline smoke test.")


if __name__ == "__main__":
    main()
