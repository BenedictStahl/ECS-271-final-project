"""Package the completed November 27 grid for private teammate evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import torch

from dataset_io import REPO_ROOT
from train_probe import CHECKPOINTS


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for transfer archives")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    grid = REPO_ROOT / "data/runs/finetuning-grid-2026-11-27"
    status = json.loads((grid / "grid_status.json").read_text())
    if status["status"] != "complete" or len(status["runs"]) != 12:
        raise ValueError("Grid must be complete before selecting models")
    handoff = REPO_ROOT / "data/evaluation-handoff"
    handoff.mkdir(exist_ok=False)
    manifest_path = REPO_ROOT / "data/replacement-holdouts-2026-11-25/holdout-qwen_image_2_1.json"
    manifest = json.loads(manifest_path.read_text())
    index_path = manifest_path.parent / manifest["index_file"]
    assert sha(index_path) == manifest["index_sha256"]
    index = {r["id"]: r for r in map(json.loads, index_path.read_text().splitlines())}
    summary = json.loads(index_path.with_name("dataset_summary.json").read_text())
    rows = []
    for image_id in manifest["splits"]["test"]:
        row = index[image_id]
        entry = {k: row[k] for k in ["id", "split", "source", "target", "reference_id", "content_group_id", "sha256"]}
        entry["image_file"] = "images/" + row["path"]
        rows.append(entry)
    (handoff / "test_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    shutil.copy2(manifest_path, handoff / "original_split_manifest.json")
    shutil.copy2(grid / "protocol.json", handoff / "training_protocol.json")
    results = []
    selections = {}
    for run in status["runs"]:
        folder = Path(run["output"])
        history = json.loads((folder / "history.json").read_text())
        best = max(history, key=lambda h: h["validation"]["auroc"])
        results.append(dict(configuration=run["configuration"], model=run["model"], history=history,
                            recipe=json.loads(Path(run["config"]).read_text()), best_epoch=best["epoch"],
                            best_validation=best["validation"], seconds=sum(h["epoch_total_seconds"] for h in history)))
    for name in CHECKPOINTS:
        candidates = [r for r in results if r["model"] == name]
        chosen = max(candidates, key=lambda r: r["best_validation"]["auroc"])
        run = next(r for r in status["runs"] if r["model"] == name and r["configuration"] == chosen["configuration"])
        original = Path(run["output"]) / "best.pt"
        payload = torch.load(original, map_location="cpu", weights_only=True)
        assert payload["metadata"]["validation"] == chosen["best_validation"]
        # Inference needs trainable deltas and metadata, not optimizer/scaler/scheduler state.
        target = handoff / f"{name}-finetuned.pt"
        torch.save({k: payload[k] for k in ["format_version", "trainable_state", "metadata"]}, target)
        head = handoff / f"{name}-frozen-head.npz"
        shutil.copy2(REPO_ROOT / f"data/runs/frozen-qwen-holdout-2026-11-25/{name}/linear_head.npz", head)
        selections[name] = dict(checkpoint=target.name, frozen_head=head.name,
            selection=dict(configuration=chosen["configuration"], recipe=chosen["recipe"],
                           epoch=chosen["best_epoch"], validation=chosen["best_validation"],
                           original_checkpoint_sha256=sha(original)), base_checkpoint=list(CHECKPOINTS[name]))
    (handoff / "validation_grid_results.json").write_text(json.dumps(results, indent=2))
    tracked = REPO_ROOT / "results/finetuning-grid-2026-11-27"
    tracked.mkdir(parents=True, exist_ok=True)
    shutil.copy2(handoff / "validation_grid_results.json", tracked / "validation_grid_results.json")
    (tracked / "selected_models.json").write_text(json.dumps(selections, indent=2))
    bases = []
    for name in CHECKPOINTS:
        folder = REPO_ROOT / f"data/models/{name}"
        if name == "dinov3":
            folder.mkdir(exist_ok=False)
            source = REPO_ROOT.parent / "dino_inspection_2026-11-25/dinov3-base"
            for filename in ["config.json", "model.safetensors", "LICENSE.md", "MANIFEST.json"]:
                shutil.copy2(source / filename, folder / filename)
        for path in folder.iterdir():
            if path.is_file() and (path.suffix == ".safetensors" or path.name in ["config.json", "model.safetensors.index.json", "LICENSE.md", "MANIFEST.json"]):
                bases.append(dict(path=path.relative_to(REPO_ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    files = [dict(path=p.relative_to(handoff).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(handoff.iterdir()) if p.is_file()]
    package = dict(format_version=1, selection_metric="seen_validation_auroc", threshold=.5,
                   held_out_generator="qwen_image_2_1", test_counts=dict(Counter(r["source"] for r in rows)),
                   original_manifest_sha256=sha(manifest_path), files=files, base_files=bases, models=selections,
                   test_status="Not evaluated locally; archives contain inputs, not test predictions",
                   historical_exposure="Earlier tiny pipeline smoke tests included test examples; primary full trained-model test evaluation remains pending.")
    (handoff / "handoff.json").write_text(json.dumps(package, indent=2))
    with zipfile.ZipFile(args.output / "evaluation-model-assets.zip", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for p in handoff.iterdir():
            if p.is_file(): archive.write(p, p.relative_to(REPO_ROOT).as_posix())
        for entry in bases:
            archive.write(REPO_ROOT / entry["path"], entry["path"])
    with zipfile.ZipFile(args.output / "evaluation-test-images.zip", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for i, row in enumerate(rows):
            original = index[row["id"]]
            source = Path(summary["image_roots"][original["image_root"]]) / original["path"]
            if sha(source) != row["sha256"]:
                raise ValueError(f"Test image hash differs: {row['id']}")
            archive.write(source, "data/evaluation-handoff/" + row["image_file"])
            if (i+1) % 5000 == 0: print(f"Packed {i+1}/{len(rows)} test images", flush=True)
    for filename in ["evaluation-model-assets.zip", "evaluation-test-images.zip"]:
        path = args.output / filename
        with (args.output / "SHA256SUMS.txt").open("a") as stream:
            stream.write(f"{sha(path)}  {filename}\n")
        print(filename, path.stat().st_size, flush=True)
    shutil.copy2(REPO_ROOT / "docs/EVALUATION_HANDOFF.md", args.output / "README_FIRST.md")


if __name__ == "__main__":
    main()
