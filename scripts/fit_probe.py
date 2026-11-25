"""Fit a balanced logistic linear head from cached features and an explicit split manifest.

Manifest JSON: {"held_out_generator": "...",
"splits": {"train": [image IDs], "validation": [image IDs], "test": [image IDs]}}.
Supplemental cache provenance identifies which images were newly generated.
Do not create this manifest until related generations/perturbations are grouped.
"""
import argparse
import copy
import hashlib
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def validate_splits(rows, manifest, supplemental_ids):
    held_out = manifest["held_out_generator"]
    if not held_out or held_out == "real":
        raise ValueError("Choose a synthetic held-out generator.")
    assignments, hash_splits, caption_splits, reference_splits, content_splits = {}, {}, {}, {}, {}
    training_ids = set(manifest["splits"]["train"])
    indices = {}
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate IDs across caches.")
    for split in ("train", "validation", "test"):
        ids = manifest["splits"][split]
        if not ids:
            raise ValueError(f"Empty {split} split")
        indices[split] = []
        for image_id in ids:
            if image_id in assignments:
                raise ValueError(f"Image ID assigned more than once: {image_id}")
            assignments[image_id] = split
            row = rows[by_id[image_id]]
            if row.get("parent_id"):
                parent_id = row["parent_id"]
                if split != "train" or parent_id not in training_ids or parent_id not in by_id:
                    raise ValueError("Augmentation must follow an included training parent")
                from training_augmentation import checked_view_row
                checked_view_row(row, rows[by_id[parent_id]])
            if row["split"] not in ("unassigned", split):
                raise ValueError("Published dataset rows cannot move to another split; exclude conflicts instead.")
            if split != "test" and row["source"] == held_out:
                raise ValueError("Held-out generator appears in training or validation.")
            reviewed = row.get("reviewed_family_id")
            approval = manifest.get("reference_family_overrides", {}).get(row.get("reference_id"))
            if reviewed or approval:
                if (not reviewed or not approval or reviewed != approval["family_id"]
                        or row.get("content_group_id") != reviewed
                        or row.get("family_review_sha256") != manifest.get("review_sha256")):
                    raise ValueError("Reviewed family is not authorized by this manifest")
                if row["source"] == "real" and row["sha256"] != approval["real_sha256"]:
                    raise ValueError("Reviewed real-image hash changed")
            # Exact caption equality remains a conservative guard unless the
            # recorded visual review established distinct photo identities.
            caption = None if reviewed else row.get("caption")
            for value, groups in ((row["sha256"], hash_splits), (caption, caption_splits),
                                  (row.get("reference_id"), reference_splits),
                                  (row.get("content_group_id"), content_splits)):
                if value:
                    if value in groups and groups[value] != split:
                        raise ValueError("An image, caption, or reference/content group crosses splits.")
                    groups[value] = split
            indices[split].append(by_id[image_id])
        if {rows[i]["target"] for i in indices[split]} != {0, 1}:
            raise ValueError(f"{split} must contain both real and synthetic images.")
    if not any(image_id in supplemental_ids for image_id in manifest["splits"]["train"]):
        raise ValueError("Primary experiment requires supplemental images in training.")
    if not any(rows[i]["source"] == held_out for i in indices["test"]):
        raise ValueError("Test must contain the held-out generator.")
    return indices


def check_manifest_status(manifest, allow_pilot=False):
    status = manifest.get("status", "ready")
    if status == "pilot_only" and not allow_pilot:
        raise ValueError("Incomplete five-generator dataset: use --allow-pilot only for an intentional pilot.")
    if status not in {"ready", "pilot_only"}:
        raise ValueError(f"Manifest is blocked: {status}")


def canonical_cache_rows(rows, expected):
    """Reuse features for unchanged images while adopting the reviewed grouping.

    This also permits later additive caches without rerunning old image features.
    Encoder/pooling/transform compatibility is checked separately.
    """
    if expected is None:
        if any(r.get("parent_id") for r in rows):
            raise ValueError("Augmented caches require a canonical replacement index")
        return rows
    result = []
    for row in rows:
        if row.get("parent_id"):
            from training_augmentation import checked_view_row
            parent = expected.get(row["parent_id"])
            if parent is None:
                raise ValueError("Unknown augmentation parent")
            result.append(checked_view_row(row, parent))
            continue
        canonical = expected.get(row["id"])
        keys = ("id", "sha256", "split", "source", "target", "reference_id")
        if canonical is None or any(row.get(k) != canonical.get(k) for k in keys):
            raise ValueError(f"Cache image identity differs from canonical index: {row['id']}")
        result.append(canonical)
    return result


def include_training_views(rows, manifest):
    """Add only views whose ORIGINAL parent is permitted by this holdout manifest."""
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Repeated original/view IDs across caches")
    parents = set(manifest["splits"]["train"])
    if any(p not in by_id or by_id[p].get("parent_id") for p in parents):
        raise ValueError("Original training parent features are required; use the original shared manifest")
    views = [r for r in rows if r.get("parent_id")]
    selected = [r for r in views if r["parent_id"] in parents]
    counts = dict.fromkeys(parents, 0)
    for row in selected:
        counts[row["parent_id"]] += 1
    if views and (not selected or len(set(counts.values())) != 1):
        raise ValueError("Incomplete augmentation coverage: every selected training parent needs the same number of views")
    expanded = copy.deepcopy(manifest)
    expanded["splits"]["train"] += [r["id"] for r in selected]
    # Each parent keeps total loss weight one, including its original view.
    # Adding views therefore does not silently multiply its contribution or C.
    weights = [1. / (1 + counts[by_id[i].get("parent_id") or i]) for i in expanded["splits"]["train"]]
    info = {"original_training_rows": len(parents), "augmented_training_rows": len(selected),
            "cached_views_not_selected": len(views) - len(selected),
            "views_per_training_parent": next(iter(counts.values()), 0),
            "loss_weighting": "Each original parent totals weight one across original and augmented views"}
    return expanded, np.asarray(weights), info


def metrics(targets, probabilities):
    predictions = probabilities >= 0.5
    return {"n": len(targets),
            "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
            "auroc": float(roc_auc_score(targets, probabilities)),
            "real_false_positive_rate": float(np.mean(predictions[targets == 0])),
            "confusion_matrix_real_fake": confusion_matrix(targets, predictions, labels=[0, 1]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True,
                        help="Repeat for Defactify and supplemental feature folders from the SAME encoder.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-pilot", action="store_true", help="Explicitly allow incomplete-snapshot pilot manifests.")
    parser.add_argument("--defer-test", action="store_true", help="Train and evaluate validation only; do not predict on test images.")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("CPU thread count must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new folder.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    check_manifest_status(manifest, args.allow_pilot)
    if manifest.get("review_file"):
        review_bytes = (args.manifest.resolve().parent / manifest["review_file"]).read_bytes()
        if hashlib.sha256(review_bytes).hexdigest() != manifest["review_sha256"]:
            raise ValueError("Review checksum differs from manifest")
        if json.loads(review_bytes)["reference_family_overrides"] != manifest["reference_family_overrides"]:
            raise ValueError("Manifest family overrides differ from signed-off review")
    expected = None
    if manifest.get("index_file"):
        index = args.manifest.resolve().parent / manifest["index_file"]
        content = index.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["index_sha256"]:
            raise ValueError("Canonical index checksum differs from manifest")
        canonical = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        expected = {r["id"]: r for r in canonical}
        if len(expected) != len(canonical):
            raise ValueError("Duplicate canonical index IDs")
    rows, arrays, models, supplemental_ids, augmentation_recipes = [], [], [], set(), []
    for folder in args.cache:
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        if provenance.get("pooling") != "mean_patch_v1":
            raise ValueError("Requires mean_patch_v1 features; old native-pooling caches are incompatible.")
        models.append((provenance["model"], provenance.get("geometry"), provenance["transform"]))
        if provenance["limit"] is not None:
            raise ValueError("Smoke-test caches cannot be used for training.")
        cache_rows = [json.loads(line) for line in (folder / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        augmentation = provenance.get("augmentation")
        if augmentation:
            from training_augmentation import PIPELINE
            if augmentation.get("pipeline") != PIPELINE or not augmentation.get("recipe_id"):
                raise ValueError("Unknown augmentation recipe")
            if not all(r.get("parent_id") for r in cache_rows):
                raise ValueError("Additional-view caches cannot silently include original rows")
            augmentation_recipes.append(augmentation)
        elif any(r.get("parent_id") for r in cache_rows):
            raise ValueError("View cache lacks augmentation provenance")
        cache_rows = canonical_cache_rows(cache_rows, expected)
        features = np.load(folder / "features.npy")
        if features.shape != (len(cache_rows), 768) or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature cache: {folder}")
        if provenance.get("supplement") or provenance.get("replacement_index"):
            supplemental_ids.update(row["id"] for row in cache_rows if row["target"] == 1)
        rows.extend(cache_rows)
        arrays.append(features)
    if any(model != models[0] for model in models):
        raise ValueError("Caches use different checkpoints or transforms.")
    if len({r["recipe_id"] for r in augmentation_recipes}) > 1:
        raise ValueError("Caches contain different augmentation recipes")
    expanded_manifest, sample_weights, augmentation_info = include_training_views(rows, manifest)
    indices = validate_splits(rows, expanded_manifest, supplemental_ids)
    x = np.concatenate(arrays)
    y = np.array([row["target"] for row in rows])
    # Scaling uses training rows only. No hyperparameter/threshold search yet:
    # fixed C=1 and probability threshold 0.5 keep this first baseline simple.
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=2000))
    started = time.perf_counter()
    with threadpool_limits(limits=args.cpu_threads), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if augmentation_info["augmented_training_rows"]:
            head.fit(x[indices["train"]], y[indices["train"]],
                     standardscaler__sample_weight=sample_weights,
                     logisticregression__sample_weight=sample_weights)
        else:
            head.fit(x[indices["train"]], y[indices["train"]])
    results = {"checkpoint_and_transform": models[0], "threshold": 0.5,
               "manifest": manifest, "validation": {}, "test": {},
               "test_evaluation": "deferred; no test predictions computed" if args.defer_test else "performed",
               "training": {"fit_seconds": time.perf_counter() - started, "cpu_threads": args.cpu_threads,
                            "iterations": head[-1].n_iter_.tolist(), "converged": True,
                            "training_feature_rows": len(indices["train"]),
                            "class_counts": dict(zip(*[a.tolist() for a in np.unique(y[indices["train"]], return_counts=True)])),
                            "C": 1, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000,
                            "cache_folders": [str(p.resolve()) for p in args.cache]}}
    parent_ids = set(manifest["splits"]["train"])
    results["augmentation"] = dict(augmentation_info, recipes=augmentation_recipes,
                                   selected_view_ids=[i for i in expanded_manifest["splits"]["train"] if i not in parent_ids])
    predictions = []
    for split in (("validation",) if args.defer_test else ("validation", "test")):
        selected = indices[split]
        probabilities = head.predict_proba(x[selected])[:, 1]
        targets = y[selected]
        predictions.extend(dict(id=rows[i]["id"], split=split, source=rows[i]["source"],
                                target=int(y[i]), probability=float(p)) for i, p in zip(selected, probabilities))
        results[split]["overall"] = metrics(targets, probabilities)
        sources = np.array([rows[i]["source"] for i in selected])
        for source in sorted(set(sources) - {"real"}):
            mask = (sources == "real") | (sources == source)
            results[split][source + "_vs_real"] = metrics(targets[mask], probabilities[mask])
        if split == "test":
            held_out = manifest["held_out_generator"]
            held_mask = (sources == "real") | (sources == held_out)
            seen_mask = sources != held_out
            results[split]["held_out_vs_real"] = metrics(targets[held_mask], probabilities[held_mask])
            if len(set(targets[seen_mask])) == 2:
                results[split]["seen_vs_real"] = metrics(targets[seen_mask], probabilities[seen_mask])
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output / "head.pkl").open("wb") as handle:
        pickle.dump(head, handle)
    # Fold train-only affine scaling into one 768->1 layer for patch-score XAI.
    scaler, classifier = head.steps[0][1], head.steps[1][1]
    weight = classifier.coef_[0] / scaler.scale_
    bias = classifier.intercept_[0] - weight @ scaler.mean_
    np.testing.assert_allclose(x[indices["validation"]] @ weight + bias,
                               head.decision_function(x[indices["validation"]]), rtol=1e-5, atol=1e-5)
    np.savez(args.output / "linear_head.npz", weight=weight, bias=bias, pooling="mean_patch_v1")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"training": results["training"], "validation": results["validation"],
                      "test_evaluation": results["test_evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
