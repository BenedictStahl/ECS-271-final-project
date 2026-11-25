"""Apply reviewed reference families and build shared leave-one-generator-out manifests."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from replacement_dataset import PROVENANCE


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_additive_rows(previous, current):
    """A later delivery may add rows, but cannot silently change reviewed inputs."""
    by_id = {r["id"]: r for r in current}
    keys = ("id", "path", "sha256", "split", "source", "target", "reference_id", "reference_caption", "content_group_id")
    for old in previous:
        new = by_id.get(old["id"])
        if new is None or any(old.get(k) != new.get(k) for k in keys):
            raise ValueError(f"Additive delivery changed/removed an earlier row: {old['id']}")


def apply_review(rows, review, review_hash):
    """Bind review decisions to real-file hashes, then propagate to every generator."""
    overrides = review["reference_family_overrides"]
    real = {r["reference_id"]: r for r in rows if r["source"] == "real"}
    if len(real) != sum(r["source"] == "real" for r in rows):
        raise ValueError("Repeated real reference IDs")
    for reference_id, decision in overrides.items():
        row = real.get(reference_id)
        if row is None or (row["path"], row["sha256"], row["split"]) != (
                decision["real_path"], decision["real_sha256"], decision["split"]):
            raise ValueError(f"Stale or mismatched review: {reference_id}")
    excluded_refs = set(review["excluded_reference_ids"])
    excluded_ids = set(review["excluded_image_ids"])
    if excluded_refs - set(real) or excluded_ids - {r["id"] for r in rows}:
        raise ValueError("Review exclusion refers to an unknown image/reference")
    kept, excluded = [], []
    for row in rows:
        row = dict(row)
        if row["reference_id"] in excluded_refs or row["id"] in excluded_ids:
            excluded.append(row["id"])
            continue
        if row["reference_id"] not in real or row["split"] != real[row["reference_id"]]["split"]:
            raise ValueError(f"Reference/split mismatch: {row['id']}")
        if row["reference_id"] in overrides:
            decision = overrides[row["reference_id"]]
            if decision["decision"] != "keep":
                raise ValueError("Unsupported review decision; exclusions must be explicit")
            row["caption_group_id"] = row["content_group_id"]
            row["content_group_id"] = decision["family_id"]
            row["reviewed_family_id"] = decision["family_id"]
            row["family_review_sha256"] = review_hash
        kept.append(row)
    if len({r["id"] for r in kept}) != len(kept):
        raise ValueError("Duplicate image IDs")
    return kept, excluded


def make_manifests(rows, generators, review, review_hash):
    from fit_probe import validate_splits

    generators = list(generators)
    expected = {r["reference_id"] for r in rows if r["source"] == "real"}
    coverage, pairs = defaultdict(set), set()
    for row in rows:
        if row["source"] not in {"real", *generators} or row["target"] != int(row["source"] != "real"):
            raise ValueError("Unknown source or incorrect binary label")
        pair = (row["source"], row["reference_id"])
        if pair in pairs:
            raise ValueError("Source has repeated images for a reference")
        pairs.add(pair)
        coverage[row["source"]].add(row["reference_id"])
    if not expected:
        raise ValueError("No real references")
    complete = [g for g in generators if coverage[g] == expected]
    incomplete = [g for g in generators if g not in complete]
    manifests = {}
    for held_out in generators:
        status = ("blocked_incomplete_holdout" if held_out not in complete else
                  "blocked_no_training_generator" if len(complete) < 2 else
                  "ready" if not incomplete else "pilot_only")
        selected = {s: [] for s in ("train", "validation", "test")}
        cohorts = {s: [] for s in ("real", "held_out_fake", "seen_fake")}
        if status in {"ready", "pilot_only"}:
            for row in rows:
                if row["source"] not in {"real", *complete}:
                    continue
                if row["split"] != "test" and row["source"] == held_out:
                    continue
                selected[row["split"]].append(row["id"])
                if row["split"] == "test":
                    cohort = "real" if row["source"] == "real" else "held_out_fake" if row["source"] == held_out else "seen_fake"
                    cohorts[cohort].append(row["id"])
        manifest = {"format_version": 1, "held_out_generator": held_out, "status": status,
                    "expected_generators": generators, "complete_generators": complete,
                    "incomplete_generators": incomplete,
                    "training_generators": [g for g in complete if g != held_out],
                    "review_sha256": review_hash,
                    "reference_family_overrides": review["reference_family_overrides"],
                    "splits": selected, "test_cohorts": cohorts,
                    "protocol": "Preserve supplied splits; holdout absent from train/validation; test seen and unseen against the same real set",
                    "scope": "Five-generator experiment" if status == "ready" else "Incomplete snapshot; not the final five-generator experiment"}
        if status in {"ready", "pilot_only"}:
            validate_splits(rows, manifest, {r["id"] for r in rows if r["target"] == 1})
        manifests[held_out] = manifest
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-base-index", type=Path,
                        help="Original reviewed inventory, required when regenerating for an additive delivery.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory")
    rows = [json.loads(line) for line in args.index.read_text(encoding="utf-8").splitlines()]
    review = json.loads(args.review.read_text(encoding="utf-8"))
    binding = "exact_reviewed_inventory"
    if review["original_index_sha256"] != sha256(args.index):
        if not args.review_base_index or sha256(args.review_base_index) != review["original_index_sha256"]:
            raise ValueError("Different input index: supply the original --review-base-index for an additive delivery")
        previous = [json.loads(line) for line in args.review_base_index.read_text(encoding="utf-8").splitlines()]
        verify_additive_rows(previous, rows)
        binding = "additive_inventory_verified_against_original"
    if review["summary"]["unresolved_flags"]:
        raise ValueError("Reference review has unresolved flags")
    review_hash = sha256(args.review)
    reviewed, excluded = apply_review(rows, review, review_hash)
    generators = list(json.loads(PROVENANCE.read_text(encoding="utf-8"))["models"])
    manifests = make_manifests(reviewed, generators, review, review_hash)
    args.output.mkdir(parents=True)
    reviewed_index = args.output / "index.jsonl"
    with reviewed_index.open("w", encoding="utf-8") as handle:
        for row in reviewed:
            handle.write(json.dumps(row) + "\n")
    index_hash = sha256(reviewed_index)
    summary = json.loads(args.index.with_name("dataset_summary.json").read_text(encoding="utf-8"))
    summary.update(rows=len(reviewed), counts=dict(Counter(r["split"] + "/" + r["source"] for r in reviewed)),
                   review_sha256=review_hash, review_binding=binding, input_index_sha256=sha256(args.index),
                   cross_split_content_groups=0, excluded_image_ids=excluded,
                   status=("Reviewed index; all five holdouts ready" if all(m["status"] == "ready" for m in manifests.values())
                           else "Caption review applied; pilot manifests only until all five generators are complete"))
    summary["notes"].append("Reviewed photo-family IDs replace caption identity only for hash-bound approved references; raw caption_group_id remains available.")
    groups = defaultdict(set)
    for row in reviewed:
        groups[row["content_group_id"]].add(row["split"])
    summary["cross_split_content_groups"] = sum(len(v) > 1 for v in groups.values())
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "review_decisions.json").write_bytes(args.review.read_bytes())
    report = {"rows": len(reviewed), "excluded_images": len(excluded),
              "reviewed_rows": sum("reviewed_family_id" in r for r in reviewed),
              "cross_split_content_groups_remaining": summary["cross_split_content_groups"], "holdouts": {}}
    for generator, manifest in manifests.items():
        manifest.update(index_file="index.jsonl", index_sha256=index_hash, review_file="review_decisions.json")
        filename = f"holdout-{generator}.json"
        (args.output / filename).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report["holdouts"][generator] = {"status": manifest["status"],
            "split_counts": {s: len(ids) for s, ids in manifest["splits"].items()},
            "test_cohort_counts": {s: len(ids) for s, ids in manifest["test_cohorts"].items()},
            "file": filename}
    (args.output / "manifest_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
