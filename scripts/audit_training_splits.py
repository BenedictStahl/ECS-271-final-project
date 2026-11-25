"""Inventory exact duplicates and caption overlap; do not assign final splits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from train_probe import dataset_rows
from dataset_io import DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new filename.")
    hashes, captions, counts = defaultdict(list), defaultdict(list), Counter()
    for row in dataset_rows(args.data_dir):
        digest = hashlib.sha256(row.pop("bytes")).hexdigest()
        hashes[digest].append(row)
        if row["caption"]:
            captions[row["caption"]].append(row)
        counts[f'{row["split"]}/{row["source"]}'] += 1
    duplicates = [{"sha256": digest, "rows": rows} for digest, rows in hashes.items() if len(rows) > 1]
    cross_split = lambda rows: len({r["split"] for r in rows}) > 1
    caption_overlap = [{"caption": caption, "rows": rows} for caption, rows in captions.items()
                       if cross_split(rows)]
    report = {"counts": dict(counts), "exact_duplicate_groups": duplicates,
              "cross_split_exact_duplicate_groups": [g for g in duplicates if cross_split(g["rows"])],
              "cross_split_caption_groups": caption_overlap,
              "limitations": "Encoded-byte hashes only. Recompressed images and augmentation families may differ. "
                             "Equal captions suggest related content but do not establish identical images. "
                             "No split exclusions or held-out generator chosen."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rows: {sum(counts.values())}; duplicate groups: {len(duplicates)}; "
          f"cross-split duplicate groups: {len(report['cross_split_exact_duplicate_groups'])}; "
          f"cross-split caption groups: {len(caption_overlap)}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
