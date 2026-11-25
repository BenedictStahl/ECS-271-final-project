"""Prepare a replacement-dataset inventory; preserve supplied splits, without approving them."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

PROVENANCE = Path(__file__).with_name("generation_provenance.json")
SPLITS = {"train", "validation", "test"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_rows(folder, reference_map, metadata_path=None, image_roots=None):
    folder = Path(folder).resolve()
    roots = [Path(p).resolve() for p in (image_roots or [folder])]
    metadata_path = Path(metadata_path or folder / "metadata.csv").resolve()
    generation = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    sources = set(generation["models"]) | {"real"}
    references = {}
    for line in Path(reference_map).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["reference_path"] in references:
            raise ValueError("Duplicate real reference in reference map")
        references[row["reference_path"]] = row
    metadata = {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            path = row["path"]
            parts = Path(path).parts
            if len(parts) == 4 and parts[0] in {"raw", "generated"}:
                if (parts[0] == "raw") != (row["source"] == "real"):
                    raise ValueError(f"Metadata prefix/source mismatch: {path}")
                parts = parts[1:]
                path = "/".join(parts)
            if len(parts) != 3 or parts[0] not in SPLITS or parts[1] not in sources:
                raise ValueError(f"Invalid metadata path: {path}")
            if (folder / path).resolve().is_relative_to(folder) is False:
                raise ValueError(f"Path outside dataset: {path}")
            if parts[:2] != (row["split"], row["source"]):
                raise ValueError(f"Metadata split/source mismatch: {path}")
            if path in metadata:
                raise ValueError(f"Duplicate metadata path: {path}")
            if int(row["label_a"]) != int(row["source"] != "real"):
                raise ValueError(f"Binary label mismatch: {path}")
            metadata[path] = row
    files = {}
    for root_number, root in enumerate(roots):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            relative = path.relative_to(root).as_posix()
            if relative in files:
                raise ValueError(f"Duplicate image across roots: {relative}")
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Image outside dataset root: {relative}")
            files[relative] = (path, root_number)
    records = []
    for relative, (path, root_number) in sorted(files.items()):
        parts = Path(relative).parts
        if len(parts) != 3 or parts[0] not in SPLITS or parts[1] not in sources:
            raise ValueError(f"Invalid image layout: {relative}")
        split, source, filename = parts
        reference_path = f"{split}/real/{filename}"
        if reference_path not in references or reference_path not in files:
            raise ValueError(f"Missing real reference for {relative}")
        reference = references[reference_path]
        index = reference["original_split_row_index"]
        if index != int(path.stem):
            raise ValueError(f"Reference index mismatch: {relative}")
        supplied = metadata.get(relative)
        if source != "real" and supplied is None:
            raise ValueError(f"Missing synthetic metadata: {relative}")
        if supplied and int(supplied["index"]) != index:
            raise ValueError(f"Metadata index mismatch: {relative}")
        # Conservative shared-content flag. A caption is not a unique photo ID.
        caption_group = hashlib.sha256(reference["original_caption"].encode()).hexdigest()
        records.append({"id": "replacement:" + relative, "path": relative,
                        "split": split, "source": source, "target": int(source != "real"),
                        "index": index, "caption": supplied["caption"] if supplied else reference["original_caption"],
                        "reference_id": reference["defactify_real_id"],
                        "reference_path": reference_path,
                        "reference_caption": reference["original_caption"],
                        "content_group_id": "original-caption:" + caption_group,
                        "assumed_seed": 42 + index if source not in {"real", "mage_flow"} else None,
                        "sha256": digest(path)})
        if image_roots:
            records[-1]["image_root"] = root_number
    if not records:
        raise ValueError("No replacement images found")
    paths = {r["path"] for r in records}
    missing = sorted(set(metadata) - paths)
    groups = defaultdict(set)
    for row in records:
        groups[row["content_group_id"]].add(row["split"])
    summary = {"dataset_root": str(folder), "rows": len(records),
               "counts": dict(Counter(r["split"] + "/" + r["source"] for r in records)),
               "unavailable_metadata_images": len(missing),
               "unavailable_by_source": dict(Counter(Path(p).parts[1] for p in missing)),
               "cross_split_content_groups": sum(len(s) > 1 for s in groups.values()),
               "metadata_sha256": digest(metadata_path),
               "metadata_file": str(metadata_path),
               "reference_map_sha256": digest(Path(reference_map)),
               "generation_provenance": generation,
               "status": "Inventory only: original splits preserved; exclusions and holdout not finalized",
               "notes": ["Real images occur once; do not concatenate a Defactify cache to this replacement cache.",
                         "Old CLS/attention-pooled features cannot be reused as mean-patch features.",
                         "content_group_id is a conservative caption flag, not a verified unique-photo identifier."]}
    if image_roots:
        summary["image_roots"] = [str(p) for p in roots]
    return records, summary


def indexed_rows(index, split=None, exclude_ids=None):
    index = Path(index).resolve()
    summary = json.loads(index.with_name("dataset_summary.json").read_text(encoding="utf-8"))
    root = Path(summary["dataset_root"]).resolve()
    for line in index.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if exclude_ids and row["id"] in exclude_ids:
            continue
        if split and row["split"] != split:
            continue
        image_root = Path(summary["image_roots"][row["image_root"]]).resolve() if "image_roots" in summary else root
        path = (image_root / row["path"]).resolve()
        if not path.is_relative_to(image_root):
            raise ValueError("Image path outside dataset root")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Image changed since index preparation: {row['path']}")
        yield dict(row, bytes=content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Supplied CSV; raw/generated prefixes are normalized in the index only.")
    parser.add_argument("--image-root", type=Path, action="append", help="Repeat for additive extraction folders; images are not copied.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory per snapshot.")
    rows, summary = prepare_rows(args.data_dir, args.reference_map, args.metadata, args.image_root)
    args.output.mkdir(parents=True)
    with (args.output / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "generation_provenance"}, indent=2))


if __name__ == "__main__":
    main()
