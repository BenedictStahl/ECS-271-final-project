"""Profile local image inputs without network access or an additional Arrow cache."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import time

from dataset_io import DEFAULT_DATA_DIR, DEFAULT_REPORT, SOURCE_NAMES, split_files


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--decode-samples", type=int, default=5, help="Fully decode/convert/resize this many images per source per split; headers are inspected for every row.")
    args = parser.parse_args()
    if args.decode_samples < 0:
        parser.error("--decode-samples must be nonnegative")
    return args


def summarize(group):
    sizes = [(tuple(map(int, size.split("x"))), count) for size, count in group["sizes"].items()]
    group["distinct_sizes"] = len(sizes)
    group["top_sizes"] = group["sizes"].most_common(10)
    group["width_range"] = [min(s[0][0] for s in sizes), max(s[0][0] for s in sizes)] if sizes else None
    group["height_range"] = [min(s[0][1] for s in sizes), max(s[0][1] for s in sizes)] if sizes else None
    group["aspect_ratio_range"] = [min(w / h for (w, h), _ in sizes), max(w / h for (w, h), _ in sizes)] if sizes else None
    for edge in (224, 384):
        group[f"short_edge_below_{edge}"] = sum(n for (w, h), n in sizes if min(w, h) < edge)


def main():
    args = arguments()
    import pyarrow.parquet as pq
    from PIL import Image

    files = split_files(args.data_dir)
    started = time.perf_counter()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("pillow", "pyarrow", "huggingface-hub")},
        "method": "All-row header scan; first N successful full decode/RGB conversion/224 resize checks per source per split. Not a full corruption or leakage audit.",
        "decode_samples_requested_per_source_per_split": args.decode_samples,
        "source_names": dict(enumerate(SOURCE_NAMES)),
        "parquet_bytes": sum(p.stat().st_size for paths in files.values() for p in paths),
        "splits": {},
    }
    manifest_path = args.data_dir.expanduser().resolve() / "download_manifest.json"
    if manifest_path.exists():
        report["download_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        # Older Hub snapshots use their immutable revision as the directory name.
        root = args.data_dir.expanduser().resolve()
        if root.name == "data":
            root = root.parent
        report["snapshot_directory"] = str(root)
    for split, paths in files.items():
        groups, pairs = {}, Counter()
        errors, label_errors = [], []
        index = 0
        schemas = []
        for path in paths:
            parquet = pq.ParquetFile(path)
            schemas.append(str(parquet.schema_arrow))
            for batch in parquet.iter_batches(batch_size=64, columns=["Image", "Label_A", "Label_B"]):
                for row in batch.to_pylist():
                    label = row["Label_B"]
                    pairs[f"{row['Label_A']},{label}"] += 1
                    if label not in range(len(SOURCE_NAMES)) or row["Label_A"] != int(label != 0):
                        label_errors.append({"index": index, "Label_A": row["Label_A"], "Label_B": label})
                    group = groups.setdefault(str(label), {"count": 0, "sizes": Counter(), "modes": Counter(), "formats": Counter(), "bytes": 0, "decoded_samples": 0, "path_only": 0})
                    group["count"] += 1
                    try:
                        raw = row["Image"]
                        if raw["bytes"] is None:
                            group["path_only"] += 1
                            external = Path(raw["path"])
                            contents = (external if external.is_absolute() else path.parent / external).read_bytes()
                        else:
                            contents = raw["bytes"]
                        group["bytes"] += len(contents)
                        with Image.open(io.BytesIO(contents)) as image:
                            group["sizes"][f"{image.width}x{image.height}"] += 1
                            group["modes"][image.mode] += 1
                            group["formats"][image.format] += 1
                            if group["decoded_samples"] < args.decode_samples:
                                image.load()
                                with image.convert("RGB") as rgb:
                                    with rgb.resize((224, 224)) as resized:
                                        resized.load()
                                group["decoded_samples"] += 1
                    except Exception as exc:
                        errors.append({"index": index, "label": label, "error": str(exc)})
                    index += 1
        for group in groups.values():
            summarize(group)
        report["splits"][split] = {"rows": index, "parquet_files": [str(p) for p in paths], "schemas": sorted(set(schemas)), "label_pairs": pairs, "label_errors": label_errors, "groups": groups, "errors": errors}
        print(f"{split}: {index:,} rows; {len(errors)} image errors; {len(label_errors)} label errors", flush=True)
    report["inspection_seconds"] = time.perf_counter() - started
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output}; elapsed {report['inspection_seconds']:.1f}s")
    if any(s["errors"] or s["label_errors"] for s in report["splits"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
