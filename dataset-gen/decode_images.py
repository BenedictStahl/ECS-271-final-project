import argparse
import csv
import os
import shutil
from pathlib import Path

from tqdm import tqdm

DATASET_REPO_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "datasets"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw"
SOURCE_NAMES = ["real", "sd21", "sdxl", "sd3", "dalle3", "midjourney"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to a saved/cached dataset directory or Hugging Face dataset ID (defaults to local data/datasets or HF cache)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for decoded images and metadata.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to decode per split",
    )
    return parser.parse_args()


def has_hf_cache(directory: Path) -> bool:
    return (
        (directory / "datasets").is_dir()
        or (directory / "hub").is_dir()
        or any(p.is_dir() and "___" in p.name for p in directory.iterdir())
    )


def configure_hf_cache(cache_root: Path):
    os.environ["HF_HOME"] = str(cache_root)
    nested = cache_root / "datasets"
    os.environ["HF_DATASETS_CACHE"] = str(nested if nested.is_dir() else cache_root)


def resolve_source(row) -> str:
    label_b = row.get("Label_B")
    if isinstance(label_b, int) and 0 <= label_b < len(SOURCE_NAMES):
        return SOURCE_NAMES[label_b]
    if label_b is not None:
        return str(label_b)
    label_a = row.get("Label_A")
    return str(label_a) if label_a is not None else "unknown"


def load_splits(dataset_arg: str | None):
    dataset_path = Path(dataset_arg) if dataset_arg else DEFAULT_DATA_DIR

    if dataset_path.is_dir():
        if (dataset_path / "dataset_dict.json").exists() or (
            dataset_path / "dataset_info.json"
        ).exists():
            from datasets import Dataset, load_from_disk

            dataset = load_from_disk(str(dataset_path))
            return {"train": dataset} if isinstance(dataset, Dataset) else dataset

        split_dirs = [
            child
            for child in sorted(dataset_path.iterdir())
            if child.is_dir() and (child / "dataset_info.json").exists()
        ]
        if split_dirs:
            from datasets import load_from_disk

            return {child.name: load_from_disk(str(child)) for child in split_dirs}

        if has_hf_cache(dataset_path):
            configure_hf_cache(dataset_path)
            from datasets import Dataset, load_dataset

            dataset = load_dataset(DATASET_REPO_ID)
            return {"train": dataset} if isinstance(dataset, Dataset) else dataset

        if dataset_arg is None:
            from datasets import Dataset, load_dataset

            dataset = load_dataset(DATASET_REPO_ID)
            return {"train": dataset} if isinstance(dataset, Dataset) else dataset

    elif DEFAULT_DATA_DIR.is_dir() and has_hf_cache(DEFAULT_DATA_DIR):
        configure_hf_cache(DEFAULT_DATA_DIR)

    from datasets import Dataset, load_dataset

    dataset = load_dataset(dataset_arg or DATASET_REPO_ID)
    return {"train": dataset} if isinstance(dataset, Dataset) else dataset


def save_image(image_data, target_path: Path):
    if isinstance(image_data, dict):
        if image_data.get("bytes") is not None:
            target_path.write_bytes(image_data["bytes"])
            return
        if image_data.get("path"):
            shutil.copyfile(image_data["path"], target_path)
            return
    image_data.save(target_path)


def main():
    arguments = parse_arguments()
    splits = load_splits(arguments.dataset)

    from datasets import Image as HFImage

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = arguments.output_dir / "metadata.csv"

    created_dirs = set()
    total_saved = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["path", "split", "index", "label_a", "label_b", "source", "caption"],
        )
        writer.writeheader()

        for split_name, split in splits.items():
            if "Image" in split.column_names:
                image_col = "Image"
            elif "image" in split.column_names:
                image_col = "image"
            else:
                raise ValueError(
                    f"Split '{split_name}' has no image column (found: {split.column_names})"
                )
            raw_split = split.cast_column(image_col, HFImage(decode=False))
            if arguments.limit is not None:
                raw_split = raw_split.select(range(min(arguments.limit, len(raw_split))))

            for index, row in enumerate(
                tqdm(raw_split, total=len(raw_split), desc=f"Decoding {split_name}")
            ):
                source = resolve_source(row)
                image_data = row[image_col]
                orig_path = image_data.get("path") if isinstance(image_data, dict) else None
                ext = Path(orig_path).suffix.lower() if orig_path else ".jpg"
                ext = ext or ".jpg"

                rel_path = Path(split_name) / source / f"{index:05d}{ext}"
                out_path = arguments.output_dir / rel_path
                if out_path.parent not in created_dirs:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    created_dirs.add(out_path.parent)

                save_image(image_data, out_path)
                writer.writerow(
                    {
                        "path": rel_path.as_posix(),
                        "split": split_name,
                        "index": index,
                        "label_a": row.get("Label_A", ""),
                        "label_b": row.get("Label_B", ""),
                        "source": source,
                        "caption": row.get("Caption", ""),
                    }
                )
                total_saved += 1

    print(f"Decoded {total_saved} images to {arguments.output_dir}")


if __name__ == "__main__":
    main()
