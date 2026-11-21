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
        default=DATASET_REPO_ID,
        help="Hugging Face dataset ID or local path",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Hugging Face cache directory",
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


def resolve_datasets_cache(data_dir: Path) -> Path:
    nested = data_dir / "datasets"
    return nested if nested.is_dir() else data_dir


def resolve_source(row) -> str:
    label_b = row.get("Label_B")
    if isinstance(label_b, int) and 0 <= label_b < len(SOURCE_NAMES):
        return SOURCE_NAMES[label_b]
    if label_b is not None:
        return str(label_b)
    label_a = row.get("Label_A")
    return str(label_a) if label_a is not None else "unknown"


def load_splits(dataset_name: str):
    from datasets import Dataset, load_dataset, load_from_disk

    dataset_path = Path(dataset_name)
    if dataset_path.is_dir() and (
        (dataset_path / "dataset_dict.json").exists()
        or (dataset_path / "dataset_info.json").exists()
    ):
        dataset = load_from_disk(str(dataset_path))
    else:
        dataset = load_dataset(dataset_name)

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
    os.environ["HF_HOME"] = str(arguments.data_dir)
    os.environ["HF_DATASETS_CACHE"] = str(resolve_datasets_cache(arguments.data_dir))

    from datasets import Image as HFImage

    splits = load_splits(arguments.dataset)
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
            image_col = "Image" if "Image" in split.column_names else "image"
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
