"""Shared paths for the Defactify download and compatibility inspection."""
from pathlib import Path

DATASET_REPO_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
DATASET_REVISION = "1c3b714c9691eb6dd11c2c1e92eab1ac29584fa0"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_REPORT = REPO_ROOT / "data" / "dataset_compatibility_profile.json"
SPLITS = ("train", "validation", "test")
SOURCE_NAMES = ("real", "sd21", "sdxl", "sd3", "dalle3", "midjourney")


def split_files(directory):
    """Accept either a download root containing data/ or a shard directory."""
    directory = Path(directory).expanduser().resolve()
    shard_dir = directory / "data" if (directory / "data").is_dir() else directory
    result = {split: sorted(shard_dir.glob(f"{split}-*.parquet")) for split in SPLITS}
    for split, paths in result.items():
        if not paths:
            raise FileNotFoundError(f"No {split} Parquet shards in {shard_dir}. Run scripts/download_dataset.py first or pass --data-dir.")
        # Hugging Face filenames encode shard index and expected total.
        parts = [path.stem.split("-") for path in paths]
        totals = {int(p[-1]) for p in parts}
        if len(totals) != 1 or {int(p[-3]) for p in parts} != set(range(next(iter(totals)))):
            raise ValueError(f"Incomplete or inconsistent {split} shard set in {shard_dir}.")
    return result
