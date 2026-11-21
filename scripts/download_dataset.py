import argparse
import os
from pathlib import Path
from datasets import enable_progress_bar, load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.utils import enable_progress_bars

DATASET_REPO_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "datasets"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    os.environ["HF_HOME"] = str(arguments.data_dir)

    from datasets import load_dataset
    enable_progress_bars()
    enable_progress_bar()

    print(f"Downloading {DATASET_REPO_ID} to {arguments.data_dir}...")
    snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset")

    dataset = load_dataset(DATASET_REPO_ID)
    print(dataset)


if __name__ == "__main__":
    main()