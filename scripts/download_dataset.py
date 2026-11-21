import argparse
from pathlib import Path
from datasets import enable_progress_bar, load_dataset
from huggingface_hub.utils import enable_progress_bars

DATASET_REPO_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "datasets"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    output_dir = arguments.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    enable_progress_bars()
    enable_progress_bar()

    print(f"Downloading {DATASET_REPO_ID}...")
    dataset = load_dataset(DATASET_REPO_ID)
    print(dataset)

    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(str(output_dir))
    print("Done.")


if __name__ == "__main__":
    main()