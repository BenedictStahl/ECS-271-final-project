import argparse
import os
from pathlib import Path

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

    dataset = load_dataset(DATASET_REPO_ID)
    print(dataset)


if __name__ == "__main__":
    main()
