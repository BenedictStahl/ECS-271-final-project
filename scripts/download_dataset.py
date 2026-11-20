import argparse
import json
from pathlib import Path

from dataset_io import DATASET_REPO_ID, DATASET_REVISION, DEFAULT_DATA_DIR, split_files


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Local Parquet destination (default: repository data/raw).",
    )
    parser.add_argument("--revision", default=DATASET_REVISION, help="Hub revision; defaults to the audited immutable commit.")
    parser.add_argument("--local-files-only", action="store_true", help="Reuse local files/Hub cache without network access.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    from huggingface_hub import HfApi, snapshot_download

    destination = arguments.data_dir.expanduser().resolve()
    # Resolve a symbolic revision once so the manifest identifies the actual data.
    revision = arguments.revision
    if not arguments.local_files_only:
        revision = HfApi().dataset_info(DATASET_REPO_ID, revision=revision).sha
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        allow_patterns=["README.md", "data/*.parquet"],
        local_files_only=arguments.local_files_only,
    )
    files = split_files(destination)
    manifest = {
        "repo_id": DATASET_REPO_ID,
        "revision": revision,
        "files": {name: [str(p.relative_to(destination)) for p in paths] for name, paths in files.items()},
        "parquet_bytes": sum(p.stat().st_size for paths in files.values() for p in paths),
    }
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Downloaded Parquet files to {destination}. Run scripts/profile_dataset.py next.")


if __name__ == "__main__":
    main()
