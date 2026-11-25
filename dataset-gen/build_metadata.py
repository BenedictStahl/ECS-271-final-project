import argparse
import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_GENERATED_DIR = REPO_ROOT / "data" / "generated"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "metadata_completed.csv"

FIELDS = ["path", "split", "index", "label_a", "label_b", "source", "caption"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Generated models to include (defaults to all models with 100%% of real images on disk)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing real/ splits and real_descriptions.csv",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
        help="Directory containing generated model splits",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output metadata CSV path (paths written relative to its parent directory)",
    )
    return parser.parse_args()


def detect_completed_models(generated_dir: Path, expected_count: int) -> list[str]:
    counts = Counter()
    for split_dir in sorted(generated_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for model_dir in sorted(split_dir.iterdir()):
            if model_dir.is_dir():
                counts[model_dir.name] += sum(1 for _ in model_dir.glob("*.jpg"))
    return sorted(m for m, count in counts.items() if count >= expected_count)


def main():
    arguments = parse_arguments()
    descriptions_path = arguments.raw_dir / "real_descriptions.csv"
    with descriptions_path.open(encoding="utf-8", newline="") as f:
        real_rows = list(csv.DictReader(f))

    models = arguments.models or detect_completed_models(
        arguments.generated_dir, len(real_rows)
    )
    if not models:
        raise RuntimeError("No completed generated models found.")

    base_dir = arguments.output.resolve().parent
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    label_map = {"real": 0, **{m: i + 1 for i, m in enumerate(models)}}

    counts = Counter()
    missing = 0

    with arguments.output.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FIELDS)
        writer.writeheader()

        for row in real_rows:
            split = row["split"]
            index = row["index"]
            filename = Path(row["path"]).name

            real_file = (arguments.raw_dir / row["path"]).resolve()
            if real_file.exists():
                writer.writerow(
                    {
                        "path": real_file.relative_to(base_dir).as_posix(),
                        "split": split,
                        "index": index,
                        "label_a": 0,
                        "label_b": 0,
                        "source": "real",
                        "caption": (row.get("original_caption") or "").strip(),
                    }
                )
                counts[("real", split)] += 1
            else:
                missing += 1

            prompt = (
                row.get("expanded_caption") or row.get("original_caption") or ""
            ).strip()
            jpg_name = Path(filename).with_suffix(".jpg").name
            for model in models:
                gen_file = (arguments.generated_dir / split / model / jpg_name).resolve()
                if gen_file.exists():
                    writer.writerow(
                        {
                            "path": gen_file.relative_to(base_dir).as_posix(),
                            "split": split,
                            "index": index,
                            "label_a": 1,
                            "label_b": label_map[model],
                            "source": model,
                            "caption": prompt,
                        }
                    )
                    counts[(model, split)] += 1
                else:
                    missing += 1

    print(f"Wrote {sum(counts.values())} rows (missing: {missing}) -> {arguments.output}")
    print(f"Label map (label_b): {label_map}")
    for source in ["real", *models]:
        per_split = ", ".join(
            f"{s}={counts[(source, s)]}" for s in ("train", "validation", "test")
        )
        print(f"  {source}: {per_split}")


if __name__ == "__main__":
    main()
