import argparse
import base64
import csv
import json
import re
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_MODEL = "qwen3-vl:4b-instruct"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_NUM_CTX = 2048

DESCRIBE_PROMPT = (
    "Describe this photograph in 2-3 concise, concrete sentences so an image "
    "generator can recreate a realistic photo of the same scene. Start with "
    'natural photographic phrasing such as "This is a picture taken of" or '
    '"A photograph of", and cover the main subjects, setting, lighting, '
    "framing, colors, and background. Output only the description."
)

VERIFY_PROMPT = (
    "Does the expanded description preserve the core meaning, main subjects, "
    "and scene of the original caption? Extra visual detail is expected; fail "
    "only for major contradictions, missing key subjects, or a different scene.\n\n"
    "Original caption: {original}\n"
    "Expanded description: {expanded}\n\n"
    "Reply with ONLY 'YES' if it matches, or 'NO: <brief reason>' if not."
)

OUTPUT_FIELDS = [
    "path",
    "split",
    "index",
    "source",
    "original_caption",
    "expanded_caption",
    "match",
    "attempts",
    "reason",
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing decoded images and metadata.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.csv (defaults to <input-dir>/metadata.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to <input-dir>/detailed_descriptions.csv)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Ollama vision model name",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama server base URL",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries when the similarity check fails",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent Ollama requests (pair with OLLAMA_NUM_PARALLEL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of real images to process",
    )
    return parser.parse_args()


def ollama_chat(
    ollama_url: str,
    model: str,
    prompt: str,
    image_b64: str | None = None,
    num_predict: int = 160,
    temperature: float | None = None,
) -> str:
    message = {"role": "user", "content": prompt}
    if image_b64 is not None:
        message["images"] = [image_b64]

    options = {"num_ctx": DEFAULT_NUM_CTX, "num_predict": num_predict}
    if temperature is not None:
        options["temperature"] = temperature

    payload = {
        "model": model,
        "stream": False,
        "messages": [message],
        "options": options,
    }
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def verify_description(
    ollama_url: str,
    model: str,
    original: str,
    expanded: str,
) -> tuple[bool, str]:
    prompt = VERIFY_PROMPT.format(original=original, expanded=expanded)
    raw = ollama_chat(
        ollama_url,
        model,
        prompt,
        num_predict=40,
        temperature=0.0,
    )
    upper = raw.upper()
    if upper.startswith("YES"):
        return True, ""
    if upper.startswith("NO"):
        return False, raw[2:].lstrip(": -").strip()
    return False, raw[:120]


def process_row(row: dict, arguments) -> dict:
    image_path = arguments.input_dir / row["path"]
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    original_caption = (row.get("caption") or "").strip()

    description = ""
    is_match = False
    reason = ""
    attempts = 0

    for attempt in range(arguments.max_retries + 1):
        attempts = attempt + 1
        prompt = DESCRIBE_PROMPT
        if attempt > 0 and reason:
            prompt += (
                f"\n\nReview feedback on previous draft: {reason}. "
                "Inspect the image carefully and accurately describe all key subjects."
            )

        description = ollama_chat(
            arguments.ollama_url,
            arguments.model,
            prompt,
            image_b64=image_b64,
            num_predict=160,
        )
        is_match, reason = verify_description(
            arguments.ollama_url,
            arguments.model,
            original_caption,
            description,
        )
        if is_match:
            break

    return {
        "path": row["path"],
        "split": row.get("split", ""),
        "index": row.get("index", ""),
        "source": row.get("source", "real"),
        "original_caption": original_caption,
        "expanded_caption": description,
        "match": str(is_match).lower(),
        "attempts": attempts,
        "reason": reason,
    }


def load_done_paths(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open(encoding="utf-8", newline="") as f:
        return {row["path"] for row in csv.DictReader(f) if row.get("path")}


def is_real_row(row: dict) -> bool:
    return row.get("source") == "real" or str(row.get("label_a")) == "0"


def main():
    arguments = parse_arguments()
    metadata_path = arguments.metadata or (arguments.input_dir / "metadata.csv")
    output_path = arguments.output or (arguments.input_dir / "detailed_descriptions.csv")

    with metadata_path.open(encoding="utf-8", newline="") as f:
        real_rows = [row for row in csv.DictReader(f) if is_real_row(row)]

    done_paths = load_done_paths(output_path)
    pending = [row for row in real_rows if row["path"] not in done_paths]
    if arguments.limit is not None:
        pending = pending[: arguments.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    matched = 0
    processed = 0

    with output_path.open("a", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        progress = tqdm(total=len(pending), desc=f"Describing ({arguments.model})")
        workers = max(1, arguments.workers)
        max_in_flight = workers * 2
        row_iter = iter(pending)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            in_flight = set()
            for _ in range(min(max_in_flight, len(pending))):
                in_flight.add(executor.submit(process_row, next(row_iter), arguments))

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result["match"] == "true":
                        matched += 1
                    writer.writerow(result)
                    out_file.flush()
                    processed += 1
                    progress.update(1)
                    progress.set_postfix(matched=f"{matched}/{processed}")

                    next_row = next(row_iter, None)
                    if next_row is not None:
                        in_flight.add(executor.submit(process_row, next_row, arguments))
        progress.close()

    print(
        f"Processed {processed} real images ({matched} matched) -> {output_path}"
    )


if __name__ == "__main__":
    main()
