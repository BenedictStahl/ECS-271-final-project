import argparse
import base64
import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

DESCRIBE_PROMPT = (
    "Describe this photograph in concrete visual detail so an image generator can "
    'recreate a realistic photo of the same scene. Include natural photographic '
    'phrasing such as "This is a picture taken of" or "A photograph of", and cover '
    "the main subjects, setting, lighting, framing, colors, and background. "
    "Output only the description."
)

VERIFY_PROMPT = (
    "Compare the original short caption and the expanded photograph description.\n"
    "Decide if the expanded description preserves the core meaning, main subjects, "
    "and scene of the original caption (extra visual detail is expected, but major "
    "contradictions, missing key subjects, or a different scene should fail).\n\n"
    "Original caption: {original}\n"
    "Expanded description: {expanded}\n\n"
    'Respond with JSON: {{"match": true or false, "reason": "brief explanation"}}.'
)

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["match", "reason"],
}

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
        help="Output CSV path (defaults to <input-dir>/real_descriptions.csv)",
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
    image_path: Path | None = None,
    response_format: dict | None = None,
    use_think_flag: list[bool] | None = None,
) -> str:
    message = {"role": "user", "content": prompt}
    if image_path is not None:
        message["images"] = [base64.b64encode(image_path.read_bytes()).decode("ascii")]

    payload = {"model": model, "stream": False, "messages": [message]}
    if response_format is not None:
        payload["format"] = response_format
    if use_think_flag is None or use_think_flag[0]:
        payload["think"] = False

    endpoint = f"{ollama_url.rstrip('/')}/api/chat"
    for _ in range(2):
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data["message"]["content"]
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except urllib.error.HTTPError as err:
            if err.code == 400 and "think" in payload:
                payload.pop("think")
                if use_think_flag is not None:
                    use_think_flag[0] = False
                continue
            raise
    raise RuntimeError("Ollama request failed")


def verify_description(
    ollama_url: str,
    model: str,
    original: str,
    expanded: str,
    use_think_flag: list[bool],
) -> tuple[bool, str]:
    prompt = VERIFY_PROMPT.format(original=original, expanded=expanded)
    raw = ollama_chat(
        ollama_url,
        model,
        prompt,
        response_format=VERIFY_SCHEMA,
        use_think_flag=use_think_flag,
    )
    try:
        parsed = json.loads(raw)
        return bool(parsed.get("match", False)), str(parsed.get("reason", "")).strip()
    except json.JSONDecodeError:
        return False, f"invalid verifier json: {raw[:120]}"


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
    output_path = arguments.output or (arguments.input_dir / "real_descriptions.csv")

    with metadata_path.open(encoding="utf-8", newline="") as f:
        real_rows = [row for row in csv.DictReader(f) if is_real_row(row)]

    done_paths = load_done_paths(output_path)
    pending = [row for row in real_rows if row["path"] not in done_paths]
    if arguments.limit is not None:
        pending = pending[: arguments.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    use_think_flag = [True]
    matched = 0

    with output_path.open("a", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=OUTPUT_FIELDS)
        if write_header:
            writer.writeheader()

        progress = tqdm(pending, desc=f"Describing ({arguments.model})")
        for row in progress:
            image_path = arguments.input_dir / row["path"]
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
                    image_path=image_path,
                    use_think_flag=use_think_flag,
                )
                is_match, reason = verify_description(
                    arguments.ollama_url,
                    arguments.model,
                    original_caption,
                    description,
                    use_think_flag,
                )
                if is_match:
                    break

            if is_match:
                matched += 1

            writer.writerow(
                {
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
            )
            out_file.flush()
            progress.set_postfix(matched=f"{matched}/{progress.n + 1}")

    print(
        f"Processed {len(pending)} real images ({matched} matched) -> {output_path}"
    )


if __name__ == "__main__":
    main()
