import argparse
import csv
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESCRIPTIONS = REPO_ROOT / "data" / "raw" / "real_descriptions.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "generated"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"

MODEL_LABELS = {
    "flux3": 1,
    "z_image_turbo": 2,
    "sdxl_turbo": 3,
    "minimax_h3": 4,
    "qwen_image_2_1": 5,
}

DEFAULT_SIZES = {
    "flux3": 768,
    "z_image_turbo": 768,
    "sdxl_turbo": 512,
    "minimax_h3": 1024,
    "qwen_image_2_1": 768,
}

METADATA_FIELDS = ["path", "split", "index", "label_a", "label_b", "source", "caption"]


def build_workflow(model: str, prompt: str, seed: int, size: int) -> dict:
    if model == "sdxl_turbo":
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_turbo_1.0_fp16.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": prompt},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": ""},
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": size, "height": size, "batch_size": 1},
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": 4,
                    "cfg": 1.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            },
            "7": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
        }

    if model == "z_image_turbo":
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "z_image_turbo_int8_convrot.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {"model": ["1", 0], "shift": 3.0},
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen_3_4b_fp8_mixed.safetensors",
                    "type": "lumina2",
                },
            },
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["3", 0], "text": prompt},
            },
            "6": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["5", 0]},
            },
            "7": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": size, "height": size, "batch_size": 1},
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["7", 0],
                    "seed": seed,
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler_name": "res_multistep",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["4", 0]},
            },
            "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
        }

    if model == "flux3":
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "flux3-image-klein-int8-convrot.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_4b_int8_convrot.safetensors",
                    "type": "flux3",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "flux2-vae.safetensors"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": prompt},
            },
            "5": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["4", 0]},
            },
            "6": {
                "class_type": "EmptyFlux2LatentImage",
                "inputs": {"width": size, "height": size, "batch_size": 1},
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["6", 0],
                    "seed": seed,
                    "steps": 4,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
            },
            "9": {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}},
        }

    if model == "qwen_image_2_1":
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "qwen_image_2.1_turbo_int8_convrot.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_8b_w4a8.safetensors",
                    "type": "qwen_image",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
            },
            "4": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": prompt,
                    "negative_prompt": "",
                    "resolution": size,
                },
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["4", 1],
                    "latent_image": ["4", 2],
                    "seed": seed,
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["5", 0], "vae": ["3", 0]},
            },
            "7": {"class_type": "PreviewImage", "inputs": {"images": ["6", 0]}},
        }

    if model == "minimax_h3":
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "minimax_h3_image_pruned_w4a8.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                    "type": "minimax",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "minimax_h3_image_vae_fp16.safetensors"},
            },
            "4": {
                "class_type": "MiniMaxH3ImageConditioning",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": prompt,
                    "width": size,
                    "height": size,
                },
            },
            "5": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["4", 0]},
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["4", 1],
                    "seed": seed,
                    "steps": 6,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["3", 0]},
            },
            "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}},
        }

    raise ValueError(f"Unknown model: {model}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_LABELS.keys()),
        default=list(MODEL_LABELS.keys()),
        help="Models to generate images with (runs one model at a time)",
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=DEFAULT_DESCRIPTIONS,
        help="CSV of real image descriptions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root (writes <output-dir>/<split>/<model>/<filename>.jpg)",
    )
    parser.add_argument(
        "--comfy-url",
        default=DEFAULT_COMFY_URL,
        help="ComfyUI server base URL",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Override square generation size for all selected models",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="JPEG quality (1-95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (combined with row index)",
    )
    parser.add_argument(
        "--only-matched",
        action="store_true",
        help="Only generate for rows where match == true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max images to generate per model",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Shard index (0-based) for splitting across multiple machines",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards across machines",
    )
    return parser.parse_args()


def comfy_free_memory(comfy_url: str):
    req = urllib.request.Request(
        f"{comfy_url.rstrip('/')}/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass


def comfy_generate_image(comfy_url: str, workflow: dict, timeout: float = 600.0) -> bytes:
    base = comfy_url.rstrip("/")
    req = urllib.request.Request(
        f"{base}/prompt",
        data=json.dumps({"prompt": workflow}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            prompt_id = json.loads(resp.read().decode("utf-8"))["prompt_id"]
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI /prompt HTTP {err.code}: {body}") from err

    deadline = time.monotonic() + timeout
    history_url = f"{base}/history/{prompt_id}"
    while time.monotonic() < deadline:
        with urllib.request.urlopen(history_url, timeout=30) as resp:
            history = json.loads(resp.read().decode("utf-8"))
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {status.get('messages')}")
            for node_out in entry.get("outputs", {}).values():
                images = node_out.get("images")
                if images:
                    img_info = images[0]
                    query = urllib.parse.urlencode(
                        {
                            "filename": img_info["filename"],
                            "subfolder": img_info.get("subfolder", ""),
                            "type": img_info.get("type", "temp"),
                        }
                    )
                    with urllib.request.urlopen(f"{base}/view?{query}", timeout=60) as img_resp:
                        return img_resp.read()
        time.sleep(0.1)

    raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")


def save_as_jpeg(raw_bytes: bytes, target_path: Path, quality: int):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img.convert("RGB").save(target_path, format="JPEG", quality=quality)


def main():
    arguments = parse_arguments()

    with arguments.descriptions.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if arguments.only_matched:
        rows = [r for r in rows if str(r.get("match", "")).lower() == "true"]

    rows = [
        r for idx, r in enumerate(rows) if idx % arguments.num_shards == arguments.shard
    ]

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = arguments.output_dir / "metadata.csv"
    write_header = not metadata_path.exists() or metadata_path.stat().st_size == 0

    with metadata_path.open("a", encoding="utf-8", newline="") as meta_file:
        writer = csv.DictWriter(meta_file, fieldnames=METADATA_FIELDS)
        if write_header:
            writer.writeheader()

        for model in arguments.models:
            size = arguments.size or DEFAULT_SIZES[model]
            pending = []
            for row in rows:
                split = row["split"]
                filename = Path(row["path"]).with_suffix(".jpg").name
                rel_path = Path(split) / model / filename
                if not (arguments.output_dir / rel_path).exists():
                    pending.append((row, rel_path))

            if arguments.limit is not None:
                pending = pending[: arguments.limit]

            if not pending:
                print(f"Skipping {model}: all images already generated.")
                continue

            comfy_free_memory(arguments.comfy_url)
            progress = tqdm(pending, desc=f"Generating ({model})")
            for row, rel_path in progress:
                prompt = (
                    row.get("expanded_caption") or row.get("original_caption") or ""
                ).strip()
                row_index = int(row.get("index") or 0)
                seed = arguments.seed + row_index
                workflow = build_workflow(model, prompt, seed, size)

                raw_bytes = comfy_generate_image(arguments.comfy_url, workflow)
                save_as_jpeg(
                    raw_bytes,
                    arguments.output_dir / rel_path,
                    arguments.jpeg_quality,
                )

                writer.writerow(
                    {
                        "path": rel_path.as_posix(),
                        "split": row["split"],
                        "index": row.get("index", ""),
                        "label_a": 1,
                        "label_b": MODEL_LABELS[model],
                        "source": model,
                        "caption": prompt,
                    }
                )
                meta_file.flush()

            print(f"Finished {model}: {len(pending)} images generated.")


if __name__ == "__main__":
    main()
