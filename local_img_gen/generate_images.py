import argparse
import json
import random
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline


def read_arguments():
    parser = argparse.ArgumentParser(description="Generate one image per MS COCO caption with a text-to-image model.")
    parser.add_argument("--captions-file", required=True, help="COCO captions file, e.g. annotations/captions_train2017.json")
    parser.add_argument("--output-dir", required=True, help="Folder for the images and metadata.jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-2512", help="Any diffusers text-to-image model")
    parser.add_argument("--generator-name", help="Name stored with every image (default: last part of --model-id)")
    parser.add_argument("--prompt-template", default="{caption}", help='e.g. "A photorealistic photograph of: {caption}"')
    parser.add_argument(
        "--model-settings",
        default='{"true_cfg_scale": 4.0, "negative_prompt": " "}',
        help="Extra model settings as JSON. The default is the recommended setting for Qwen-Image.",
    )
    parser.add_argument("--count", type=int, help="How many COCO images to use (default: all)")
    parser.add_argument("--order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--limit", type=int, help="Stop after this many new images")
    parser.add_argument("--width", type=int, default=1104)
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=271)
    parser.add_argument("--cpu-offload", action="store_true", help="Use if the GPU runs out of memory")
    return parser.parse_args()


def load_one_caption_per_image(captions_file):
    with open(captions_file, encoding="utf-8") as file:
        coco = json.load(file)

    caption_by_image_id = {}
    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in caption_by_image_id:
            caption_by_image_id[image_id] = annotation["caption"].strip()
    return caption_by_image_id


def choose_image_ids(caption_by_image_id, count, seed, order):
    image_ids = sorted(caption_by_image_id.keys())
    random.Random(seed).shuffle(image_ids)
    if count is not None:
        image_ids = image_ids[:count]
    if order == "reverse":
        image_ids.reverse()
    return image_ids


def read_finished_file_names(metadata_path):
    finished_file_names = set()
    if not metadata_path.exists():
        return finished_file_names

    with open(metadata_path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                finished_file_names.add(json.loads(line)["file_name"])
    return finished_file_names


def load_model(model_id, use_cpu_offload):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cpu":
        number_type = torch.float32
    else:
        number_type = torch.bfloat16

    model = DiffusionPipeline.from_pretrained(model_id, dtype=number_type)
    if use_cpu_offload:
        model.enable_model_cpu_offload()
    else:
        model.to(device)
    return model


def generate_image(model, prompt, seed, arguments, model_settings):
    random_generator = torch.Generator("cpu").manual_seed(seed)
    with torch.inference_mode():
        result = model(
            prompt=prompt,
            width=arguments.width,
            height=arguments.height,
            num_inference_steps=arguments.steps,
            generator=random_generator,
            **model_settings,
        )
    return result.images[0]


def save_image_and_metadata(image, file_name, metadata, output_dir, jpeg_quality):
    image.convert("RGB").save(output_dir / file_name, format="JPEG", quality=jpeg_quality)
    with open(output_dir / "metadata.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(metadata) + "\n")


def main():
    arguments = read_arguments()
    if "{caption}" not in arguments.prompt_template:
        raise SystemExit("--prompt-template must contain {caption}")

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_name = arguments.generator_name
    if generator_name is None:
        generator_name = arguments.model_id.split("/")[-1]

    caption_by_image_id = load_one_caption_per_image(arguments.captions_file)
    image_ids = choose_image_ids(caption_by_image_id, arguments.count, arguments.seed, arguments.order)
    finished_file_names = read_finished_file_names(output_dir / "metadata.jsonl")

    image_ids_to_generate = []
    for image_id in image_ids:
        file_name = f"{generator_name}_{image_id}.jpg"
        if file_name not in finished_file_names:
            image_ids_to_generate.append(image_id)

    print(f"{len(image_ids) - len(image_ids_to_generate)} of {len(image_ids)} images already done")
    if arguments.limit is not None:
        image_ids_to_generate = image_ids_to_generate[: arguments.limit]
    print(f"Generating {len(image_ids_to_generate)} images now")

    model = load_model(arguments.model_id, arguments.cpu_offload)
    model_settings = json.loads(arguments.model_settings)
    start_time = time.time()

    for number, image_id in enumerate(image_ids_to_generate, start=1):
        caption = caption_by_image_id[image_id]
        prompt = arguments.prompt_template.replace("{caption}", caption)
        seed = arguments.seed + image_id
        file_name = f"{generator_name}_{image_id}.jpg"

        image = generate_image(model, prompt, seed, arguments, model_settings)
        metadata = {
            "file_name": file_name,
            "generator": generator_name,
            "prompt": prompt,
            "coco_image_id": image_id,
            "seed": seed,
        }
        save_image_and_metadata(image, file_name, metadata, output_dir, arguments.jpeg_quality)

        seconds_per_image = (time.time() - start_time) / number
        hours_left = seconds_per_image * (len(image_ids_to_generate) - number) / 3600
        print(f"[{number}/{len(image_ids_to_generate)}] {file_name}  {seconds_per_image:.1f}s per image, {hours_left:.1f}h left")


if __name__ == "__main__":
    main()
