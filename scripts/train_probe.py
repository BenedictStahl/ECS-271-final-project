"""Frozen image encoders: check real weights or cache features before fitting a head.

No split selection or classifier training is performed yet. Cache published splits
as supplied; a later, reviewed manifest must select leakage-safe training rows.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset_io import DEFAULT_DATA_DIR, REPO_ROOT, SOURCE_NAMES, split_files

POOLING = "mean_patch_v1"


CHECKPOINTS = {
    "deit3": ("timm/deit3_base_patch16_224.fb_in22k_ft_in1k",
              "b5791d4983082b0c680da556954beb5b1962e85f"),
    "siglip2": ("google/siglip2-base-patch16-224",
                "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
    "dinov3": ("facebook/dinov3-vitb16-pretrain-lvd1689m",
               "5931719e67bbdb9737e363e781fb0c67687896bc"),
}


def load_encoder(name, device, offline=False, local_checkpoint=None):
    from huggingface_hub import snapshot_download

    repo, revision = CHECKPOINTS[name]
    # An explicit, hash-checked local DINO delivery requires no Hub connection.
    if local_checkpoint is not None:
        if name != "dinov3":
            raise ValueError("Explicit local checkpoint currently supports DINOv3 only")
        folder = Path(local_checkpoint).resolve()
        manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
        if (manifest["repository"], manifest["revision"]) != (repo, revision):
            raise ValueError("Local checkpoint repository/revision differs from selected model")
        entries = {f["filename"]: f for f in manifest["files"]}
        for filename in ("config.json", "model.safetensors", "LICENSE.md"):
            entry = entries[filename]
            path = folder / filename
            with path.open("rb") as handle:
                checksum = hashlib.file_digest(handle, "sha256").hexdigest()
            if path.stat().st_size != entry["bytes"] or checksum != entry["sha256"]:
                raise ValueError(f"Local checkpoint checksum mismatch: {filename}")
    else:
        folder = snapshot_download(
            repo, revision=revision, local_dir=REPO_ROOT / "data" / "models" / name,
            allow_patterns=["config.json", "model.safetensors", "model.safetensors.index.json",
                            "model-*.safetensors"],
            max_workers=1, local_files_only=offline,
        )
    if name == "deit3":
        import timm
        model = timm.create_model(f"local-dir:{folder}", pretrained=True, num_classes=0)
    else:
        from transformers import DINOv3ViTModel, SiglipVisionModel
        cls = DINOv3ViTModel if name == "dinov3" else SiglipVisionModel
        model, info = cls.from_pretrained(folder, local_files_only=True, output_loading_info=True)
        if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"Incomplete checkpoint load: {info}")
        # The SigLIP repository also contains a text tower and contrastive scalars.
        unexpected = [k for k in info["unexpected_keys"]
                      if name != "siglip2" or not k.startswith(("text_model.", "logit_scale", "logit_bias"))]
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        if name == "siglip2":
            model.use_head = False  # Return normalized tokens without attention pooling.
    return model.to(device).eval().requires_grad_(False)


def image_crop(image):
    """The exact RGB crop explained by the 14x14 score grid, before normalization."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True)(image)
    return transforms.CenterCrop(224)(image)


def image_normalization(name):
    # Common geometry across encoders, native normalization per checkpoint.
    # This deliberately uses bicubic for all three (SigLIP's native is bilinear).
    mean, std = (([0.5] * 3, [0.5] * 3) if name == "siglip2" else
                 ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Normalize(mean, std)


def image_transform(name):
    return transforms.Compose([
        transforms.Lambda(image_crop), transforms.ToTensor(), image_normalization(name),
    ])


def training_view(image, name, row, view_number):
    """Reusable for cache extraction and fine-tuning; no augmentation in encoder code."""
    from training_augmentation import augment_crop
    crop, details = augment_crop(image_crop(image), row["id"], view_number, row["split"])
    # Geometry/augmentation is shared; checkpoint-specific normalization comes last.
    tensor = image_normalization(name)(transforms.ToTensor()(crop))
    return tensor, dict(row, **details)


def patch_tokens(model, name, images):
    """Final per-token LayerNorm, excluding non-image tokens; no L2 normalization."""
    if name == "deit3":
        tokens = model.forward_features(images)[:, model.num_prefix_tokens:]
    elif name == "dinov3":
        tokens = model(pixel_values=images).last_hidden_state[:, 1 + model.config.num_register_tokens:]
    elif name == "siglip2":
        tokens = model(pixel_values=images).last_hidden_state
    else:
        raise ValueError(name)
    if tokens.shape[1:] != (196, 768) or not torch.isfinite(tokens).all():
        raise RuntimeError(f"Invalid {name} patch tokens: {tokens.shape}")
    return tokens


def patch_scores(tokens, weight, bias):
    """Signed fake-evidence scores on the cropped image's 14x14 grid.

    Bias is separate: mean(scores) + bias equals the image logit.
    Tokens already mix context across the image; these are not causal pixel maps.
    """
    tokens, weight = tokens.float(), weight.float().reshape(768)
    scores = (tokens @ weight).reshape(-1, 14, 14)
    logits = tokens.mean(dim=1) @ weight + bias
    torch.testing.assert_close(scores.mean(dim=(1, 2)) + bias, logits, rtol=1e-5, atol=1e-5)
    return scores, logits


def encode(model, name, images, device, verify_xai=False):
    with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda",
                                               dtype=torch.float16):
        images = images.to(device)
        tokens = patch_tokens(model, name, images).float()
        features = tokens.mean(dim=1)
    if verify_xai:
        # A deterministic diagnostic head, not a trained detector.
        weight = torch.linspace(-0.1, 0.1, 768, device=device)
        scores, logits = patch_scores(tokens, weight, 0.25)
        torch.testing.assert_close(features @ weight + 0.25, logits)
        print(f"{name}: {tuple(tokens.shape)} patches; 14x14 score/logit identity OK", flush=True)
    if features.shape != (len(images), 768) or not torch.isfinite(features).all():
        raise RuntimeError(f"Invalid {name} features: {features.shape}")
    return features.float().cpu().numpy()


def dataset_rows(data_dir):
    for split, paths in split_files(data_dir).items():
        for path in paths:
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    source = int(row["Label_B"])
                    target = int(row["Label_A"])
                    if source not in range(6) or target != int(source != 0):
                        raise ValueError(f"Unexpected labels in {path.name}:{offset}")
                    yield {"id": f"{path.name}:{offset}", "split": split,
                           "source": SOURCE_NAMES[source], "target": target,
                           "caption": row["Caption"], "bytes": row["Image"]["bytes"]}
                    offset += 1


def supplemental_rows(folder):
    """Expected layout: root/generator_name/image.jpg (nested folders allowed).

    All files are synthetic. This inventories them without assigning splits.
    Related generations must still be grouped before a final split is made.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths = [p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not paths:
        raise ValueError(f"No images in {folder}")
    for path in paths:
        relative = path.relative_to(folder)
        if len(relative.parts) < 2:
            raise ValueError(f"Place images in a generator-named subfolder: {path}")
        yield {"id": relative.as_posix(), "split": "unassigned",
               "source": relative.parts[0], "target": 1, "caption": None,
               "bytes": path.read_bytes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "cache"])
    parser.add_argument("--model", required=True, choices=CHECKPOINTS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--supplement", type=Path, help="Cache supplemental images instead of Defactify.")
    parser.add_argument("--replacement", type=Path, help="Prepared replacement index.jsonl; includes real images once.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Smoke-test row count; omit for full cache.")
    parser.add_argument("--output", type=Path, help="New output folder (must not already exist).")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--local-checkpoint", type=Path, help="Verified DINOv3 folder with weights, config, license and MANIFEST.json; no Hub access.")
    parser.add_argument("--reuse-cache", type=Path, help="Original-image cache from this encoder: verify compatibility and extract only missing IDs into the new output.")
    parser.add_argument("--augment-views", type=int, default=0,
                        help="Cache ONLY this many extra views per supplied training image; combine with the original cache when fitting.")
    args = parser.parse_args()
    if args.supplement and args.replacement:
        parser.error("Choose --replacement or --supplement, not both.")
    if args.augment_views < 0 or (args.augment_views and not args.replacement):
        parser.error("--augment-views requires --replacement and a nonnegative count")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive.")
    if args.action == "cache" and args.output is None:
        parser.error("cache requires --output (use a new folder per cache)")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new folder to avoid overwriting features.")
    device = torch.device(args.device)
    # Many CPU threads make the small per-image normalization operations slower.
    torch.set_num_threads(4)
    model = load_encoder(args.model, device, args.offline, args.local_checkpoint)
    print(f"Loaded {args.model}: {sum(p.numel() for p in model.parameters()):,} frozen parameters", flush=True)
    transform = image_transform(args.model)
    reused_ids = set()
    if args.reuse_cache:
        if not args.replacement or args.augment_views:
            parser.error("--reuse-cache requires unaugmented --replacement extraction")
        previous = json.loads((args.reuse_cache / "provenance.json").read_text(encoding="utf-8"))
        if (previous["model"] != list(CHECKPOINTS[args.model]) or previous.get("pooling") != POOLING
                or previous.get("geometry") != "exif_rgb_shortedge224_bicubic_antialias_centercrop224_v1"
                or previous["transform"] != str(transform) or previous.get("limit") is not None
                or previous.get("augmentation")):
            raise ValueError("Incompatible original feature cache")
        from fit_probe import canonical_cache_rows
        canonical = {r["id"]: r for r in map(json.loads, args.replacement.read_text(encoding="utf-8").splitlines())}
        old_rows = list(map(json.loads, (args.reuse_cache / "rows.jsonl").read_text(encoding="utf-8").splitlines()))
        canonical_cache_rows(old_rows, canonical)
        old_features = np.load(args.reuse_cache / "features.npy", mmap_mode="r")
        if old_features.shape != (len(old_rows), 768) or not np.isfinite(old_features).all():
            raise ValueError("Invalid reused feature array")
        reused_ids = {r["id"] for r in old_rows}
        if len(reused_ids) != len(old_rows):
            raise ValueError("Repeated IDs in reused cache")
        print(f"Reusing {len(reused_ids)} compatible original features; extracting missing images only", flush=True)
    if args.replacement:
        from replacement_dataset import indexed_rows
        rows = indexed_rows(args.replacement, split="train" if args.action == "check" or args.augment_views else None,
                            exclude_ids=reused_ids)
    else:
        rows = supplemental_rows(args.supplement) if args.supplement else dataset_rows(args.data_dir)
    limit = args.limit if args.action == "cache" else (args.limit or 4)
    images, metadata, chunks = [], [], []
    source_count = 0
    for row in rows:
        content = row.pop("bytes")
        row["sha256"] = hashlib.sha256(content).hexdigest()
        source_count += 1
        with Image.open(io.BytesIO(content)) as image:
            for view in range(1, args.augment_views + 1) if args.augment_views else (0,):
                if view:
                    tensor, record = training_view(image, args.model, row, view)
                else:
                    tensor, record = transform(image), row
                images.append(tensor)
                metadata.append(record)
                if len(images) == args.batch_size:
                    chunks.append(encode(model, args.model, torch.stack(images), device, args.action == "check"))
                    images.clear()
                    if len(metadata) % 1024 == 0:
                        print(f"Encoded {len(metadata)} views", flush=True)
        if limit and source_count >= limit:
            break
    if images:
        chunks.append(encode(model, args.model, torch.stack(images), device, args.action == "check"))
    if not chunks:
        raise RuntimeError("No images were encoded")
    features = np.concatenate(chunks)
    print(f"Real-image features: {features.shape}; all finite", flush=True)
    if args.action == "cache":
        args.output.mkdir(parents=True)
        np.save(args.output / "features.npy", features)
        with (args.output / "rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in metadata:
                handle.write(json.dumps(row) + "\n")
        provenance = {"model": CHECKPOINTS[args.model], "rows": len(metadata),
                      "pooling": POOLING,
                      "local_checkpoint": str(args.local_checkpoint.resolve()) if args.local_checkpoint else None,
                      "reused_cache": str(args.reuse_cache.resolve()) if args.reuse_cache else None,
                      "reused_rows": len(reused_ids),
                      "geometry": "exif_rgb_shortedge224_bicubic_antialias_centercrop224_v1",
                      "replacement_index": str(args.replacement.resolve()) if args.replacement else None,
                      "data_dir": str(args.data_dir.resolve()) if not args.replacement else None,
                      "supplement": str(args.supplement.resolve()) if args.supplement else None,
                      "limit": args.limit, "transform": str(transform),
                      "status": "features only; splits not approved; no classifier trained"}
        if args.augment_views:
            from training_augmentation import augmentation_config
            provenance["augmentation"] = dict(augmentation_config(), views_per_training_image=args.augment_views,
                                               source_images=source_count, cache_contents="additional_training_views_only")
        if args.replacement:
            provenance["replacement_index_sha256"] = hashlib.sha256(args.replacement.read_bytes()).hexdigest()
            provenance["dataset_summary"] = json.loads(args.replacement.with_name("dataset_summary.json").read_text(encoding="utf-8"))
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
