"""Shared deterministic training views for feature caching and later fine-tuning."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import augment_image as ben

PIPELINE = "ben_after_crop_restore224_v1"
BEN_COMMIT = "4ebd24098f3f8f76fb63483c7cb26bafd316aef6"


def augmentation_config():
    config = {"pipeline": PIPELINE, "ben_source_commit": BEN_COMMIT,
              "module_sha256": hashlib.sha256(Path(ben.__file__).read_bytes()).hexdigest(),
              "settings": dict(ben.SETTINGS),
              "order": "EXIF/RGB -> shared crop224 -> Ben augment -> restore224 bicubic -> encoder normalization"}
    config["recipe_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config


def augmented_id(parent_id, view_number):
    return f"{parent_id}::aug-v{view_number}"


def augment_crop(crop, parent_id, view_number, split):
    """Use this SAME function for fixed cached views or epoch-varying fine-tuning views.

    view_number starts at 1. Callers leave validation/test and original views alone.
    No dataset/model selection occurs here. Frozen features may be precomputed for
    every supplied train source; each fit selects ONLY views of its allowed parents.
    """
    if split != "train":
        raise ValueError("Augmentation is training-only")
    if not parent_id or type(view_number) is not int or view_number < 1:
        raise ValueError("A parent image ID and positive view number are required")
    if crop.size != (224, 224) or crop.mode != "RGB":
        raise ValueError("Apply the common RGB 224x224 crop before augmentation")
    seed = ben.view_seed(parent_id, view_number)
    choices = ben.choose_augmentations(seed)
    image = ben.augment_image(crop, seed)
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
    details = {"id": augmented_id(parent_id, view_number), "parent_id": parent_id,
               "augmentation_view": view_number, "augmentation_seed": seed,
               "augmentation_choices": choices,
               "view_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
    return image, details


def checked_view_row(row, parent):
    """Bind cache view identity to its unchanged original image, without trusting view labels."""
    number = row.get("augmentation_view")
    if type(number) is not int or number < 1 or parent["split"] != "train" or row["split"] != "train":
        raise ValueError("Invalid training augmentation view")
    if row["id"] != augmented_id(parent["id"], number) or row.get("augmentation_seed") != ben.view_seed(parent["id"], number):
        raise ValueError("Augmentation ID/seed differs from parent view")
    keys = ("sha256", "split", "source", "target", "reference_id")
    if any(row.get(k) != parent.get(k) for k in keys):
        raise ValueError("Augmentation changed original identity, label, or reference")
    details = ("id", "parent_id", "augmentation_view", "augmentation_seed", "augmentation_choices", "view_pixel_sha256")
    if any(k not in row for k in details):
        raise ValueError("Incomplete augmentation metadata")
    return dict(parent, **{k: row[k] for k in details})
