import hashlib
import json
import random

from PIL import Image, ImageEnhance, ImageFilter

SETTINGS = {
    "scale_probability": 0.5,
    "scale_factor_min": 0.8,
    "scale_factor_max": 1.2,

    "contrast_probability": 0.3,
    "contrast_min": 0.8,
    "contrast_max": 1.2,

    "blur_probability": 0.3,
    "blur_radius_min": 0.1,
    "blur_radius_max": 2.0,
}


def augment_image(image, seed):
    choices = choose_augmentations(seed)
    return apply_augmentations(image, choices)


def choose_augmentations(seed):
    choices = {}

    scale_dice = make_dice(seed, "scale")
    if happens(scale_dice, SETTINGS["scale_probability"]):
        choices["scale_factor"] = random_number(
            scale_dice, SETTINGS["scale_factor_min"], SETTINGS["scale_factor_max"]
        )

    contrast_dice = make_dice(seed, "contrast")
    if happens(contrast_dice, SETTINGS["contrast_probability"]):
        choices["contrast"] = random_number(
            contrast_dice, SETTINGS["contrast_min"], SETTINGS["contrast_max"]
        )

    blur_dice = make_dice(seed, "blur")
    if happens(blur_dice, SETTINGS["blur_probability"]):
        choices["blur_radius"] = random_number(
            blur_dice, SETTINGS["blur_radius_min"], SETTINGS["blur_radius_max"]
        )

    return choices


def apply_augmentations(image, choices):
    image = image.convert("RGB")

    if "scale_factor" in choices:
        image = scale(image, choices["scale_factor"])

    if "contrast" in choices:
        image = ImageEnhance.Contrast(image).enhance(choices["contrast"])

    if "blur_radius" in choices:
        image = image.filter(ImageFilter.GaussianBlur(choices["blur_radius"]))

    return image


def scale(image, factor):
    new_width = round(image.width * factor)
    new_height = round(image.height * factor)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def make_dice(seed, step_name):
    return random.Random(f"{seed}:{step_name}")


def happens(dice, probability):
    return dice.random() < probability


def random_number(dice, lowest, highest):
    return lowest + (highest - lowest) * dice.random()


def view_seed(image_name, view_number):
    text = f"{image_name}:{view_number}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)


def describe_choices(choices):
    parts = []
    if "scale_factor" in choices:
        parts.append(f"scale x{choices['scale_factor']:.2f}")
    if "contrast" in choices:
        parts.append(f"contrast {choices['contrast']:.2f}")
    if "blur_radius" in choices:
        parts.append(f"blur {choices['blur_radius']:.2f}")
    if not parts:
        return "unchanged"
    return ", ".join(parts)


def settings_fingerprint():
    settings_as_text = json.dumps(SETTINGS, sort_keys=True)
    return hashlib.sha256(settings_as_text.encode()).hexdigest()[:12]