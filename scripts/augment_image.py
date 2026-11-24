import hashlib
import io
import random

from PIL import Image, ImageEnhance, ImageFilter

SETTINGS = {
    "scale_probability": 0.33,
    "scale_factor_min": 0.8,
    "scale_factor_max": 1.2,

    "contrast_probability": 0.33,
    "contrast_min": 0.8,
    "contrast_max": 1.2,

    "blur_probability": 0.33,
    "blur_radius_min": 0.1,
    "blur_radius_max": 2.0,

    "jpeg_probability": 0.33,
    "jpeg_quality_min": 60,
    "jpeg_quality_max": 95,
}

STEPS = ["scale", "contrast", "blur", "jpeg"]


def augment_image(image, seed):
    choices = choose_augmentations(seed)
    return apply_augmentations(image, choices)


def choose_augmentations(seed):
    chosen_steps = choose_steps(seed)
    choices = {}

    if "scale" in chosen_steps:
        choices["scale_factor"] = random_number(
            make_dice(seed, "scale_factor"), SETTINGS["scale_factor_min"], SETTINGS["scale_factor_max"]
        )

    if "contrast" in chosen_steps:
        choices["contrast"] = random_number(
            make_dice(seed, "contrast_factor"), SETTINGS["contrast_min"], SETTINGS["contrast_max"]
        )

    if "blur" in chosen_steps:
        choices["blur_radius"] = random_number(
            make_dice(seed, "blur_radius"), SETTINGS["blur_radius_min"], SETTINGS["blur_radius_max"]
        )

    if "jpeg" in chosen_steps:
        choices["jpeg_quality"] = random_whole_number(
            make_dice(seed, "jpeg_quality"), SETTINGS["jpeg_quality_min"], SETTINGS["jpeg_quality_max"]
        )

    return choices


def choose_steps(seed):
    chosen_steps = []
    for step in STEPS:
        if happens(make_dice(seed, step), SETTINGS[f"{step}_probability"]):
            chosen_steps.append(step)

    if not chosen_steps:
        fallback_step = random_pick(make_dice(seed, "fallback"), STEPS)
        chosen_steps.append(fallback_step)

    return chosen_steps


def apply_augmentations(image, choices):
    image = image.convert("RGB")

    if "scale_factor" in choices:
        image = scale(image, choices["scale_factor"])

    if "contrast" in choices:
        image = ImageEnhance.Contrast(image).enhance(choices["contrast"])

    if "blur_radius" in choices:
        image = image.filter(ImageFilter.GaussianBlur(choices["blur_radius"]))

    if "jpeg_quality" in choices:
        image = compress_as_jpeg(image, choices["jpeg_quality"])

    return image


def scale(image, factor):
    new_width = round(image.width * factor)
    new_height = round(image.height * factor)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def compress_as_jpeg(image, quality):
    file_in_memory = io.BytesIO()
    image.save(file_in_memory, format="JPEG", quality=quality)
    file_in_memory.seek(0)
    return Image.open(file_in_memory).convert("RGB")


def make_dice(seed, step_name):
    return random.Random(f"{seed}:{step_name}")


def happens(dice, probability):
    return dice.random() < probability


def random_number(dice, lowest, highest):
    return lowest + (highest - lowest) * dice.random()


def random_whole_number(dice, lowest, highest):
    number_of_options = highest - lowest + 1
    return lowest + int(number_of_options * dice.random())


def random_pick(dice, options):
    return options[int(len(options) * dice.random())]


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
    if "jpeg_quality" in choices:
        parts.append(f"jpeg q{choices['jpeg_quality']}")
    return ", ".join(parts)
