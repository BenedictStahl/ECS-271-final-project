"""Deterministic views, holdout isolation, and a real fitting-CLI integration fixture."""
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from training_augmentation import augment_crop, augmentation_config, checked_view_row
try:
    import numpy as np
    import torch
    from train_probe import training_view, image_transform, image_crop, image_normalization
    from fit_probe import canonical_cache_rows, include_training_views, validate_splits
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch", "torchvision", "pyarrow", "sklearn"}:
        raise
    torch = None


@unittest.skipIf(torch is None, "Training libraries required")
class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (240, 320, 3), dtype=np.uint8))
        self.row = dict(id="replacement:train/real/00000.jpg", split="train", source="real", target=0,
                        reference_id="train-reference", content_group_id="train-family", caption=None, sha256="parent-hash")

    def test_deterministic_views_without_input_or_global_random_mutation(self):
        before = self.image.tobytes()
        state = random.getstate()
        for name in ("deit3", "siglip2", "dinov3"):
            a, record_a = training_view(self.image, name, self.row, 1)
            b, record_b = training_view(self.image, name, self.row, 1)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(record_a, record_b)
            self.assertEqual(tuple(a.shape), (3, 224, 224))
            self.assertEqual(record_a["reference_id"], self.row["reference_id"])
        self.assertEqual(before, self.image.tobytes())
        self.assertEqual(state, random.getstate())
        self.assertEqual(augmentation_config(), augmentation_config())

    def test_same_augmented_pixels_across_encoder_normalizations(self):
        recovered = []
        for name in ("deit3", "siglip2", "dinov3"):
            tensor, record = training_view(self.image, name, self.row, 2)
            norm = image_normalization(name)
            pixels = tensor * torch.tensor(norm.std).reshape(3, 1, 1) + torch.tensor(norm.mean).reshape(3, 1, 1)
            recovered.append((pixels, record["view_pixel_sha256"]))
        for pixels, digest in recovered[1:]:
            torch.testing.assert_close(pixels, recovered[0][0], atol=1e-6, rtol=0)
            self.assertEqual(digest, recovered[0][1])

    def test_rejects_eval_augmentation_and_forged_view_identity(self):
        for split in ("validation", "test", "unassigned"):
            with self.assertRaisesRegex(ValueError, "training-only"):
                training_view(self.image, "deit3", dict(self.row, split=split), 1)
        with self.assertRaisesRegex(ValueError, "crop"):
            augment_crop(self.image, self.row["id"], 1, "train")
        _, view = training_view(self.image, "deit3", self.row, 1)
        changed = dict(view, source="wrong")
        with self.assertRaisesRegex(ValueError, "identity"):
            checked_view_row(changed, self.row)
        changed = dict(view, augmentation_seed=123)
        with self.assertRaisesRegex(ValueError, "seed"):
            checked_view_row(changed, self.row)

    def test_original_transform_remains_unchanged(self):
        from torchvision import transforms
        for name in ("deit3", "siglip2"):
            expected = image_normalization(name)(transforms.ToTensor()(image_crop(self.image)))
            torch.testing.assert_close(image_transform(name)(self.image), expected, atol=0, rtol=0)

    def fixture(self):
        rows, splits = [], {}
        for split in ("train", "validation", "test"):
            splits[split] = []
            for source in ("real", "seen", "heldout"):
                image_id = f"{split}/{source}"
                row = dict(id=image_id, split=split, source=source, target=int(source != "real"), caption=None,
                           sha256=hashlib.sha256(image_id.encode()).hexdigest(), reference_id=split + "-reference",
                           content_group_id=split + "-family")
                rows.append(row)
                if split == "test" or source != "heldout":
                    splits[split].append(image_id)
        manifest = dict(held_out_generator="heldout", splits=splits)
        crop = image_crop(self.image)
        views = []
        for row in rows:
            if row["split"] == "train":
                _, details = augment_crop(crop, row["id"], 1, "train")
                views.append(dict(row, **details))
        return rows, views, manifest

    def test_holdout_views_excluded_originals_kept_and_parent_weights_normalized(self):
        rows, views, manifest = self.fixture()
        expanded, weights, info = include_training_views(rows + views, manifest)
        validate_splits(rows + views, expanded, {r["id"] for r in rows + views if r["target"]})
        self.assertEqual(manifest["splits"]["train"], ["train/real", "train/seen"])
        self.assertEqual(info["augmented_training_rows"], 2)
        self.assertEqual(info["cached_views_not_selected"], 1)
        self.assertTrue(set(manifest["splits"]["train"]).issubset(expanded["splits"]["train"]))
        self.assertNotIn("train/heldout::aug-v1", expanded["splits"]["train"])
        self.assertEqual(manifest["splits"]["validation"], expanded["splits"]["validation"])
        self.assertEqual(manifest["splits"]["test"], expanded["splits"]["test"])
        np.testing.assert_equal(weights, [.5, .5, .5, .5])
        self.assertEqual(weights.sum(), 2)
        with self.assertRaisesRegex(ValueError, "coverage"):
            include_training_views(rows + views[:1], manifest)
        with self.assertRaisesRegex(ValueError, "Original training parent"):
            include_training_views(views, manifest)

    def test_augmented_fitting_cli_uses_canonical_parents_and_exports_head(self):
        rows, views, manifest = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.jsonl"
            index.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            manifest.update(index_file="index.jsonl", index_sha256=hashlib.sha256(index.read_bytes()).hexdigest())
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            arrays = np.random.default_rng(19).normal(size=(len(rows), 768)).astype("float32")
            for folder_name, cache_rows, features, config in (
                ("base", rows, arrays, None),
                ("views", views, arrays[:3].copy(), dict(augmentation_config(), views_per_training_image=1))):
                folder = root / folder_name
                folder.mkdir()
                np.save(folder / "features.npy", features)
                (folder / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in cache_rows), encoding="utf-8")
                provenance = dict(model=["toy", "revision"], pooling="mean_patch_v1", transform="same",
                                  limit=None, replacement_index="toy", supplement=None)
                if config:
                    provenance["augmentation"] = config
                (folder / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/fit_probe.py"),
                       "--cache", str(root / "base"), "--cache", str(root / "views"),
                       "--manifest", str(root / "manifest.json"), "--output", str(root / "output"), "--defer-test"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads((root / "output/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["augmentation"]["augmented_training_rows"], 2)
            self.assertEqual(saved["augmentation"]["cached_views_not_selected"], 1)
            self.assertEqual(saved["test"], {})
            self.assertIn("deferred", saved["test_evaluation"])
            predictions = [json.loads(line) for line in (root / "output/predictions.jsonl").read_text().splitlines()]
            self.assertEqual({r["split"] for r in predictions}, {"validation"})
            self.assertTrue((root / "output/linear_head.npz").is_file())


if __name__ == "__main__":
    unittest.main()
