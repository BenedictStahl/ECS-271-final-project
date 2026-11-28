import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_jpeg_control import jpeg_control, table_hash, ValidationPairs, deltas


class JPEGControlTests(unittest.TestCase):
    def test_in_memory_reencoding_preserves_geometry_and_original(self):
        buffer = io.BytesIO()
        Image.new("RGB", (320, 240), (100, 120, 140)).save(buffer, format="JPEG", quality=92)
        original = buffer.getvalue()
        with Image.open(io.BytesIO(original)) as image:
            before = table_hash(image)
            controlled, after = jpeg_control(image)
            self.assertEqual(controlled.size, image.size)
            self.assertEqual(controlled.mode, "RGB")
            self.assertNotEqual(before, after)
        self.assertEqual(original, buffer.getvalue())

    def test_dataset_rejects_test_qwen_and_unapproved_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "validation/real/example.jpg"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (256, 240)).save(path, quality=75)
            row = dict(id="validation-real", path="validation/real/example.jpg", image_root=0,
                       split="validation", source="real", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            summary = {"image_roots": [str(root)]}
            data = ValidationPairs([row], summary, "deit3", [row["id"]], "qwen_image_2_1")
            sample = data[0]
            self.assertEqual(tuple(sample["original"].shape), (3, 224, 224))
            self.assertTrue(torch.isfinite(sample["control"]).all())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            for changed in (dict(row, split="test"), dict(row, source="qwen_image_2_1"), dict(row, id="unapproved")):
                with self.assertRaisesRegex(ValueError, "approved seen-generator validation"):
                    ValidationPairs([changed], summary, "deit3", [row["id"]], "qwen_image_2_1")

    def test_paired_metric_deltas_are_zero_for_identical_predictions(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([.2, .8, .7, .9])
        np.testing.assert_equal(deltas(y, p, p), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
