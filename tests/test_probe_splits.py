"""Checks for leakage guards before any real classifier is trained."""
import copy
from pathlib import Path
import sys
import unittest
import json
import subprocess
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from fit_probe import validate_splits
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "sklearn"}:
        raise
    validate_splits = None


@unittest.skipIf(validate_splits is None, "Install requirements-training.txt for linear-probe tests.")
class ProbeSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.manifest = {"held_out_generator": "heldout", "splits": {}}
        for split in ("train", "validation", "test"):
            ids = []
            for target in (0, 1):
                image_id = f"{split}-{target}"
                source = "real" if target == 0 else ("heldout" if split == "test" else "new-generator")
                self.rows.append({"id": image_id, "source": source, "target": target,
                                  "sha256": image_id, "caption": None,
                                  "split": "unassigned" if target else split})
                ids.append(image_id)
            self.manifest["splits"][split] = ids

    def test_valid_primary_experiment(self):
        selected = validate_splits(self.rows, self.manifest, {"train-1"})
        self.assertEqual(len(selected["test"]), 2)

    def test_reject_heldout_training_and_missing_supplement(self):
        with self.assertRaisesRegex(ValueError, "supplemental"):
            validate_splits(self.rows, self.manifest, set())
        self.rows[1]["source"] = "heldout"
        with self.assertRaisesRegex(ValueError, "Held-out"):
            validate_splits(self.rows, self.manifest, {"train-1"})

    def test_reject_cross_split_duplicates_and_related_captions(self):
        for field in ("sha256", "caption"):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = rows[2][field] = "same"
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                validate_splits(rows, self.manifest, {"train-1"})

    def test_reject_moving_published_test_into_train(self):
        self.rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "cannot move"):
            validate_splits(self.rows, self.manifest, {"train-1"})

    def test_reject_inherited_groups_even_when_captions_differ(self):
        for field in ("reference_id", "content_group_id"):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = rows[2][field] = "shared-reference"
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                validate_splits(rows, self.manifest, {"train-1"})

    def test_reject_old_native_pooling_before_loading_feature_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "provenance.json").write_text(json.dumps({"model": ["old", "revision"]}), encoding="utf-8")
            (root / "unused.json").write_text("{}", encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "fit_probe.py"),
                                     "--cache", str(root), "--manifest", str(root / "unused.json"),
                                     "--output", str(root / "output")], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("old native-pooling caches", result.stderr)

    def test_toy_cli_exports_equivalent_single_linear_head(self):
        import numpy as np
        import pickle
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows, splits = [], {}
            for split in ("train", "validation", "test"):
                splits[split] = []
                for i in range(8):
                    image_id = f"{split}-{i}"
                    target = i % 2
                    rows.append(dict(id=image_id, split=split, target=target, sha256=image_id,
                                     caption=None, source="real" if not target else ("heldout" if split == "test" else "seen")))
                    splits[split].append(image_id)
            features = np.random.default_rng(17).normal(size=(24, 768)).astype("float32")
            np.save(root / "features.npy", features)
            (root / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            (root / "provenance.json").write_text(json.dumps(dict(model=["toy", "revision"], pooling="mean_patch_v1",
                transform="toy", limit=None, replacement_index="toy", supplement=None)), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps(dict(held_out_generator="heldout", splits=splits)), encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "fit_probe.py"),
                "--cache", str(root), "--manifest", str(root / "manifest.json"), "--output", str(root / "output")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "output" / "head.pkl").open("rb") as handle:
                head = pickle.load(handle)
            with np.load(root / "output" / "linear_head.npz") as linear:
                np.testing.assert_allclose(features @ linear["weight"] + linear["bias"], head.decision_function(features), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
