"""Checks for evaluation cohorts, labels and portable path safety."""
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_handoff import checked_path, cohort_metrics, validate_rows


class EvaluationTests(unittest.TestCase):
    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                checked_path(Path(folder), "../other/file.jpg")

    def test_test_only_labels_and_unique_ids(self):
        sources = ["real", "flux3", "sdxl_turbo", "z_image_turbo", "mage_flow", "qwen_image_2_1"]
        rows = [dict(id=f"{s}/{i}", split="test", source=s, target=int(s != "real")) for s in sources for i in range(7500)]
        validate_rows(rows)
        rows[0]["split"] = "validation"
        with self.assertRaises(ValueError):
            validate_rows(rows)
        rows[0]["split"] = "test"
        rows[-1]["id"] = rows[0]["id"]
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_qwen_failure_not_hidden_by_seen_cohort(self):
        rows = [dict(source="real", target=0, probability=.1),
                dict(source="flux3", target=1, probability=.9),
                dict(source="qwen_image_2_1", target=1, probability=.05)]
        result = cohort_metrics(rows)
        self.assertEqual(result["seen_generators"]["auroc"], 1.)
        self.assertEqual(result["held_out_qwen"]["auroc"], 0.)
        self.assertEqual(result["all_generators"]["auroc"], .5)
        self.assertEqual(result["held_out_qwen"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
