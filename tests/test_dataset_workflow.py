import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import io

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import dataset_io
import download_dataset


def fixture(root, corrupt=False):
    data = root / "data"
    data.mkdir()
    stream = io.BytesIO()
    Image.new("RGB", (270, 351)).save(stream, format="JPEG")
    for split in dataset_io.SPLITS:
        pq.write_table(pa.Table.from_pylist([
            {"Caption": "fixture", "Image": {"bytes": stream.getvalue(), "path": None}, "Label_A": 0, "Label_B": 0},
            {"Caption": "fixture", "Image": {"bytes": b"broken" if corrupt else stream.getvalue(), "path": None}, "Label_A": 1, "Label_B": 4},
        ]), data / f"{split}-00000-of-00001.parquet")


class WorkflowTests(unittest.TestCase):
    def test_incomplete_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "data/train-00000-of-00001.parquet"
            path.rename(path.with_name("train-00000-of-00002.parquet"))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                dataset_io.split_files(root)

    def test_profiler_from_different_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            output = root / "results/profile.json"
            run = subprocess.run([sys.executable, str(SCRIPTS / "profile_dataset.py"), "--data-dir", str(root), "--output", str(output)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_text())
            for split in result["splits"].values():
                self.assertEqual(split["rows"], 2)
                self.assertEqual(split["label_pairs"], {"0,0": 1, "1,4": 1})
                self.assertEqual(split["groups"]["4"]["sizes"], {"270x351": 1})
                self.assertEqual(split["groups"]["0"]["decoded_samples"], 1)
                self.assertEqual(split["errors"], [])

    def test_corrupt_image_retains_report_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root, corrupt=True)
            output = root / "profile.json"
            run = subprocess.run([sys.executable, str(SCRIPTS / "profile_dataset.py"), "--data-dir", str(root), "--output", str(output)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["splits"]["train"]["errors"]), 1)
            self.assertIsNone(report["splits"]["train"]["groups"]["4"]["width_range"])

    def test_downloader_pins_resolved_revision_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            with patch.object(sys, "argv", ["download_dataset.py", "--data-dir", tmp]), patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as download:
                api.return_value.dataset_info.return_value.sha = dataset_io.DATASET_REVISION
                download_dataset.main()
                self.assertEqual(download.call_args.kwargs["revision"], dataset_io.DATASET_REVISION)
                self.assertEqual(download.call_args.kwargs["repo_type"], "dataset")
            manifest = json.loads((root / "download_manifest.json").read_text())
            self.assertEqual(manifest["revision"], dataset_io.DATASET_REVISION)
            self.assertEqual(set(manifest["files"]), set(dataset_io.SPLITS))


if __name__ == "__main__":
    unittest.main()
