import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_probe import CHECKPOINTS, load_encoder


class LocalCheckpointTests(unittest.TestCase):
    def test_wrong_checkpoint_and_changed_weights_rejected_without_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(repository="facebook/dinov3-vits16-pretrain-lvd1689m", revision="wrong", files=[])
            path = root / "MANIFEST.json"
            path.write_text(json.dumps(manifest))
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Unexpected Hub access")):
                with self.assertRaisesRegex(ValueError, "repository/revision"):
                    load_encoder("dinov3", "cpu", local_checkpoint=root)
                manifest.update(repository=CHECKPOINTS["dinov3"][0], revision=CHECKPOINTS["dinov3"][1])
                for name in ("config.json", "model.safetensors", "LICENSE.md"):
                    (root / name).write_bytes(b"original")
                    manifest["files"].append(dict(filename=name, bytes=8, sha256=hashlib.sha256(b"original").hexdigest()))
                path.write_text(json.dumps(manifest))
                (root / "model.safetensors").write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum mismatch: model.safetensors"):
                    load_encoder("dinov3", "cpu", local_checkpoint=root)


if __name__ == "__main__":
    unittest.main()
