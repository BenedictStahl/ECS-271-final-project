import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from replacement_dataset import prepare_rows, indexed_rows


class ReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "images"
        refs, metadata = [], []
        for split in ("train", "validation", "test"):
            refs.append({"reference_path": f"{split}/real/00000.jpg",
                         "defactify_real_id": split + "-original:0",
                         "original_split_row_index": 0, "original_caption": "same conservative caption flag"})
            for source in ("real", "flux3"):
                relative = f"{split}/{source}/00000.jpg"
                path = self.data / relative
                path.parent.mkdir(parents=True)
                path.write_bytes((split + source).encode())
                if source != "real":
                    metadata.append(dict(path=relative, split=split, index=0, source=source,
                                         label_a=1, label_b=99, caption=split + " different wording"))
        # Numeric generator ID deliberately differs: source names are authoritative.
        metadata.append(dict(metadata[0], path="train/flux3/00001.jpg", index=1))
        self.metadata = self.data / "metadata.csv"
        with self.metadata.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=metadata[0])
            writer.writeheader()
            writer.writerows(metadata)
        self.refs = self.root / "references.jsonl"
        self.refs.write_text("\n".join(json.dumps(r) for r in refs), encoding="utf-8")

    def test_inventory_real_once_inherited_groups_and_unavailable_rows(self):
        rows, summary = prepare_rows(self.data, self.refs)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r["target"] == 0 for r in rows), 3)
        self.assertEqual(summary["unavailable_metadata_images"], 1)
        self.assertEqual(summary["cross_split_content_groups"], 1)
        self.assertEqual({r["assumed_seed"] for r in rows if r["target"]}, {42})
        self.assertEqual(rows[0]["reference_id"], rows[1]["reference_id"])

    def test_changed_image_is_rejected_by_cached_index(self):
        rows, summary = prepare_rows(self.data, self.refs)
        index = self.root / "index.jsonl"
        index.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        index.with_name("dataset_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.assertEqual(len(list(indexed_rows(index))), 6)
        (self.data / rows[0]["path"]).write_bytes(b"changed")
        self.assertEqual(len(list(indexed_rows(index, exclude_ids={rows[0]["id"]}))), 5)
        with self.assertRaisesRegex(ValueError, "changed"):
            list(indexed_rows(index))

    def test_rejects_duplicate_or_missing_metadata(self):
        text = self.metadata.read_text(encoding="utf-8")
        self.metadata.write_text(text + text.splitlines()[1] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate metadata"):
            prepare_rows(self.data, self.refs)
        self.metadata.write_text(text.splitlines()[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing synthetic metadata"):
            prepare_rows(self.data, self.refs)

    def test_additive_roots_and_prefixed_metadata_preserve_identity(self):
        original, _ = prepare_rows(self.data, self.refs)
        other = self.root / "delivery"
        for split in ("train", "validation", "test"):
            path = self.data / split / "flux3/00000.jpg"
            target = other / split / "flux3/00000.jpg"
            target.parent.mkdir(parents=True)
            path.rename(target)
        text = self.metadata.read_text(encoding="utf-8")
        for split in ("train", "validation", "test"):
            text = text.replace(split + "/flux3/", "generated/" + split + "/flux3/")
        self.metadata.write_text(text, encoding="utf-8")
        rows, summary = prepare_rows(self.data, self.refs, self.metadata, [self.data, other])
        for old, new in zip(original, rows):
            self.assertEqual(old, {k: v for k, v in new.items() if k != "image_root"})
        index = self.root / "index.jsonl"
        index.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        index.with_name("dataset_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        self.assertEqual(len(list(indexed_rows(index))), 6)
        duplicate = other / "train/real/00000.jpg"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(b"duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate image across roots"):
            prepare_rows(self.data, self.refs, self.metadata, [self.data, other])


if __name__ == "__main__":
    unittest.main()
