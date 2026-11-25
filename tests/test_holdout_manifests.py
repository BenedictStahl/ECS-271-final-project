"""Review propagation and holdout isolation, without training on project images."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_holdout_manifests import apply_review, make_manifests, verify_additive_rows
try:
    from fit_probe import canonical_cache_rows, check_manifest_status, validate_splits
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "sklearn"}:
        raise
    validate_splits = None


@unittest.skipIf(validate_splits is None, "Training libraries required")
class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.rows, overrides = [], {}
        for split in ("train", "validation", "test"):
            ref = split + "-original:0"
            for source in ("real", "flux3", "sdxl_turbo"):
                path = f"{split}/{source}/00000.jpg"
                digest = hashlib.sha256(path.encode()).hexdigest()
                self.rows.append(dict(id="replacement:" + path, path=path, source=source,
                    target=int(source != "real"), split=split, reference_id=ref,
                    caption="same generic wording", content_group_id="original-caption:same", sha256=digest))
                if source == "real":
                    overrides[ref] = dict(family_id="reviewed-photo:" + split, real_path=path,
                                         real_sha256=digest, split=split, decision="keep")
        self.review = dict(reference_family_overrides=overrides, excluded_reference_ids=[], excluded_image_ids=[])
        self.reviewed, excluded = apply_review(self.rows, self.review, "review-digest")
        self.assertEqual(excluded, [])

    def test_reviewed_caption_collisions_are_cleared_and_holdouts_isolated(self):
        manifests = make_manifests(self.reviewed, ["flux3", "sdxl_turbo"], self.review, "review-digest")
        by_id = {r["id"]: r for r in self.reviewed}
        for generator, manifest in manifests.items():
            self.assertEqual(manifest["status"], "ready")
            for split in ("train", "validation"):
                self.assertNotIn(generator, {by_id[i]["source"] for i in manifest["splits"][split]})
            cohorts = manifest["test_cohorts"]
            ids = [i for group in cohorts.values() for i in group]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), set(manifest["splits"]["test"]))
        self.assertEqual(len({r["content_group_id"] for r in self.reviewed}), 3)

    def test_missing_generator_is_blocked_and_two_generator_runs_are_pilots(self):
        manifests = make_manifests(self.reviewed, ["flux3", "sdxl_turbo", "missing"], self.review, "review-digest")
        self.assertEqual(manifests["flux3"]["status"], "pilot_only")
        with self.assertRaisesRegex(ValueError, "allow-pilot"):
            check_manifest_status(manifests["flux3"])
        check_manifest_status(manifests["flux3"], allow_pilot=True)
        with self.assertRaisesRegex(ValueError, "blocked"):
            check_manifest_status(manifests["missing"], allow_pilot=True)
        self.assertTrue(all(not ids for ids in manifests["missing"]["splits"].values()))

    def test_partial_coverage_cannot_masquerade_as_a_complete_generator(self):
        rows = [r for r in self.reviewed if not (r["source"] == "flux3" and r["split"] == "test")]
        manifests = make_manifests(rows, ["flux3", "sdxl_turbo"], self.review, "review-digest")
        self.assertEqual(manifests["flux3"]["status"], "blocked_incomplete_holdout")
        self.assertEqual(manifests["sdxl_turbo"]["status"], "blocked_no_training_generator")

    def test_changed_real_file_invalidates_review_and_duplicate_source_is_rejected(self):
        review = copy.deepcopy(self.review)
        review["reference_family_overrides"]["train-original:0"]["real_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "mismatched review"):
            apply_review(self.rows, review, "review-digest")
        with self.assertRaisesRegex(ValueError, "repeated images"):
            make_manifests(self.reviewed + [self.reviewed[1]], ["flux3", "sdxl_turbo"], self.review, "review-digest")

    def test_cache_identity_bound_but_group_metadata_can_be_updated(self):
        expected = {r["id"]: r for r in self.reviewed}
        updated = canonical_cache_rows(self.rows, expected)
        self.assertEqual(updated, self.reviewed)
        changed = copy.deepcopy(self.rows)
        changed[0]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            canonical_cache_rows(changed, expected)

    def test_review_cannot_disable_actual_shared_family_guard(self):
        rows = copy.deepcopy(self.reviewed)
        review = copy.deepcopy(self.review)
        for row in rows:
            row["content_group_id"] = row["reviewed_family_id"] = "reviewed-photo:shared"
        for approval in review["reference_family_overrides"].values():
            approval["family_id"] = "reviewed-photo:shared"
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            make_manifests(rows, ["flux3", "sdxl_turbo"], review, "review-digest")

    def test_unapproved_review_metadata_cannot_bypass_caption_guard(self):
        manifest = make_manifests(self.reviewed, ["flux3", "sdxl_turbo"], self.review, "review-digest")["flux3"]
        manifest["review_sha256"] = "different-review"
        with self.assertRaisesRegex(ValueError, "not authorized"):
            validate_splits(self.reviewed, manifest, {r["id"] for r in self.reviewed if r["target"]})

    def test_later_deliveries_may_add_but_not_change_existing_images(self):
        added = dict(self.rows[1], id="new", source="z_image_turbo")
        verify_additive_rows(self.rows, self.rows + [added])
        changed = copy.deepcopy(self.rows)
        changed[1]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed/removed"):
            verify_additive_rows(self.rows, changed)
        with self.assertRaisesRegex(ValueError, "changed/removed"):
            verify_additive_rows(self.rows, self.rows[1:])


if __name__ == "__main__":
    unittest.main()
