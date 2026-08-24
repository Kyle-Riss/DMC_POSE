import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.serve_fallvision_annotation import FRAMES, IdentityUpdate, Store, Update, validate_annotation


class FallVisionAnnotationValidationTest(unittest.TestCase):
    def test_complete_annotation_requires_ordered_boundaries(self):
        valid = Update(
            fall_onset_frame=10,
            impact_frame=20,
            post_fall_stable_frame=30,
            fall_end_frame=40,
            onset_earliest_frame=8,
            onset_latest_frame=12,
            annotation_status="complete",
            annotation_confidence="high",
        )
        self.assertEqual(validate_annotation(valid, 50), [])

    def test_rejects_reverse_order_and_incomplete_complete_status(self):
        invalid = Update(
            fall_onset_frame=20,
            impact_frame=10,
            annotation_status="complete",
        )
        errors = validate_annotation(invalid, 50)
        self.assertTrue(any("onset <= impact" in error for error in errors))
        self.assertTrue(any("complete requires" in error for error in errors))

    def test_store_exposes_proposal_without_persisting_it_into_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "annotations.csv"
            proposals = Path(directory) / "proposals.csv"
            annotation_row = {
                "video_id": "v1", "frame_count": "50", "media_sha256": "abc",
                "fall_onset_frame": "", "impact_frame": "", "post_fall_stable_frame": "",
                "fall_end_frame": "", "onset_earliest_frame": "", "onset_latest_frame": "",
                "annotation_status": "unreviewed", "annotation_confidence": "",
                "annotator": "", "notes": "",
            }
            with annotations.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(annotation_row))
                writer.writeheader()
                writer.writerow(annotation_row)
            with proposals.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["video_id", "proposed_fall_onset_frame"],
                )
                writer.writeheader()
                writer.writerow({"video_id": "v1", "proposed_fall_onset_frame": "12"})
            store = Store(annotations, proposals)
            self.assertEqual(store.public()[0]["proposed_fall_onset_frame"], "12")
            with annotations.open(newline="", encoding="utf-8") as handle:
                persisted = next(csv.DictReader(handle))
            self.assertNotIn("proposed_fall_onset_frame", persisted)

    def test_progress_and_recording_group_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            fields = ["video_id", "recording_id", "frame_count", "annotation_status", *FRAMES,
                      "annotation_confidence", "annotator", "notes"]
            rows = [
                {**{field: "" for field in fields}, "video_id": "a", "recording_id": "r1", "frame_count": "20", "annotation_status": "complete"},
                {**{field: "" for field in fields}, "video_id": "b", "recording_id": "r1", "frame_count": "20", "annotation_status": "unreviewed"},
                {**{field: "" for field in fields}, "video_id": "c", "recording_id": "r2", "frame_count": "20", "annotation_status": "excluded"},
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            store = Store(path)
            progress = store.progress()
            self.assertEqual(progress["views_complete"], 1)
            self.assertEqual(progress["views_reviewed"], 2)
            self.assertEqual(progress["views_resolved"], 2)
            self.assertEqual(progress["recordings_total"], 2)
            self.assertEqual(progress["recordings_complete"], 0)
            self.assertEqual(progress["recordings_reviewed"], 1)
            self.assertEqual(progress["recordings_resolved"], 1)
            self.assertEqual(progress["next_unresolved_index"], 1)
            self.assertEqual(store.recording_indices("r1"), [0, 1])

    def test_identity_update_is_atomic_and_rejects_subject_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.csv"
            identity = root / "identity.json"
            fields = ["video_id", "recording_id", "frame_count", "annotation_status"]
            with annotations.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerow({"video_id":"a","recording_id":"r1","frame_count":"20","annotation_status":"unreviewed"})
                writer.writerow({"video_id":"b","recording_id":"r2","frame_count":"20","annotation_status":"unreviewed"})
            identity.write_text(json.dumps({"recordings":{"r1":{},"r2":{}}}), encoding="utf-8")
            store = Store(annotations, identity_path=identity)
            saved = store.update_identity("r1", IdentityUpdate(subject_id="person_a", session_id="s1", split="train"))
            self.assertEqual(saved["split"], "train")
            with self.assertRaisesRegex(ValueError, "already belongs"):
                store.update_identity("r2", IdentityUpdate(subject_id="person_a", session_id="s2", split="test"))
            persisted = json.loads(identity.read_text(encoding="utf-8"))
            self.assertEqual(persisted["recordings"]["r1"]["subject_id"], "person_a")

    def test_public_training_gate_refreshes_after_annotation_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations.csv"
            identity = root / "identity.json"
            fields = [
                "video_id", "recording_id", "frame_count", "annotation_status", *FRAMES,
                "annotation_confidence", "annotator", "notes",
                "training_eligible", "training_blockers",
            ]
            blank = {field: "" for field in fields}
            rows = [
                {**blank, "video_id": "a", "recording_id": "r1", "frame_count": "50",
                 "annotation_status": "unreviewed", "training_eligible": "false",
                 "training_blockers": "temporal_annotation_incomplete;subject_identity_unknown"},
                {**blank, "video_id": "b", "recording_id": "r1", "frame_count": "50",
                 "annotation_status": "unreviewed", "training_eligible": "false",
                 "training_blockers": "temporal_annotation_incomplete;subject_identity_unknown"},
            ]
            with annotations.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            identity.write_text(json.dumps({"recordings": {"r1": {}}}), encoding="utf-8")
            store = Store(annotations, identity_path=identity)
            complete = Update(
                fall_onset_frame=10, impact_frame=20, post_fall_stable_frame=30,
                fall_end_frame=40, onset_earliest_frame=9, onset_latest_frame=11,
                annotation_status="complete", annotation_confidence="high",
            )
            store.update(0, complete)
            first = store.public()[0]
            self.assertEqual(first["training_eligible"], "false")
            self.assertIn("multiview_recording_incomplete", first["training_blockers"])
            store.update(1, complete)
            store.update_identity(
                "r1", IdentityUpdate(subject_id="person_a", session_id="s1", split="train")
            )
            refreshed = store.public()
            self.assertTrue(all(row["training_eligible"] == "true" for row in refreshed))
            self.assertTrue(all(row["training_blockers"] == "" for row in refreshed))


if __name__ == "__main__":
    unittest.main()
