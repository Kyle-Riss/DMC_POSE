import csv
import tempfile
import unittest
from pathlib import Path

from scripts.serve_fallvision_annotation import Store, Update, validate_annotation


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


if __name__ == "__main__":
    unittest.main()
