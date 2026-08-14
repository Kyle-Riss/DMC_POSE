import unittest
from pathlib import Path

from scripts.prepare_fallvision_manifest import (
    classify_path,
    eligibility_fields,
    parse_recording_name,
)


class FallVisionManifestParsingTest(unittest.TestCase):
    def test_variant_suffixes_do_not_change_recording_group(self):
        base = parse_recording_name(Path("B_N_146_resized_anonymized.mp4"))
        self.assertEqual(base["recording_id"], "B_N_146")
        self.assertEqual(base["recording_number"], 146)
        self.assertEqual(base["variant"], "resized_anonymized")

    def test_smoke_group_classification(self):
        root = Path("/dataset")
        self.assertEqual(
            classify_path(root / "fall_bed/f_raw_b_3/B_N_01.mp4", root),
            ("fall", "bed"),
        )
        self.assertEqual(
            classify_path(root / "nonfall_bed/nf_raw_b_2/B_N_01.mp4", root),
            ("non_fall", "bed"),
        )

    def test_unclassified_path_is_rejected(self):
        root = Path("/dataset")
        self.assertIsNone(classify_path(root / "unknown/B_N_01.mp4", root))

    def test_label_availability_is_separate_from_temporal_readiness(self):
        fall = eligibility_fields("fall")
        self.assertTrue(fall["video_classification_eligible"])
        self.assertFalse(fall["temporal_tcn_eligible"])
        self.assertFalse(fall["subject_disjoint_split_ready"])
        self.assertIn("fall_event_intervals_missing", fall["excluded_reasons"])

        non_fall = eligibility_fields("non_fall")
        self.assertTrue(non_fall["video_classification_eligible"])
        self.assertNotIn("fall_event_intervals_missing", non_fall["excluded_reasons"])


if __name__ == "__main__":
    unittest.main()
