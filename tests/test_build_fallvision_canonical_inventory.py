import unittest

from scripts.build_fallvision_canonical_inventory import (
    determine_pair_status,
    parse_archive_name,
    recording_id_from_member,
)


class FallVisionCanonicalInventoryTest(unittest.TestCase):
    def test_archive_name_parses_typo_keypoint_suffix(self):
        parsed = parse_archive_name("nf_mask_b_1_ketpoints_csv.rar")
        self.assertEqual(parsed["activity_label"], "non_fall")
        self.assertEqual(parsed["scene_id"], "bed")
        self.assertEqual(parsed["modality"], "keypoint_csv")

    def test_recording_id_preserves_original_provenance(self):
        self.assertEqual(
            recording_id_from_member(
                "folder/S_D_00462_keypoints.csv", "keypoint_csv"
            ),
            "S_D_00462",
        )
        self.assertEqual(
            recording_id_from_member(
                "folder/B_N_336 - Copy_resized.mp4", "raw"
            ),
            "B_N_336 - Copy_resized",
        )

    def test_pair_status_distinguishes_missing_components(self):
        self.assertEqual(determine_pair_status(True, True, True), "complete")
        self.assertEqual(determine_pair_status(True, True, False), "missing_csv")
        self.assertEqual(determine_pair_status(True, False, False), "raw_only")
        self.assertEqual(determine_pair_status(False, True, True), "missing_raw")


if __name__ == "__main__":
    unittest.main()
