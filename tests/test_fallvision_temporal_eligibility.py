import unittest

from fallvision_temporal_eligibility import evaluate_temporal_eligibility


class FallVisionTemporalEligibilityTest(unittest.TestCase):
    def test_all_required_conditions_are_needed(self):
        record = {
            "raw_video_exists": True,
            "decode_ok": True,
            "temporal_annotation_complete": True,
            "split_group_resolved": True,
            "observed_pose_samples_available": True,
            "pre_context_sufficient": True,
            "pre_onset_observed_samples": 35,
            "pre_onset_ready": True,
            "event_evaluable": True,
        }
        result = evaluate_temporal_eligibility(record)
        self.assertTrue(result["temporal_tcn_eligible"])
        self.assertEqual(result["temporal_tcn_blockers"], [])

    def test_missing_boundary_and_split_stay_blocked(self):
        result = evaluate_temporal_eligibility(
            {
                "raw_video_exists": True,
                "decode_ok": True,
                "observed_pose_samples_available": True,
                "pre_context_sufficient": True,
            }
        )
        self.assertFalse(result["temporal_tcn_eligible"])
        self.assertIn("temporal_annotation_complete", result["temporal_tcn_blockers"])
        self.assertIn("split_group_resolved", result["temporal_tcn_blockers"])


if __name__ == "__main__":
    unittest.main()
