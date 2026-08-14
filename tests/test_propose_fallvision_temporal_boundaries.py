import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.propose_fallvision_temporal_boundaries import (
    Params,
    extract_provided_keypoint_curve,
    moving_average,
    propose,
)


class TemporalBoundaryProposalTest(unittest.TestCase):
    def test_moving_average_preserves_length(self):
        values = np.arange(10, dtype=np.float64)
        self.assertEqual(len(moving_average(values, 2)), len(values))

    def test_proposal_is_ordered_and_bounded(self):
        curve = np.zeros(60, dtype=np.float64)
        curve[20:31] = np.linspace(0.2, 1.0, 11)
        curve[31:40] = np.linspace(0.8, 0.1, 9)
        result = propose(
            {"motion": curve, "pose": curve * 0.5},
            30.0,
            Params(0.03, 0.12, 0.03, 0.12, 0.20, 0.5),
        )
        boundaries = [
            result["proposed_fall_onset_frame"],
            result["proposed_impact_frame"],
            result["proposed_post_fall_stable_frame"],
            result["proposed_fall_end_frame"],
        ]
        self.assertEqual(boundaries, sorted(boundaries))
        self.assertGreaterEqual(result["proposed_onset_earliest_frame"], 0)
        self.assertLess(result["proposed_onset_latest_frame"], 60)

    def test_provided_keypoint_csv_becomes_one_value_per_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keypoints.csv"
            names = ("Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip", "Left Knee")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("Frame", "Keypoint", "X", "Y", "Confidence"))
                for frame in range(1, 5):
                    for index, name in enumerate(names):
                        writer.writerow((frame, name, 10 + index, 20 + frame + index, 0.9))
            curve = extract_provided_keypoint_curve(path)
            self.assertEqual(len(curve), 4)
            self.assertTrue(np.all(np.isfinite(curve)))


if __name__ == "__main__":
    unittest.main()
