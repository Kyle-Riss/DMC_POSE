import unittest

import numpy as np

from temporal_features import labels_at, normalize_pose


class TemporalFeaturesTest(unittest.TestCase):
    def test_normalization_is_translation_and_scale_invariant(self):
        xy = np.arange(34, dtype=np.float32).reshape(17, 2) + 1
        conf = np.ones(17, dtype=np.float32)
        first = normalize_pose(xy, conf)["xy_norm"]
        second = normalize_pose(xy * 3.0 + 50.0, conf)["xy_norm"]
        np.testing.assert_allclose(first, second, atol=1e-6)

    def test_missing_pose_has_explicit_visibility(self):
        result = normalize_pose(np.zeros((17, 2)), np.zeros(17))
        self.assertEqual(float(result["visibility"].sum()), 0.0)
        self.assertTrue(np.isnan(result["scale"]))

    def test_binary_label_keeps_active_context(self):
        intervals = [
            {"label": "walking", "start_sec": 0, "end_sec": 2},
            {"label": "fall", "start_sec": 1.5, "end_sec": 4},
        ]
        target, active = labels_at(1.8, intervals)
        self.assertEqual(target, "fall")
        self.assertEqual(active, ["walking", "fall"])

    def test_ignore_has_priority_over_fall(self):
        intervals = [
            {"label": "fall", "start_sec": 1.0, "end_sec": 3.0},
            {"label": "ignore", "start_sec": 0.8, "end_sec": 1.2},
        ]
        target, active = labels_at(1.1, intervals)
        self.assertEqual(target, "ignore")
        self.assertEqual(active, ["fall", "ignore"])


if __name__ == "__main__":
    unittest.main()
