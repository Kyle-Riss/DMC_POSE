import unittest

import numpy as np
import pandas as pd

from build_temporal_windows import feature_matrix, windows_from_video


def sample_df(rows=6):
    data = {"timestamp_sec": np.arange(rows) / 10, "target": ["non_fall"] * (rows - 1) + ["fall"], "video_id": ["v"] * rows, "subject_id": ["s"] * rows, "split": ["train"] * rows, "person_detected": [True] * rows}
    for i in range(17):
        data[f"kpt_{i}_x_norm"] = np.arange(rows, dtype=float)
        data[f"kpt_{i}_y_norm"] = np.arange(rows, dtype=float)
        data[f"kpt_{i}_conf"] = np.ones(rows)
        data[f"kpt_{i}_visible"] = np.ones(rows)
    for i in range(6):
        data[f"pose_prob_{i}"] = np.zeros(rows)
    return pd.DataFrame(data)


class TemporalWindowsTest(unittest.TestCase):
    def test_feature_shape_includes_velocity(self):
        matrix, names = feature_matrix(sample_df())
        self.assertEqual(matrix.shape, (6, 109))
        self.assertEqual(len(names), 109)

    def test_target_is_causal_window_end(self):
        windows, targets, metadata, _ = windows_from_video(sample_df(), 3, 1)
        self.assertEqual(targets, [0, 0, 0, 1])
        self.assertEqual(metadata[-1]["end_sec"], 0.5)
        self.assertEqual(windows[0].shape, (3, 109))

    def test_missing_person_rows_are_never_window_members(self):
        df = sample_df(7)
        df.loc[3, "person_detected"] = False
        windows, _, metadata, _ = windows_from_video(df, 3, 1)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(meta["end_sec"] <= 0.2 or meta["start_sec"] >= 0.4 for meta in metadata))

    def test_track_change_breaks_window(self):
        df = sample_df(8)
        df["track_id"] = [1, 1, 1, 1, 2, 2, 2, 2]
        windows, _, metadata, _ = windows_from_video(df, 3, 1)
        self.assertEqual(len(windows), 4)
        self.assertTrue(all((meta["start_sec"] < 0.4 and meta["end_sec"] < 0.4) or meta["start_sec"] >= 0.4 for meta in metadata))
        self.assertEqual({meta["track_id"] for meta in metadata}, {1, 2})

    def test_over_150ms_gap_breaks_window(self):
        df = sample_df(6)
        df["timestamp_sec"] = [0.0, 0.1, 0.2, 0.4, 0.5, 0.6]
        windows, _, metadata, _ = windows_from_video(df, 3, 1)
        self.assertEqual(len(windows), 2)
        self.assertEqual([(meta["start_sec"], meta["end_sec"]) for meta in metadata], [(0.0, 0.2), (0.4, 0.6)])

    def test_ignore_endpoint_is_not_a_training_window(self):
        df = sample_df(7)
        df["target"] = ["non_fall", "non_fall", "ignore", "ignore", "fall", "fall", "fall"]
        _, targets, metadata, _ = windows_from_video(df, 3, 1)
        self.assertEqual(targets, [1, 1, 1])
        self.assertEqual([row["end_sec"] for row in metadata], [0.4, 0.5, 0.6])
        self.assertEqual(metadata[0]["ignored_fraction"], 0.6667)


if __name__ == "__main__":
    unittest.main()
