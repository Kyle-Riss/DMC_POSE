import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_temporal_windows import feature_matrix
from live_temporal import TemporalModelService, TemporalShadowRunner
from temporal_features import normalize_pose, temporal_feature_vector


class FakeService:
    threshold = 0.7

    def __init__(self):
        self.windows = []

    def predict(self, window):
        self.windows.append(np.asarray(window).copy())
        return 0.8


def pose(offset=0.0):
    xy = np.arange(34, dtype=np.float32).reshape(17, 2) + 10.0
    xy[:, 0] += np.linspace(0.0, offset, 17, dtype=np.float32)
    return xy, np.ones(17, dtype=np.float32)


def offline_row(xy, conf, probs, timestamp):
    normalized = normalize_pose(xy, conf)
    row = {
        "timestamp_sec": timestamp,
        "person_detected": int(np.asarray(normalized["visibility"]).sum() >= 5),
    }
    for index in range(17):
        row[f"kpt_{index}_x_norm"] = normalized["xy_norm"][index, 0]
        row[f"kpt_{index}_y_norm"] = normalized["xy_norm"][index, 1]
        row[f"kpt_{index}_conf"] = conf[index]
        row[f"kpt_{index}_visible"] = normalized["visibility"][index]
    for index in range(6):
        row[f"pose_prob_{index}"] = probs[index]
    return row


class LiveTemporalTest(unittest.TestCase):
    def test_live_vector_matches_offline_feature_builder(self):
        probs0 = np.linspace(0.0, 0.5, 6, dtype=np.float32)
        probs1 = probs0[::-1].copy()
        xy0, conf0 = pose(0.0)
        xy1, conf1 = pose(5.0)
        vector0, norm0, visible0 = temporal_feature_vector(xy0, conf0, probs0)
        vector1, _, _ = temporal_feature_vector(
            xy1, conf1, probs1,
            previous_xy_norm=norm0, previous_visibility=visible0, dt=0.1,
        )
        frame = pd.DataFrame([
            offline_row(xy0, conf0, probs0, 0.0),
            offline_row(xy1, conf1, probs1, 0.1),
        ])
        matrix, _ = feature_matrix(frame)
        np.testing.assert_allclose(vector0, matrix[0], atol=1e-6)
        np.testing.assert_allclose(vector1, matrix[1], atol=1e-5)

    def test_runner_warms_up_persists_and_resets_after_gap(self):
        service = FakeService()
        runner = TemporalShadowRunner(service)
        xy, conf = pose(1.0)
        probs = np.full(6, 1.0 / 6.0, dtype=np.float32)
        for index in range(30):
            status = runner.push(index / 10.0, xy, conf, probs)
        self.assertTrue(status["ready"])
        self.assertEqual(status["prediction_count"], 1)
        self.assertFalse(status["candidate"])
        for index in range(30, 35):
            status = runner.push(index / 10.0, xy, conf, probs)
        self.assertEqual(status["prediction_count"], 2)
        self.assertTrue(status["candidate"])
        self.assertEqual(service.windows[-1].shape, (30, 109))
        runner.observe_gap(4.0)
        self.assertEqual(runner.status()["samples"], 0)
        self.assertFalse(runner.status()["candidate"])

    def test_latest_observation_exposes_only_last_appended_row(self):
        runner = TemporalShadowRunner(FakeService())
        xy, conf = pose(1.0)
        probs = np.full(6, 1.0 / 6.0, dtype=np.float32)
        runner.push(1.0, xy, conf, probs)
        sample_ts, vector = runner.latest_observation()
        self.assertEqual(sample_ts, 1.0)
        self.assertEqual(vector.shape, (109,))
        vector[:] = 99
        self.assertFalse(np.all(runner.latest_observation()[1] == 99))
        runner.observe_gap(2.0)
        self.assertIsNone(runner.latest_observation())

    def test_trained_artifact_predicts_one_offline_window(self):
        root = Path(__file__).resolve().parents[1]
        model_dir = root / "runs/temporal_tcn/gmdcsa24_tcn"
        window_path = root / "external_datasets/windows/gmdcsa24_3s/test.npz"
        if not (model_dir / "model.pt").exists() or not window_path.exists():
            self.skipTest("trained temporal artifact is not available")
        service = TemporalModelService(
            model_dir / "model.pt", model_dir / "report.json", device="cpu"
        )
        window = np.load(window_path)["x"][0]
        probability = service.predict(window)
        self.assertTrue(0.0 <= probability <= 1.0)


if __name__ == "__main__":
    unittest.main()
