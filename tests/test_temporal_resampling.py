import unittest

import numpy as np

from live_temporal import TemporalShadowRunner
from temporal_sequence import cadence_interval_bounds, decide_observation, observed_sequence_contract


class FakeService:
    threshold = 0.5

    def __init__(self):
        self.windows = []

    def predict(self, window):
        self.windows.append(np.asarray(window).copy())
        return 0.6


class Fake20HzService(FakeService):
    sample_hz = 20.0
    window_rows = 80


def pose():
    xy = np.arange(34, dtype=np.float32).reshape(17, 2) + 10.0
    conf = np.ones(17, dtype=np.float32)
    probs = np.full(6, 1.0 / 6.0, dtype=np.float32)
    return xy, conf, probs


class TemporalResamplingTest(unittest.TestCase):
    def test_20hz_contract_and_bounds_are_explicit(self):
        self.assertEqual(observed_sequence_contract(20.0), "observed_only_20hz_v1")
        minimum, maximum = cadence_interval_bounds(20.0)
        self.assertAlmostEqual(minimum, 0.035)
        self.assertAlmostEqual(maximum, 0.075)

    def test_shared_cadence_decision_boundaries(self):
        self.assertEqual(decide_observation(1.0, None).action, "append")
        self.assertEqual(decide_observation(1.04, 1.0).action, "duplicate_skip")
        self.assertEqual(decide_observation(1.07, 1.0).action, "append")
        self.assertEqual(decide_observation(1.15, 1.0).action, "append")
        self.assertEqual(decide_observation(1.151, 1.0).action, "gap_reset_append")

    def test_late_observation_resets_without_missing_rows(self):
        runner = TemporalShadowRunner(FakeService())
        xy, conf, probs = pose()
        runner.push(0.0, xy, conf, probs)
        status = runner.push(0.31, xy, conf, probs)

        self.assertEqual(status["samples"], 1)
        self.assertEqual(status["missing_samples_window"], 0)
        self.assertEqual(status["missing_samples_total"], 0)
        self.assertEqual(status["gap_reset_total"], 1)
        self.assertAlmostEqual(status["sample_timestamp"], 0.31)
        self.assertEqual(status["sampling_contract"], "observed_only_70_150ms")

    def test_sub_70ms_observation_is_skipped(self):
        runner = TemporalShadowRunner(FakeService())
        xy, conf, probs = pose()
        runner.push(0.0, xy, conf, probs)
        skipped = runner.push(0.04, xy, conf, probs)
        status = runner.push(0.10, xy, conf, probs)

        self.assertEqual(skipped["samples"], 1)
        self.assertEqual(skipped["last_action"], "duplicate_skip")
        self.assertEqual(status["samples"], 2)
        self.assertEqual(status["duplicate_skip_total"], 1)

    def test_70_and_150ms_boundaries_are_valid(self):
        runner = TemporalShadowRunner(FakeService())
        xy, conf, probs = pose()
        runner.push(0.0, xy, conf, probs)
        runner.push(0.070, xy, conf, probs)
        status = runner.push(0.220, xy, conf, probs)

        self.assertEqual(status["samples"], 3)
        self.assertEqual(status["gap_reset_total"], 0)
        self.assertAlmostEqual(status["last_dt_sec"], 0.150)

    def test_over_150ms_resets_window_and_starts_fresh(self):
        runner = TemporalShadowRunner(FakeService())
        xy, conf, probs = pose()
        runner.push(0.0, xy, conf, probs)
        runner.push(0.1, xy, conf, probs)
        status = runner.push(0.251, xy, conf, probs)

        self.assertEqual(status["samples"], 1)
        self.assertEqual(status["gap_reset_total"], 1)
        self.assertFalse(status["ready"])

    def test_pose_gap_invalidates_history_at_150ms(self):
        runner = TemporalShadowRunner(FakeService())
        xy, conf, probs = pose()
        runner.push(0.0, xy, conf, probs)
        runner.push(0.1, xy, conf, probs)
        runner.observe_gap(0.251)
        status = runner.status()

        self.assertEqual(status["samples"], 0)
        self.assertEqual(status["gap_reset_total"], 1)
        self.assertFalse(status["ready"])

    def test_30_consecutive_observations_produce_exact_window(self):
        service = FakeService()
        runner = TemporalShadowRunner(service, inference_stride=1)
        xy, conf, probs = pose()
        for timestamp in np.arange(0.0, 3.0, 0.1):
            status = runner.push(float(timestamp), xy, conf, probs)

        self.assertTrue(status["ready"])
        self.assertEqual(status["samples"], 30)
        self.assertEqual(status["timestamp_source"], "decode_mono_ts")
        self.assertEqual(service.windows[-1].shape, (30, 109))
        self.assertEqual(status["missing_samples_window"], 0)

    def test_20hz_model_contract_drives_80_row_runner(self):
        service = Fake20HzService()
        runner = TemporalShadowRunner(service, inference_stride=1)
        xy, conf, probs = pose()
        for index in range(80):
            status = runner.push(index * 0.05, xy, conf, probs)

        self.assertTrue(status["ready"])
        self.assertEqual(status["sample_hz"], 20.0)
        self.assertEqual(status["window_rows"], 80)
        self.assertEqual(service.windows[-1].shape, (80, 109))

    def test_runner_rejects_checkpoint_cadence_override(self):
        with self.assertRaisesRegex(ValueError, "sample_hz"):
            TemporalShadowRunner(Fake20HzService(), sample_hz=10.0)

    def test_non_monotonic_timestamp_is_skipped(self):
        runner = TemporalShadowRunner(FakeService())
        xy, conf, probs = pose()
        runner.push(1.0, xy, conf, probs, timestamp_source="source_pts")
        status = runner.push(0.9, xy, conf, probs, timestamp_source="source_pts")

        self.assertEqual(status["samples"], 1)
        self.assertEqual(status["non_monotonic_skip_total"], 1)
        self.assertEqual(status["last_action"], "non_monotonic_skip")
        self.assertEqual(status["timestamp_source"], "source_pts")


if __name__ == "__main__":
    unittest.main()
