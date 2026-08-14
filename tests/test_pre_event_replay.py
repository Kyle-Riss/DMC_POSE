import unittest

import numpy as np

from person_tracker import PersonDetection
from pre_event_replay import ReplayPoseFrame, replay_temporal_context


class FakeService:
    threshold = 0.5

    def __init__(self):
        self.calls = 0

    def predict(self, window):
        self.calls += 1
        self.last_shape = window.shape
        return 0.9


def detection(x_offset=0.0):
    xy = np.stack([
        np.linspace(10, 60, 17) + x_offset,
        np.linspace(20, 100, 17),
    ], axis=1).astype(np.float32)
    conf = np.full(17, 0.9, dtype=np.float32)
    return PersonDetection(
        keypoints_xy=xy,
        keypoints_conf=conf,
        bbox=(10 + x_offset, 20, 60 + x_offset, 100),
        confidence=0.9,
        bed_overlap=0.8,
    )


class PreEventReplayTests(unittest.TestCase):
    def test_replay_builds_ready_context_and_persistent_candidate(self):
        service = FakeService()
        frames = [
            ReplayPoseFrame(i * 0.1, 160, 90, (detection(),))
            for i in range(35)
        ]

        status = replay_temporal_context(
            frames, service,
            lambda batch: np.tile(
                np.array([[0, 0, 0, 0, 0, 1]], dtype=np.float32),
                (len(batch), 1),
            ),
        )

        self.assertTrue(status["ready"])
        self.assertTrue(status["candidate"])
        self.assertEqual(status["requested_frames"], 35)
        self.assertEqual(status["observed_pose_frames"], 35)
        self.assertEqual(status["prediction_count"], 2)
        self.assertEqual(service.last_shape, (30, 109))

    def test_pose_gap_prevents_false_ready(self):
        service = FakeService()
        frames = []
        for i in range(35):
            detections = () if i == 18 else (detection(),)
            frames.append(ReplayPoseFrame(i * 0.1, 160, 90, detections))
        status = replay_temporal_context(
            frames, service,
            lambda batch: np.zeros((len(batch), 6), dtype=np.float32),
        )
        self.assertFalse(status["ready"])
        self.assertGreaterEqual(status["gap_reset_total"], 1)

    def test_late_gap_preserves_completed_ready_window(self):
        service = FakeService()
        frames = [
            ReplayPoseFrame(i * 0.1, 160, 90, (detection(),))
            for i in range(30)
        ]
        frames.extend([
            ReplayPoseFrame(3.0, 160, 90, ()),
            ReplayPoseFrame(3.1, 160, 90, ()),
        ])
        status = replay_temporal_context(
            frames, service,
            lambda batch: np.zeros((len(batch), 6), dtype=np.float32),
        )
        self.assertTrue(status["ready"])
        self.assertEqual(status["samples"], 30)
        self.assertEqual(status["latest_segment_samples"], 0)
        self.assertGreaterEqual(status["gap_reset_total"], 1)

    def test_classifier_shape_is_enforced(self):
        frames = [ReplayPoseFrame(0.0, 160, 90, (detection(),))]
        with self.assertRaises(ValueError):
            replay_temporal_context(
                frames, FakeService(),
                lambda batch: np.zeros((len(batch), 5), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
