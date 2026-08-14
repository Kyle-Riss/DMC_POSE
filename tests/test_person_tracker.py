import unittest

import numpy as np

from person_tracker import MultiPersonTracker, PersonDetection


def detection(cx, cy, *, bed=0.0, confidence=0.9):
    xy = np.zeros((17, 2), dtype=np.float32)
    for idx in range(17):
        xy[idx] = [cx + (idx % 4) * 2, cy + (idx // 4) * 3]
    conf = np.full(17, confidence, dtype=np.float32)
    return PersonDetection(
        keypoints_xy=xy,
        keypoints_conf=conf,
        bbox=(cx - 10, cy - 20, cx + 10, cy + 20),
        confidence=confidence,
        bed_overlap=bed,
    )


class MultiPersonTrackerTests(unittest.TestCase):
    def test_detection_order_change_does_not_change_track_identity(self):
        tracker = MultiPersonTracker()
        first = tracker.update(
            [detection(100, 100, bed=0.9), detection(500, 100, bed=0.1)],
            1.0, frame_width=640, frame_height=360,
        )
        ids_by_x = {round((t.bbox[0] + t.bbox[2]) / 2): t.track_id for t in first.tracks}
        second = tracker.update(
            [detection(498, 102, bed=0.1), detection(103, 101, bed=0.9)],
            1.1, frame_width=640, frame_height=360,
        )
        second_ids = {round((t.bbox[0] + t.bbox[2]) / 2): t.track_id for t in second.tracks}
        self.assertEqual(second_ids[103], ids_by_x[100])
        self.assertEqual(second_ids[498], ids_by_x[500])

    def test_primary_prefers_sustained_bed_overlap(self):
        tracker = MultiPersonTracker()
        result = tracker.update(
            [detection(100, 100, bed=0.15), detection(400, 180, bed=0.92)],
            1.0, frame_width=640, frame_height=360,
        )
        self.assertIsNotNone(result.primary)
        self.assertGreater(result.primary.bed_overlap_ema, 0.9)

    def test_continuity_prevents_small_challenger_steal(self):
        tracker = MultiPersonTracker(primary_switch_margin=0.20)
        first = tracker.update(
            [detection(200, 180, bed=0.8)],
            1.0, frame_width=640, frame_height=360,
        )
        primary_id = first.primary_track_id
        result = tracker.update(
            [detection(202, 181, bed=0.65), detection(500, 180, bed=0.95)],
            1.1, frame_width=640, frame_height=360,
        )
        self.assertEqual(result.primary_track_id, primary_id)
        self.assertFalse(result.primary_switched)

    def test_expired_primary_selects_new_track_without_mixing_id(self):
        tracker = MultiPersonTracker(track_ttl_sec=0.5)
        first = tracker.update(
            [detection(100, 100, bed=0.9)],
            1.0, frame_width=640, frame_height=360,
        )
        old_id = first.primary_track_id
        result = tracker.update(
            [detection(500, 200, bed=0.8)],
            2.0, frame_width=640, frame_height=360,
        )
        self.assertNotEqual(result.primary_track_id, old_id)
        self.assertIn(old_id, result.expired_track_ids)

    def test_unobserved_primary_returns_gap_not_stale_skeleton(self):
        tracker = MultiPersonTracker(track_ttl_sec=2.0)
        first = tracker.update(
            [detection(100, 100, bed=0.9)],
            1.0, frame_width=640, frame_height=360,
        )
        result = tracker.update([], 1.1, frame_width=640, frame_height=360)
        self.assertEqual(result.primary_track_id, first.primary_track_id)
        self.assertIsNone(result.primary)


if __name__ == "__main__":
    unittest.main()
