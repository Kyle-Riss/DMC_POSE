import unittest

import numpy as np

from latest_frame_capture import LatestFrameCapture
from motion_watcher import MotionWatcher


class MotionWatcherTest(unittest.TestCase):
    def make_watcher(self, **kwargs):
        capture = LatestFrameCapture("test", "unused")
        return MotionWatcher(
            "test",
            capture,
            small_width=64,
            small_height=36,
            pixel_threshold=10,
            motion_ratio_threshold=0.02,
            consecutive_hits=2,
            burst_hold_sec=1.0,
            **kwargs,
        )

    def test_two_hits_open_burst_and_hold_it(self):
        watcher = self.make_watcher()
        frames = []
        for x in (0, 10, 20):
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            frame[20:60, x:x + 40] = 255
            frames.append(frame)

        watcher.process_frame(frames[0], frame_seq=1, mono_ts=1.0)
        first = watcher.process_frame(frames[1], frame_seq=2, mono_ts=1.1)
        second = watcher.process_frame(frames[2], frame_seq=3, mono_ts=1.2)

        self.assertFalse(first["burst_active"])
        self.assertTrue(second["burst_active"])
        self.assertEqual(second["motion_trigger_total"], 1)
        self.assertTrue(watcher.burst_active(now=2.1))
        self.assertFalse(watcher.burst_active(now=2.21))

    def test_single_glitch_does_not_trigger(self):
        watcher = self.make_watcher()
        dark = np.zeros((90, 160, 3), dtype=np.uint8)
        bright = np.full((90, 160, 3), 255, dtype=np.uint8)
        watcher.process_frame(dark, frame_seq=1, mono_ts=1.0)
        status = watcher.process_frame(bright, frame_seq=2, mono_ts=1.1)
        self.assertFalse(status["burst_active"])
        self.assertEqual(status["motion_hit_streak"], 0)

    def test_roi_ignores_motion_outside_bed_neighborhood(self):
        watcher = self.make_watcher(roi_margin_ratio=0.0)
        roi = (80, 20, 150, 80)
        frame1 = np.zeros((100, 160, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        frame3 = frame1.copy()
        frame2[10:60, 0:50] = 255
        frame3[20:70, 0:50] = 255
        watcher.process_frame(frame1, frame_seq=1, mono_ts=1.0, roi_bbox=roi)
        watcher.process_frame(frame2, frame_seq=2, mono_ts=1.1, roi_bbox=roi)
        status = watcher.process_frame(
            frame3, frame_seq=3, mono_ts=1.2, roi_bbox=roi
        )
        self.assertFalse(status["burst_active"])
        self.assertEqual(status["motion_ratio"], 0.0)

    def test_status_reports_processing_fps(self):
        watcher = self.make_watcher()
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        for seq, ts in enumerate((1.0, 1.1, 1.2), start=1):
            watcher.process_frame(frame, frame_seq=seq, mono_ts=ts)
        status = watcher.status(now=1.2)
        self.assertAlmostEqual(status["watcher_fps"], 10.0, places=4)
        self.assertEqual(status["watcher_processed_total"], 3)

    def test_pre_event_ring_is_time_bounded_and_decodable(self):
        watcher = self.make_watcher(
            pre_event_duration_sec=1.0,
            pre_event_sample_hz=10.0,
            pre_event_frame_width=80,
        )
        frame = np.full((90, 160, 3), 127, dtype=np.uint8)
        for seq in range(16):
            watcher.process_frame(
                frame, frame_seq=seq + 1, mono_ts=1.0 + seq * 0.1
            )
        items = watcher.pre_event_snapshot()
        status = watcher.status(now=2.5)
        self.assertGreaterEqual(len(items), 9)
        self.assertLessEqual(items[-1].mono_ts - items[0].mono_ts, 1.01)
        self.assertEqual(items[-1].frame_seq, 16)
        decoded = items[-1].decode()
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[1], 80)
        self.assertTrue(status["pre_event_ready"])
        self.assertGreater(status["pre_event_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
