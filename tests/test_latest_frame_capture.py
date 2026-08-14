import threading
import time
import unittest

import numpy as np

from latest_frame_capture import LatestFrameCapture


class LatestFrameCaptureTest(unittest.TestCase):
    def test_latest_slot_overwrites_without_backlog(self):
        capture = LatestFrameCapture("bed_test", "unused")
        for value in range(1, 6):
            capture.publish(
                np.full((4, 5, 3), value, dtype=np.uint8),
                capture_mono_ts=float(value),
                capture_wall_ts=1000.0 + value,
            )

        latest = capture.latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.frame_seq, 5)
        self.assertEqual(int(latest.frame[0, 0, 0]), 5)
        self.assertEqual(capture.metrics()["frame_seq"], 5)

    def test_consumer_skips_intermediate_sequences(self):
        capture = LatestFrameCapture("bed_test", "unused")
        capture.publish(np.zeros((2, 2, 3), dtype=np.uint8))
        first = capture.wait_for_frame(0, timeout=0.01)
        self.assertEqual(first.frame_seq, 1)

        capture.publish(np.ones((2, 2, 3), dtype=np.uint8))
        capture.publish(np.full((2, 2, 3), 2, dtype=np.uint8))
        newest = capture.wait_for_frame(first.frame_seq, timeout=0.01)
        self.assertEqual(newest.frame_seq, 3)
        self.assertEqual(int(newest.frame[0, 0, 0]), 2)

    def test_wait_blocks_until_new_frame(self):
        capture = LatestFrameCapture("bed_test", "unused")
        capture.publish(np.zeros((2, 2, 3), dtype=np.uint8))
        received = []

        def consume():
            received.append(capture.wait_for_frame(1, timeout=0.5))

        thread = threading.Thread(target=consume)
        thread.start()
        time.sleep(0.02)
        capture.publish(np.ones((2, 2, 3), dtype=np.uint8))
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(received[0].frame_seq, 2)

    def test_consumer_frame_is_a_copy(self):
        capture = LatestFrameCapture("bed_test", "unused")
        capture.publish(np.zeros((2, 2, 3), dtype=np.uint8))
        consumer = capture.latest()
        consumer.frame[:] = 255
        stored = capture.latest()
        self.assertEqual(int(stored.frame.max()), 0)

    def test_invalid_frame_is_rejected(self):
        capture = LatestFrameCapture("bed_test", "unused")
        with self.assertRaises(ValueError):
            capture.publish(np.array([], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

