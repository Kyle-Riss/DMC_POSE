import time
import unittest

import numpy as np

from latest_frame_capture import LatestFrameCapture


class FakeVideoCapture:
    def __init__(self, frames, *, opened=True, delay=0.005):
        self.frames = [frame.copy() for frame in frames]
        self.opened = opened
        self.delay = delay
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        time.sleep(self.delay)
        if self.released or not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class LatestFrameCaptureThreadTest(unittest.TestCase):
    def test_capture_thread_resizes_and_publishes(self):
        frame = np.zeros((30, 60, 3), dtype=np.uint8)

        def factory(_url):
            return FakeVideoCapture([frame, frame + 1, frame + 2])

        capture = LatestFrameCapture(
            "bed_test",
            "fake://stream",
            frame_width=30,
            reconnect_delay_sec=0.01,
            capture_factory=factory,
        )
        capture.start()
        packet = capture.wait_for_frame(2, timeout=1.0)
        capture.stop()

        self.assertIsNotNone(packet)
        self.assertGreaterEqual(packet.frame_seq, 3)
        self.assertEqual(packet.frame.shape[:2], (15, 30))
        self.assertFalse(capture.is_alive())

    def test_capture_thread_recovers_after_open_failure(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        attempts = {"count": 0}

        def factory(_url):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return FakeVideoCapture([], opened=False)
            return FakeVideoCapture([frame])

        capture = LatestFrameCapture(
            "bed_test",
            "fake://stream",
            reconnect_delay_sec=0.01,
            capture_factory=factory,
        )
        capture.start()
        packet = capture.wait_for_frame(0, timeout=1.0)
        metrics = capture.metrics()
        capture.stop()

        self.assertIsNotNone(packet)
        self.assertGreaterEqual(attempts["count"], 2)
        self.assertGreaterEqual(metrics["reconnect_total"], 1)


if __name__ == "__main__":
    unittest.main()

