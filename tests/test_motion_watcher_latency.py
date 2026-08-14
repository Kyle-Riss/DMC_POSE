import threading
import time
import unittest

import numpy as np

from latest_frame_capture import LatestFrameCapture
from motion_watcher import MotionWatcher


class MotionWatcherLatencyTest(unittest.TestCase):
    def test_burst_waiter_wakes_within_300ms(self):
        watcher = MotionWatcher(
            "test",
            LatestFrameCapture("test", "unused"),
            small_width=64,
            small_height=36,
            pixel_threshold=10,
            motion_ratio_threshold=0.02,
            consecutive_hits=2,
        )
        result = {}

        def wait():
            started = time.perf_counter()
            result["active"] = watcher.wait_for_burst(1.0)
            result["elapsed"] = time.perf_counter() - started

        thread = threading.Thread(target=wait)
        thread.start()
        time.sleep(0.02)
        timestamp = time.monotonic()
        for seq, x in enumerate((0, 10, 20), start=1):
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            frame[20:60, x:x + 40] = 255
            watcher.process_frame(
                frame, frame_seq=seq, mono_ts=timestamp + seq * 0.05
            )
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertTrue(result["active"])
        self.assertLess(result["elapsed"], 0.3)


if __name__ == "__main__":
    unittest.main()
