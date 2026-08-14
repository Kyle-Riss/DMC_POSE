import unittest

from phase10_soak import build_summary


def state(fps=20.0, age=50.0, reconnect=0):
    return {
        "camera_id": "bed_161",
        "capture_connected": True,
        "capture_fps": fps,
        "capture_frame_age_ms": age,
        "capture_decode_error_total": 0,
        "capture_reconnect_total": reconnect,
        "analysis_frame_age_ms": 100.0,
        "watcher_thread_alive": True,
        "watcher_fps": 20.0,
        "scheduler_thread_alive": True,
        "scheduler_pending": 0,
        "scheduler_queue_latency_ms": 5.0,
        "scheduler_error_total": 0,
        "scheduler_timeout_total": 0,
        "scheduler_stale_drop_total": 0,
        "bed_roi_ready": True,
    }


class Phase10SoakTests(unittest.TestCase):
    def test_summary_aggregates_latency_and_counter_delta(self):
        samples = [
            {
                "sampled_at": "start",
                "status": {"bed_161": state(age=10, reconnect=2)},
                "ready": {"ready": True},
                "errors": {},
            },
            {
                "sampled_at": "end",
                "status": {"bed_161": state(age=110, reconnect=4)},
                "ready": {"ready": True},
                "errors": {},
            },
        ]
        summary = build_summary(samples)
        camera = summary["cameras"]["bed_161"]
        self.assertEqual(summary["process_ready_ratio"], 1.0)
        self.assertEqual(camera["reconnects_delta"], 2)
        self.assertGreater(camera["capture_frame_age_ms_p95"], 100)

    def test_counter_reset_does_not_create_negative_delta(self):
        samples = [
            {"sampled_at": "a", "status": {"bed_161": state(reconnect=9)},
             "ready": {"ready": True}, "errors": {}},
            {"sampled_at": "b", "status": {"bed_161": state(reconnect=1)},
             "ready": {"ready": True}, "errors": {}},
        ]
        self.assertEqual(
            build_summary(samples)["cameras"]["bed_161"]["reconnects_delta"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
