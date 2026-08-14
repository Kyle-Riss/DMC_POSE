import unittest

from runtime_health import (
    DEGRADED,
    HEALTHY,
    OFFLINE,
    evaluate_camera,
    evaluate_fleet,
    render_prometheus,
)


def healthy_state(camera_id="bed_161"):
    return {
        "camera_id": camera_id,
        "capture_connected": True,
        "capture_fps": 20.0,
        "capture_frame_age_ms": 50.0,
        "analysis_frame_age_ms": 100.0,
        "watcher_thread_alive": True,
        "watcher_fps": 20.0,
        "scheduler_thread_alive": True,
        "scheduler_pending": 0,
        "scheduler_queue_latency_ms": 5.0,
        "scheduler_error_total": 0,
        "bed_roi_ready": True,
    }


class RuntimeHealthTests(unittest.TestCase):
    def test_healthy_camera(self):
        result = evaluate_camera(healthy_state())
        self.assertEqual(result["status"], HEALTHY)
        self.assertTrue(result["ready"])

    def test_disconnected_camera_is_offline(self):
        state = healthy_state()
        state["capture_connected"] = False
        result = evaluate_camera(state)
        self.assertEqual(result["status"], OFFLINE)
        self.assertIn("capture_disconnected", result["critical"])

    def test_stale_analysis_is_degraded_not_offline(self):
        state = healthy_state()
        state["analysis_frame_age_ms"] = 3000.0
        result = evaluate_camera(state)
        self.assertEqual(result["status"], DEGRADED)
        self.assertIn("analysis_result_stale", result["warnings"])

    def test_one_offline_camera_marks_fleet_offline(self):
        good = healthy_state("good")
        bad = healthy_state("bad")
        bad["watcher_thread_alive"] = False
        result = evaluate_fleet({"good": good, "bad": bad})
        self.assertEqual(result["status"], OFFLINE)
        self.assertEqual(result["counts"][OFFLINE], 1)

    def test_prometheus_does_not_expose_rtsp(self):
        state = healthy_state()
        state["rtsp_url"] = "rtsp://user:secret@example/stream"
        output = render_prometheus({"bed_161": state}, process_ready=True)
        self.assertIn("dmc_pose_capture_fps", output)
        self.assertNotIn("rtsp", output)
        self.assertNotIn("secret", output)


if __name__ == "__main__":
    unittest.main()
