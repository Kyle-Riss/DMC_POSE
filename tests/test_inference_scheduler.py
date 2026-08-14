import threading
import time
import unittest

from inference_scheduler import (
    LatestInferenceScheduler,
    P0_VERIFY,
    P1_BURST,
    P3_EMPTY_PROBE,
    P4_BED_SEG,
)


class InferenceSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.order = []
        self.started = threading.Event()
        self.release = threading.Event()

        def infer(frame):
            self.order.append(frame)
            if frame == "block":
                self.started.set()
                self.release.wait(2.0)
            time.sleep(0.005)
            return [frame]

        def infer_replay(frame):
            self.order.append(f"replay:{frame}")
            return [frame]

        self.scheduler = LatestInferenceScheduler(
            infer_pose=infer,
            infer_seg=infer,
            infer_pose_replay=infer_replay,
            urgent_quota=4,
            metrics_window_sec=5.0,
        )
        self.scheduler.start()

    def tearDown(self):
        self.release.set()
        self.scheduler.stop()

    def submit_thread(self, model, camera, frame, priority, deadline=2.0):
        result = {}

        def run():
            fn = self.scheduler.request_pose if model == "pose" else self.scheduler.request_seg
            result["outcome"] = fn(
                camera, frame, frame_seq=1,
                priority=priority, deadline_sec=deadline,
            )

        thread = threading.Thread(target=run)
        thread.start()
        return thread, result

    def occupy_worker(self):
        thread, result = self.submit_thread(
            "seg", "bed_block", "block", P4_BED_SEG
        )
        self.assertTrue(self.started.wait(0.5))
        return thread, result

    def test_verify_priority_runs_before_empty_probe(self):
        blocker, _ = self.occupy_worker()
        normal, normal_result = self.submit_thread(
            "pose", "bed_normal", "normal", P3_EMPTY_PROBE
        )
        urgent, urgent_result = self.submit_thread(
            "pose", "bed_urgent", "urgent", P0_VERIFY
        )
        time.sleep(0.03)
        self.release.set()
        for thread in (blocker, normal, urgent):
            thread.join(1.0)
        self.assertEqual(self.order[:3], ["block", "urgent", "normal"])
        self.assertTrue(urgent_result["outcome"].completed)
        self.assertTrue(normal_result["outcome"].completed)

    def test_latest_request_supersedes_same_mailbox(self):
        blocker, _ = self.occupy_worker()
        old_thread, old_result = self.submit_thread(
            "pose", "bed_161", "old", P1_BURST
        )
        time.sleep(0.02)
        new_thread, new_result = self.submit_thread(
            "pose", "bed_161", "new", P1_BURST
        )
        time.sleep(0.02)
        self.release.set()
        for thread in (blocker, old_thread, new_thread):
            thread.join(1.0)
        self.assertTrue(old_result["outcome"].dropped)
        self.assertEqual(old_result["outcome"].drop_reason, "superseded")
        self.assertTrue(new_result["outcome"].completed)
        self.assertNotIn("old", self.order)
        self.assertIn("new", self.order)
        self.assertEqual(
            self.scheduler.metrics("bed_161")["superseded_drop_total"], 1
        )

    def test_stale_request_is_dropped_without_inference(self):
        blocker, _ = self.occupy_worker()
        stale_thread, stale_result = self.submit_thread(
            "pose", "bed_stale", "stale", P3_EMPTY_PROBE, deadline=0.05
        )
        time.sleep(0.10)
        self.release.set()
        blocker.join(1.0)
        stale_thread.join(1.0)
        self.assertTrue(stale_result["outcome"].dropped)
        self.assertEqual(stale_result["outcome"].drop_reason, "stale")
        self.assertNotIn("stale", self.order)
        self.assertEqual(
            self.scheduler.metrics("bed_stale")["stale_drop_total"], 1
        )

    def test_urgent_quota_prevents_normal_starvation(self):
        blocker, _ = self.occupy_worker()
        threads = []
        normal, _ = self.submit_thread(
            "pose", "bed_normal", "normal", P3_EMPTY_PROBE
        )
        threads.append(normal)
        for idx in range(5):
            thread, _ = self.submit_thread(
                "pose", f"urgent_{idx}", f"urgent_{idx}", P0_VERIFY
            )
            threads.append(thread)
        time.sleep(0.04)
        self.release.set()
        blocker.join(1.0)
        for thread in threads:
            thread.join(2.0)
        after_block = self.order[1:]
        self.assertLess(after_block.index("normal"), after_block.index("urgent_4"))
        self.assertEqual(after_block.index("normal"), 4)

    def test_metrics_report_live_completed_rate(self):
        outcome = self.scheduler.request_pose(
            "bed_161", "one", frame_seq=1,
            priority=P3_EMPTY_PROBE, deadline_sec=1.0,
        )
        self.assertTrue(outcome.completed)
        metrics = self.scheduler.metrics("bed_161")
        self.assertEqual(metrics["completed_total"], 1)
        self.assertGreater(metrics["completed_hz"], 0.0)
        self.assertTrue(metrics["thread_alive"])

    def test_replay_uses_dedicated_inference_path(self):
        outcome = self.scheduler.request_pose_replay(
            "bed_161", "history", frame_seq=1,
            priority=P1_BURST, deadline_sec=1.0,
        )
        self.assertTrue(outcome.completed)
        self.assertIn("replay:history", self.order)


if __name__ == "__main__":
    unittest.main()
