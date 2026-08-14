import json
from pathlib import Path
import tempfile
import time
import unittest

from shadow_feature_recorder import ShadowFeatureRecorder


class ShadowFeatureRecorderTests(unittest.TestCase):
    def test_throttles_but_records_phase_change(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ShadowFeatureRecorder(
                Path(directory), sample_interval_sec=1.0, flush_interval_sec=0.1
            )
            recorder.start()
            self.assertTrue(recorder.submit("bed_1", {"fusion_phase": "SAFE"}, mono_ts=1.0))
            self.assertFalse(recorder.submit("bed_1", {"fusion_phase": "SAFE"}, mono_ts=1.1))
            self.assertTrue(
                recorder.submit("bed_1", {"fusion_phase": "CANDIDATE"}, mono_ts=1.2)
            )
            recorder.stop()
            rows = [
                json.loads(line)
                for path in Path(directory).glob("*.jsonl")
                for line in path.read_text().splitlines()
            ]
            self.assertEqual([row["fusion_phase"] for row in rows], ["SAFE", "CANDIDATE"])

    def test_whitelist_excludes_sensitive_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ShadowFeatureRecorder(Path(directory))
            recorder.start()
            recorder.submit(
                "bed_1",
                {
                    "fusion_phase": "SAFE",
                    "frame": "sensitive",
                    "keypoints": [1, 2, 3],
                    "rtsp_url": "secret",
                },
                mono_ts=1.0,
            )
            recorder.stop()
            path = next(Path(directory).glob("*.jsonl"))
            row = json.loads(path.read_text().strip())
            self.assertNotIn("frame", row)
            self.assertNotIn("keypoints", row)
            self.assertNotIn("rtsp_url", row)

    def test_status_reports_writer_health(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = ShadowFeatureRecorder(Path(directory))
            recorder.start()
            recorder.submit("bed_1", {"fusion_phase": "SAFE"}, mono_ts=time.monotonic())
            deadline = time.monotonic() + 1.0
            while recorder.status()["written_total"] < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            status = recorder.status()
            self.assertTrue(status["thread_alive"])
            self.assertEqual(status["dropped_total"], 0)
            self.assertEqual(status["error_total"], 0)
            recorder.stop()
