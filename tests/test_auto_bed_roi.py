import tempfile
import unittest
from pathlib import Path

import numpy as np

from auto_bed_roi import AutoBedROIManager, bbox_iou


def bed_detection(x1=10, y1=8, x2=40, y2=24, confidence=0.9):
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[y1:y2 + 1, x1:x2 + 1] = 1
    return {
        "mask": mask,
        "bbox": (x1, y1, x2, y2),
        "confidence": confidence,
        "source": "seg_mask",
    }


class AutoBedROIManagerTest(unittest.TestCase):
    def test_bbox_iou(self):
        self.assertEqual(bbox_iou((0, 0, 9, 9), (0, 0, 9, 9)), 1.0)
        self.assertEqual(bbox_iou((0, 0, 4, 4), (10, 10, 14, 14)), 0.0)

    def test_three_consistent_detections_create_automatic_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = AutoBedROIManager(
                "bed_test", Path(directory), sample_every_n=1,
                min_detections=3, candidate_window=5, consensus_iou=0.7,
            )
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            detections = [
                bed_detection(10, 8, 40, 24),
                bed_detection(11, 8, 41, 24),
                bed_detection(10, 9, 40, 25),
            ]
            for index, detection in enumerate(detections):
                manager.mark_segmentation_attempt(float(index))
                ready = manager.observe_detection(
                    detection, frame, mono_ts=float(index), wall_ts=100.0 + index
                )

            status = manager.status()
            self.assertTrue(ready)
            self.assertTrue(status["ready"])
            self.assertEqual(status["version"], 1)
            self.assertEqual(status["source"], "auto_consensus")
            self.assertEqual(status["roi_state"], "READY")
            self.assertFalse(status["restored_from_cache"])
            self.assertTrue((Path(directory) / "bed_test.json").is_file())
            self.assertTrue((Path(directory) / "bed_test.mask.png").is_file())
            self.assertNotIn("manual", status["source"])

    def test_automatic_cache_is_loaded_without_manual_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            manager = AutoBedROIManager(
                "bed_test", cache_dir, sample_every_n=1,
                min_detections=3, consensus_iou=0.7,
            )
            for index in range(3):
                manager.observe_detection(
                    bed_detection(), frame, mono_ts=float(index), wall_ts=100.0 + index
                )

            restored = AutoBedROIManager("bed_test", cache_dir)
            self.assertTrue(restored.status()["ready"])
            self.assertEqual(restored.status()["source"], "auto_cache")
            self.assertEqual(restored.status()["roi_state"], "READY")
            self.assertTrue(restored.status()["restored_from_cache"])
            self.assertEqual(restored.current()["bbox"], (10, 8, 40, 24))

    def test_disagreeing_detections_remain_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = AutoBedROIManager(
                "bed_test", Path(directory), sample_every_n=1,
                min_detections=3, candidate_window=3, consensus_iou=0.8,
            )
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            detections = [
                bed_detection(0, 0, 15, 10),
                bed_detection(20, 5, 35, 15),
                bed_detection(40, 20, 58, 38),
            ]
            for index, detection in enumerate(detections):
                manager.observe_detection(
                    detection, frame, mono_ts=float(index), wall_ts=100.0 + index
                )
            self.assertFalse(manager.status()["ready"])
            self.assertIsNone(manager.current()["bbox"])
            self.assertEqual(manager.current()["source"], "auto_not_ready")
            self.assertEqual(manager.status()["invalid_reason"], "consensus_not_reached")
            self.assertEqual(manager.status()["roi_state"], "DEGRADED")

    def test_stable_roi_stops_segmentation_until_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = AutoBedROIManager(
                "bed_test", Path(directory), sample_every_n=1,
                min_detections=3, consensus_iou=0.7, refresh_sec=30.0,
            )
            frame = np.zeros((40, 60, 3), dtype=np.uint8)
            for index in range(3):
                manager.observe_detection(
                    bed_detection(), frame, mono_ts=float(index), wall_ts=100.0 + index
                )
            self.assertFalse(manager.should_run_segmentation(10, 20.0))
            self.assertTrue(manager.should_run_segmentation(10, 32.1))

    def test_persistent_scene_change_invalidates_auto_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = AutoBedROIManager(
                "bed_test", Path(directory), sample_every_n=1,
                min_detections=3, consensus_iou=0.7,
                scene_check_sec=0.1, scene_change_ratio=0.7,
                scene_change_persistence=3,
            )
            dark = np.zeros((40, 60, 3), dtype=np.uint8)
            bright = np.full((40, 60, 3), 255, dtype=np.uint8)
            for index in range(3):
                manager.observe_detection(
                    bed_detection(), dark, mono_ts=float(index), wall_ts=100.0 + index
                )
            self.assertTrue(manager.status()["ready"])
            self.assertFalse(manager.observe_frame(bright, 3.0))
            self.assertFalse(manager.observe_frame(bright, 3.2))
            self.assertTrue(manager.observe_frame(bright, 3.4))
            self.assertFalse(manager.status()["ready"])
            self.assertEqual(manager.status()["invalid_reason"], "scene_changed")


if __name__ == "__main__":
    unittest.main()

