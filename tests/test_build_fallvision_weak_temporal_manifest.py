import unittest

from scripts.build_fallvision_weak_temporal_manifest import build_item, build_non_fall_item


class FallVisionWeakManifestTest(unittest.TestCase):
    def test_transition_is_ignored_and_split_is_train_only(self):
        source = {
            "video_id": "fallvision:fall:bed:1:B_D_1", "scene_id": "bed", "chunk_id": "1",
            "local_video_path": "/tmp/v.mp4", "fps": "10", "frame_count": "50",
            "duration_sec": "5", "width": "640", "height": "480", "media_sha256": "abc",
        }
        proposal = {
            "proposed_fall_onset_frame": "10", "proposed_impact_frame": "20",
            "proposed_fall_end_frame": "45", "proposed_onset_earliest_frame": "8",
            "proposal_status": "review_required",
        }
        item = build_item(source, proposal)
        self.assertEqual(item["split"], "train")
        self.assertFalse(item["evaluation_eligible"])
        self.assertEqual(item["intervals"][0], {
            "source_label": "auto_boundary_uncertain_transition", "label": "ignore",
            "start_sec": 0.8, "end_sec": 1.9,
        })
        self.assertEqual(item["intervals"][1]["start_sec"], 2.0)

    def test_non_fall_uses_official_video_label_without_intervals(self):
        source = {
            "video_id": "fallvision:non_fall:bed:1:B_N_1", "scene_id": "bed", "chunk_id": "1",
            "local_video_path": "/tmp/v.mp4", "fps": "10", "frame_count": "50",
            "duration_sec": "5", "width": "640", "height": "480", "media_sha256": "abc",
        }
        item = build_non_fall_item(source)
        self.assertEqual(item["binary_fall_label"], 0)
        self.assertEqual(item["intervals"], [])
        self.assertFalse(item["evaluation_eligible"])


if __name__ == "__main__":
    unittest.main()
