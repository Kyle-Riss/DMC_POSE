import unittest

from summarize_shadow_features import summarize_rows


class ShadowSummaryTests(unittest.TestCase):
    def test_counts_bed_hours_and_merges_alert_rows(self):
        rows = [
            {
                "camera_id": "bed_1",
                "recorded_at": "2026-07-31T00:00:00Z",
                "fusion_phase": "SAFE",
                "primary_track_id": 1,
            },
            {
                "camera_id": "bed_1",
                "recorded_at": "2026-07-31T00:00:01Z",
                "fusion_phase": "SHADOW_ALERT",
                "fusion_risk": 0.7,
                "fusion_track_id": 1,
                "fusion_evidence": ["rapid_motion"],
                "primary_track_id": 1,
            },
            {
                "camera_id": "bed_1",
                "recorded_at": "2026-07-31T00:00:02Z",
                "fusion_phase": "SHADOW_ALERT",
                "fusion_risk": 0.9,
                "fusion_track_id": 1,
                "fusion_evidence": ["tcn_persistent"],
                "primary_track_id": 1,
            },
            {
                "camera_id": "bed_1",
                "recorded_at": "2026-07-31T00:00:03Z",
                "fusion_phase": "SAFE",
                "primary_track_id": 1,
            },
        ]
        report = summarize_rows(rows)
        camera = report["cameras"]["bed_1"]
        self.assertEqual(camera["shadow_alert_events"], 1)
        self.assertEqual(
            camera["review_candidates"][0]["evidence"],
            ["rapid_motion", "tcn_persistent"],
        )
        self.assertEqual(camera["review_candidates"][0]["peak_risk"], 0.9)
        self.assertGreater(camera["recorded_bed_hours"], 0)
        self.assertAlmostEqual(
            camera["policy_bed_hours"]["legacy_unknown"],
            camera["recorded_bed_hours"], places=6,
        )

    def test_long_gaps_are_not_counted_as_recording_time(self):
        rows = [
            {
                "camera_id": "bed_1",
                "recorded_at": "2026-07-31T00:00:00Z",
                "fusion_phase": "SAFE",
            },
            {
                "camera_id": "bed_1",
                "recorded_at": "2026-07-31T01:00:00Z",
                "fusion_phase": "SAFE",
            },
        ]
        report = summarize_rows(rows, max_sample_gap_sec=2.0)
        self.assertAlmostEqual(
            report["cameras"]["bed_1"]["recorded_bed_hours"],
            2.0 / 3600.0,
            places=6,
        )
