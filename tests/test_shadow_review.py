import unittest

from shadow_review import candidate_id, evaluate_operations, prepare_review_rows


class ShadowReviewTests(unittest.TestCase):
    def test_candidate_id_does_not_change_when_end_grows(self):
        first = {"camera_id": "bed_1", "started_at": "2026-01-01T00:00:00Z", "ended_at": "a"}
        later = dict(first, ended_at="b")
        self.assertEqual(candidate_id(first), candidate_id(later))

    def test_review_labels_survive_queue_refresh(self):
        candidate = {
            "camera_id": "bed_1",
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:00:01Z",
            "peak_risk": 0.9,
            "evidence": ["motion"],
            "track_ids": [7],
        }
        item_id = candidate_id(candidate)
        existing = [{
            "candidate_id": item_id,
            "label": "false_alarm",
            "reviewer": "operator",
            "note": "blanket movement",
        }]
        rows = prepare_review_rows([dict(candidate, ended_at="2026-01-01T00:00:04Z")], existing)
        self.assertEqual(rows[0]["label"], "false_alarm")
        self.assertEqual(rows[0]["reviewer"], "operator")
        self.assertEqual(rows[0]["ended_at"], "2026-01-01T00:00:04Z")

    def test_metrics_separate_real_and_staged_falls(self):
        summary = {
            "total_bed_hours": 200.0,
            "cameras": {"bed_1": {"recorded_bed_hours": 200.0}},
        }
        review = [
            {"candidate_id": "a", "camera_id": "bed_1", "label": "true_fall"},
            {"candidate_id": "b", "camera_id": "bed_1", "label": "false_alarm"},
            {"candidate_id": "c", "camera_id": "bed_1", "label": "staged_fall"},
        ]
        events = [
            {"event_id": "e1", "camera_id": "bed_1", "event_type": "actual_fall",
             "matched_candidate_id": "a"},
            {"event_id": "e2", "camera_id": "bed_1", "event_type": "actual_fall",
             "matched_candidate_id": ""},
            {"event_id": "s1", "camera_id": "bed_1", "event_type": "staged_fall",
             "matched_candidate_id": "c"},
        ]
        report = evaluate_operations(summary, review, events)
        overall = report["overall"]
        self.assertEqual(overall["alert_precision_real_only"], 0.5)
        self.assertEqual(overall["false_alarms_per_bed_hour"], 0.005)
        self.assertEqual(overall["sensitivity_actual_falls"], 0.5)
        self.assertEqual(overall["sensitivity_staged_falls"], 1.0)
        self.assertEqual(report["readiness"], "FAIL")

    def test_pending_review_blocks_false_alarm_gate(self):
        summary = {
            "total_bed_hours": 500.0,
            "cameras": {"bed_1": {"recorded_bed_hours": 500.0}},
        }
        report = evaluate_operations(
            summary,
            [{"candidate_id": "a", "camera_id": "bed_1", "label": "pending"}],
            [],
        )
        self.assertEqual(report["overall"]["false_alarm_gate"], "NOT_READY")
        self.assertIsNone(report["overall"]["false_alarms_per_bed_hour"])
        self.assertEqual(
            report["overall"]["confirmed_false_alarms_per_bed_hour_lower_bound"], 0.0
        )
        self.assertEqual(report["overall"]["detection_gate"], "NOT_MEASURED")
        self.assertEqual(report["readiness"], "NOT_READY")

    def test_policy_filter_separates_candidates_and_bed_hours(self):
        summary = {
            "total_bed_hours": 300.0,
            "policy_bed_hours": {"legacy_unknown": 100.0, "v2": 200.0},
            "cameras": {"bed_1": {
                "recorded_bed_hours": 300.0,
                "policy_bed_hours": {"legacy_unknown": 100.0, "v2": 200.0},
            }},
        }
        review = [
            {"candidate_id": "old", "camera_id": "bed_1",
             "policy_versions": "legacy_unknown", "label": "false_alarm"},
            {"candidate_id": "new", "camera_id": "bed_1",
             "policy_versions": "v2", "label": "true_fall"},
        ]
        events = [{"event_id": "event", "camera_id": "bed_1",
                   "event_type": "actual_fall", "policy_version": "v2",
                   "matched_candidate_id": "new"}]
        report = evaluate_operations(summary, review, events, policy_version="v2")
        self.assertEqual(report["policy_version"], "v2")
        self.assertEqual(report["overall"]["bed_hours"], 200.0)
        self.assertEqual(report["overall"]["candidate_count"], 1)
        self.assertEqual(report["overall"]["false_alarms_per_bed_hour"], 0.0)
        self.assertEqual(report["overall"]["sensitivity_actual_falls"], 1.0)
        self.assertEqual(report["readiness"], "PASS")


if __name__ == "__main__":
    unittest.main()
