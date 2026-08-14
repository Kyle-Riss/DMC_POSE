import unittest

import numpy as np

from evaluate_temporal_events import (
    contiguous_events,
    event_has_persistence_capacity,
    event_is_pre_onset_ready,
    poisson_rate_ci,
)


class EventMergingTests(unittest.TestCase):
    def test_short_probability_gap_is_one_event(self):
        rows = [{"end_sec": float(i)} for i in range(7)]
        probabilities = np.asarray([0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1])
        events = contiguous_events(rows, probabilities, 0.5, 2, merge_gap_sec=3.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_sec"], 0.0)
        self.assertEqual(events[0]["end_sec"], 4.0)

    def test_distant_events_remain_separate(self):
        rows = [{"end_sec": float(i)} for i in range(10)]
        probabilities = np.asarray([0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0])
        events = contiguous_events(rows, probabilities, 0.5, 2, merge_gap_sec=2.0)
        self.assertEqual(len(events), 2)

    def test_persistence_and_merge_never_cross_sequence_boundary(self):
        rows = [
            {"end_sec": 0.0, "sequence_id": 1, "track_id": 1},
            {"end_sec": 0.5, "sequence_id": 1, "track_id": 1},
            {"end_sec": 0.6, "sequence_id": 2, "track_id": 1},
            {"end_sec": 1.1, "sequence_id": 2, "track_id": 1},
        ]
        probabilities = np.asarray([0.9, 0.9, 0.9, 0.9])
        events = contiguous_events(rows, probabilities, 0.5, 2, merge_gap_sec=3.0)
        self.assertEqual(len(events), 2)
        self.assertEqual([event["sequence_id"] for event in events], [1, 2])

    def test_pre_onset_ready_requires_live_sequence_at_onset(self):
        event = {"start_sec": 4.0, "end_sec": 5.0}
        live = [{"track_id": 1, "sequence_id": 2, "sequence_ready_sec": 3.0, "sequence_observation_end_sec": 4.05}]
        stale = [{"track_id": 1, "sequence_id": 2, "sequence_ready_sec": 3.0, "sequence_observation_end_sec": 3.8}]
        self.assertTrue(event_is_pre_onset_ready(event, live))
        self.assertFalse(event_is_pre_onset_ready(event, stale))

    def test_persistence_capacity_never_crosses_sequence_boundary(self):
        event = {"start_sec": 1.0, "end_sec": 2.0}
        rows = [
            {"end_sec": 1.2, "track_id": 1, "sequence_id": 1},
            {"end_sec": 1.7, "track_id": 1, "sequence_id": 2},
        ]
        self.assertTrue(event_has_persistence_capacity(event, rows, 1))
        self.assertFalse(event_has_persistence_capacity(event, rows, 2))
        rows.append({"end_sec": 1.9, "track_id": 1, "sequence_id": 2})
        self.assertTrue(event_has_persistence_capacity(event, rows, 2))

    def test_poisson_interval_contains_observed_rate(self):
        interval = poisson_rate_ci(8, 0.080295)
        self.assertIsNotNone(interval)
        self.assertLess(interval[0], 8 / 0.080295)
        self.assertGreater(interval[1], 8 / 0.080295)


if __name__ == "__main__":
    unittest.main()
