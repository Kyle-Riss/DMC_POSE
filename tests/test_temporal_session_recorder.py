import json

import numpy as np

from temporal_session_recorder import (
    TemporalEventSessionRecorder,
    derive_temporal_session_triggers,
)


def feature(value: float) -> np.ndarray:
    return np.full(109, value, dtype=np.float32)


def test_no_trigger_writes_nothing(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path, pre_roll_sec=2, post_roll_sec=2)
    recorder.start()
    recorder.observe("bed_161", 1.0, feature(1), track_id=1, quality=0.8)
    recorder.stop()
    assert list(tmp_path.glob("*/manifest.json")) == []


def test_trigger_writes_pre_and_post_roll_without_images(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path, pre_roll_sec=2, post_roll_sec=2)
    recorder.start()
    recorder.observe("bed_161", 1.0, feature(1), track_id=7, quality=0.8)
    recorder.observe("bed_161", 2.0, feature(2), track_id=7, quality=0.9)
    recorder.observe(
        "bed_161", 3.0, feature(3), track_id=7, quality=1.0,
        triggers={"PERSON_ENTER"},
    )
    recorder.observe("bed_161", 4.0, feature(4), track_id=7, quality=0.9)
    recorder.tick("bed_161", 5.1)
    recorder.stop()

    manifest_path = next(tmp_path.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    arrays = np.load(manifest_path.parent / "features.npz")
    assert arrays["features"].shape == (4, 109)
    assert arrays["track_ids"].tolist() == [7, 7, 7, 7]
    assert manifest["label"] == "UNREVIEWED"
    assert manifest["trigger_counts"] == {"PERSON_ENTER": 1}
    assert manifest["contains_video"] is False
    assert manifest["contains_raw_keypoints"] is False
    assert manifest["trigger_contexts"][0]["name"] == "PERSON_ENTER"
    # One-second gaps are not a valid 10 Hz context and must reset continuity.
    assert manifest["trigger_contexts"][0]["contiguous_observed_samples"] == 1
    assert manifest["trigger_contexts"][0]["coverage_to_trigger_sec"] == 0.0
    assert manifest["tcn_context_ready"] is False
    assert manifest["long_pre_context_ready"] is False
    assert "capture_ts" not in manifest["triggers"][0]


def test_multiple_triggers_extend_one_session(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path, pre_roll_sec=1, post_roll_sec=2)
    recorder.start()
    recorder.observe(
        "bed_161", 10.0, feature(1), track_id=1, quality=1,
        triggers={"PERSON_ENTER"},
    )
    recorder.observe(
        "bed_161", 11.5, feature(2), track_id=1, quality=1,
        triggers={"TCN_CANDIDATE"},
    )
    recorder.tick("bed_161", 13.6)
    recorder.stop()
    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["trigger_counts"] == {
        "PERSON_ENTER": 1,
        "TCN_CANDIDATE": 1,
    }


def test_invalid_feature_is_rejected(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path)
    recorder.start()
    assert not recorder.observe(
        "bed_161", 1.0, np.zeros(108), track_id=1, quality=1,
        triggers={"PERSON_ENTER"},
    )
    recorder.stop()
    assert recorder.status()["invalid_total"] == 1


def test_trigger_without_pose_is_counted_but_does_not_write_empty_npz(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path, pre_roll_sec=2, post_roll_sec=1)
    recorder.start()
    recorder.tick("bed_161", 10.0, triggers={"EDGE_WAKE_RISE"})
    recorder.tick("bed_161", 11.1)
    recorder.stop()
    status = recorder.status()
    assert status["completed_total"] == 1
    assert status["written_total"] == 0
    assert status["skipped_empty_total"] == 1
    assert list(tmp_path.glob("*/manifest.json")) == []


def test_idle_tick_expires_stale_pre_roll(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path, pre_roll_sec=2, post_roll_sec=1)
    recorder.start()
    recorder.observe("bed_161", 1.0, feature(1), track_id=1, quality=1)
    recorder.tick("bed_161", 10.0, triggers={"EDGE_WAKE_RISE"})
    recorder.observe("bed_161", 10.2, feature(2), track_id=2, quality=1)
    recorder.tick("bed_161", 11.1)
    recorder.stop()
    manifest_path = next(tmp_path.glob("*/manifest.json"))
    arrays = np.load(manifest_path.parent / "features.npz")
    assert arrays["features"].shape == (1, 109)
    assert arrays["track_ids"].tolist() == [2]


def test_tcn_only_trigger_cannot_reopen_during_rearm(tmp_path):
    recorder = TemporalEventSessionRecorder(
        tmp_path, pre_roll_sec=1, post_roll_sec=1,
        model_trigger_rearm_sec=60,
    )
    recorder.start()
    recorder.observe(
        "bed_161", 1.0, feature(1), track_id=1, quality=1,
        triggers={"TCN_CANDIDATE_RISE"},
    )
    recorder.tick("bed_161", 2.1)
    recorder.observe(
        "bed_161", 3.0, feature(2), track_id=1, quality=1,
        triggers={"TCN_CANDIDATE_RISE"},
    )
    recorder.tick("bed_161", 4.1)
    recorder.stop()
    assert len(list(tmp_path.glob("*/manifest.json"))) == 1
    assert recorder.status()["suppressed_model_trigger_total"] == 1


def test_external_trigger_can_reopen_during_model_rearm(tmp_path):
    recorder = TemporalEventSessionRecorder(
        tmp_path, pre_roll_sec=0, post_roll_sec=1,
        model_trigger_rearm_sec=60,
    )
    recorder.start()
    recorder.observe(
        "bed_161", 1.0, feature(1), track_id=1, quality=1,
        triggers={"TCN_CANDIDATE_RISE"},
    )
    recorder.tick("bed_161", 2.1)
    recorder.observe(
        "bed_161", 3.0, feature(2), track_id=2, quality=1,
        triggers={"EDGE_WAKE_RISE"},
    )
    recorder.tick("bed_161", 4.1)
    recorder.stop()
    assert len(list(tmp_path.glob("*/manifest.json"))) == 2


def test_trigger_derivation_is_transition_only():
    previous = {
        "person_observed": False,
        "edge_wake": False,
        "local_motion": False,
        "tcn_candidate": False,
        "fusion_phase": "SAFE",
        "track_id": None,
    }
    current = {
        "person_observed": True,
        "edge_wake": True,
        "local_motion": True,
        "tcn_candidate": True,
        "fusion_phase": "CANDIDATE",
        "track_id": 7,
    }
    assert derive_temporal_session_triggers(previous, current) == {
        "PERSON_ENTER",
        "EDGE_WAKE_RISE",
        "LOCAL_MOTION_RISE",
        "TCN_CANDIDATE_RISE",
        "FUSION_CANDIDATE_RISE",
    }
    assert derive_temporal_session_triggers(current, current) == set()
    exited = dict(current, person_observed=False, track_id=None)
    assert derive_temporal_session_triggers(current, exited) == {"PERSON_EXIT"}


def test_trigger_context_reports_tcn_and_long_preroll_readiness(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path, pre_roll_sec=10, post_roll_sec=1)
    recorder.start()
    for index in range(101):
        recorder.observe(
            "bed_161",
            index / 10,
            feature(index),
            track_id=4,
            quality=0.9,
            triggers={"TCN_CANDIDATE_RISE"} if index == 100 else (),
        )
    recorder.tick("bed_161", 11.1)
    recorder.stop()

    manifest_path = next(tmp_path.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    context = manifest["best_pre_trigger_context"]
    assert context["contiguous_observed_samples"] == 101
    assert context["coverage_to_trigger_sec"] == 10.0
    assert context["tcn_context_ready"] is True
    assert context["long_pre_context_ready"] is True
    assert manifest["tcn_context_ready"] is True
    assert manifest["long_pre_context_ready"] is True
    status = recorder.status()
    assert status["tcn_context_ready_total"] == 1
    assert status["long_pre_context_ready_total"] == 1
    assert status["last_session_context"]["session_id"] == manifest["session_id"]


def test_trigger_context_resets_at_track_change(tmp_path):
    recorder = TemporalEventSessionRecorder(tmp_path, pre_roll_sec=10, post_roll_sec=1)
    recorder.start()
    for index in range(90):
        recorder.observe(
            "bed_161", index / 10, feature(index), track_id=1, quality=1
        )
    for index in range(90, 101):
        recorder.observe(
            "bed_161",
            index / 10,
            feature(index),
            track_id=2,
            quality=1,
            triggers={"FUSION_CANDIDATE_RISE"} if index == 100 else (),
        )
    recorder.tick("bed_161", 11.1)
    recorder.stop()

    manifest_path = next(tmp_path.glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    context = manifest["best_pre_trigger_context"]
    assert context["track_id"] == 2
    assert context["contiguous_observed_samples"] == 11
    assert context["tcn_context_ready"] is False
    assert context["long_pre_context_ready"] is False
