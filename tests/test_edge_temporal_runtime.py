import json

import numpy as np

from edge_temporal_runtime import EdgeTemporalRuntime
from live_temporal import TemporalShadowRunner
from temporal_candidate_recorder import TemporalCandidateRecorder


class PositiveService:
    threshold = 0.5

    def predict(self, window):
        assert np.asarray(window).shape == (30, 109)
        return 0.9


def pose():
    xy = np.arange(34, dtype=np.float32).reshape(17, 2) + 10
    conf = np.ones(17, dtype=np.float32)
    probs = np.full(6, 1 / 6, dtype=np.float32)
    return xy, conf, probs


def push(runtime, index, track_id=1):
    xy, conf, probs = pose()
    return runtime.push(
        timestamp=index / 10, captured_at="2026-08-07T12:00:00+09:00",
        frame_seq=index, track_id=track_id, keypoints_xy=xy,
        keypoints_conf=conf, pose_probs=probs, roi_version=1,
        fusion_risk=0.7, fusion_phase="CANDIDATE", quality=0.9,
        evidence=["rapid_descent"],
    )


def runtime(tmp_path):
    recorder = TemporalCandidateRecorder(tmp_path, cooldown_sec=5)
    recorder.start()
    temporal = EdgeTemporalRuntime(
        TemporalShadowRunner(PositiveService()), recorder,
        node_id="rpi-bed-161", camera_id="bed_161", boot_id="boot-a",
        model_bundle_version="bundle-v1",
    )
    return temporal, recorder


def test_candidate_is_automatically_recorded_after_real_30_row_window(tmp_path):
    temporal, recorder = runtime(tmp_path)
    try:
        status = None
        for index in range(35):
            status = push(temporal, index)
        assert status["candidate"]
        assert status["candidate_recorded"]
    finally:
        recorder.stop()
    record = json.loads((tmp_path / "candidates.jsonl").read_text().splitlines()[0])
    assert record["shape"] == [30, 109]
    assert len(record["sample_timestamps"]) == 30


def test_track_change_prevents_cross_person_window(tmp_path):
    temporal, recorder = runtime(tmp_path)
    try:
        for index in range(20):
            push(temporal, index, track_id=1)
        status = push(temporal, 20, track_id=2)
        assert status["samples"] == 1
        assert not status["ready"]
        assert len(temporal.sample_timestamps) == 1
    finally:
        recorder.stop()


def test_gap_prevents_cross_gap_window(tmp_path):
    temporal, recorder = runtime(tmp_path)
    try:
        for index in range(20):
            push(temporal, index)
        status = push(temporal, 30)
        assert status["samples"] == 1
        assert status["gap_reset_total"] == 1
        assert len(temporal.sample_timestamps) == 1
    finally:
        recorder.stop()
