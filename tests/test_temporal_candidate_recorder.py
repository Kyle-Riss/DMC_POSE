import json

import numpy as np
import pytest

from temporal_candidate_recorder import TemporalCandidateRecorder


def metadata(frame_seq=30):
    return {
        "node_id": "rpi-bed-161",
        "camera_id": "bed_161",
        "boot_id": "boot-a",
        "frame_seq": frame_seq,
        "captured_at": "2026-08-07T12:00:00+09:00",
        "track_id": 1,
        "trigger": "temporal_candidate",
        "temporal_probability": 0.8,
        "fusion_risk": 0.7,
        "evidence": ["rapid_descent"],
        "sample_timestamps": [index / 10 for index in range(30)],
    }


def test_records_exact_feature_window_and_index(tmp_path):
    recorder = TemporalCandidateRecorder(tmp_path, cooldown_sec=0)
    recorder.start()
    receipt = recorder.submit(np.ones((30, 109), dtype=np.float32), metadata())
    assert receipt.accepted
    recorder.stop()
    rows = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert rows[0]["shape"] == [30, 109]
    artifact = np.load(tmp_path / rows[0]["filename"])
    assert artifact["window"].shape == (30, 109)
    assert recorder.status()["errors"] == 0


def test_rejects_bad_shape_nonfinite_and_secret_metadata(tmp_path):
    recorder = TemporalCandidateRecorder(tmp_path)
    with pytest.raises(ValueError):
        recorder.submit(np.ones((29, 109)), metadata())
    bad = np.ones((30, 109))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        recorder.submit(bad, metadata())
    unsafe = metadata()
    unsafe["rtsp_url"] = "rtsp://user:secret@camera/stream"
    with pytest.raises(ValueError):
        recorder.submit(np.ones((30, 109)), unsafe)


def test_cooldown_deduplicates_same_camera_track_trigger(tmp_path):
    recorder = TemporalCandidateRecorder(tmp_path, cooldown_sec=5)
    recorder.start()
    try:
        window = np.zeros((30, 109), dtype=np.float32)
        assert recorder.submit(window, metadata(), now_mono=10).accepted
        second = recorder.submit(window, metadata(31), now_mono=12)
        assert not second.accepted
        assert second.reason == "cooldown"
        assert recorder.submit(window, metadata(32), now_mono=16).accepted
    finally:
        recorder.stop()
