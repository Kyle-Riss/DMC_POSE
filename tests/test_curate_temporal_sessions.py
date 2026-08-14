import json

import numpy as np

from scripts.curate_temporal_sessions import contiguous_slices, curate_session


def test_contiguous_slices_break_on_gap_and_track_change():
    timestamps = np.asarray([0, .1, .2, .6, .7, .8, .9])
    track_ids = np.asarray([1, 1, 1, 1, 1, 2, 2])
    assert [(item.start, item.stop) for item in contiguous_slices(timestamps, track_ids)] == [
        (0, 3), (3, 5), (5, 7)
    ]


def test_reviewed_negative_builds_windows_without_crossing_gap(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    timestamps = np.r_[np.arange(35) * .1, 10 + np.arange(34) * .1]
    features = np.arange(len(timestamps) * 109, dtype=np.float32).reshape(-1, 109)
    np.savez_compressed(
        session / "features.npz",
        features=features,
        relative_timestamps_sec=timestamps,
        track_ids=np.ones(len(timestamps), dtype=np.int64),
        pose_quality=np.ones(len(timestamps), dtype=np.float32),
    )
    (session / "manifest.json").write_text(json.dumps({
        "session_id": "s1",
        "review_status": "reviewed",
        "label": "NORMAL_SIT",
        "binary_fall_label": 0,
    }))
    windows, summary = curate_session(session, stride=5)
    assert len(windows) == 3
    assert [item["y"] for item in windows] == [0, 0, 0]
    assert summary["segment_lengths"] == [35, 34]
    assert all(item["x"].shape == (30, 109) for item in windows)


def test_unreviewed_session_is_excluded(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "manifest.json").write_text(json.dumps({
        "session_id": "s1", "review_status": "unreviewed"
    }))
    windows, summary = curate_session(session)
    assert windows == []
    assert summary["excluded"] == "not_reviewed"
