import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_reviewed_staged_shadow_windows.py"
SPEC = importlib.util.spec_from_file_location("reviewed_shadow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(recording, status, *, notes="", onset="", impact="", end=""):
    return {
        "recording_id": recording, "video_id": f"c1_{recording}", "camera_id": "c1",
        "annotation_status": status, "notes": notes,
        "fall_onset_frame": onset, "impact_frame": impact, "fall_end_frame": end,
    }


def test_reviewed_recordings_accepts_impactless_positive_and_explicit_hard_negative():
    rows = [
        row("positive", "complete", onset="3", impact="5", end="8"),
        row("slow", "needs_adjudication", onset="2", end="9"),
        row("negative", "excluded", notes="retained as a bed-exit hard negative"),
        row("excluded", "excluded", notes="unusable clip"),
    ]
    selected, labels = MODULE.reviewed_recordings(rows)
    assert labels == {"positive": 1, "slow": 1, "negative": 0}
    assert "excluded" not in selected


def test_complete_positive_requires_impact():
    with pytest.raises(ValueError, match="lacks impact"):
        MODULE.reviewed_recordings([row("positive", "complete", onset="3", end="8")])


def test_recording_split_is_stratified_and_locked():
    labels = {**{f"p{i}": 1 for i in range(5)}, **{f"n{i}": 0 for i in range(5)}}
    splits = MODULE.recording_splits(labels)
    for label in (0, 1):
        assert {splits[key] for key, value in labels.items() if value == label} == {"train", "val", "test"}
    assert splits == MODULE.recording_splits(dict(reversed(list(labels.items()))))


def test_target_policy():
    frame = pd.DataFrame({"frame_idx": [0, 2, 3, 8, 9], "target": ["non_fall"] * 5})
    positive = row("p", "needs_adjudication", onset="3", end="8")
    out = MODULE.apply_reviewed_target(frame, positive, 1, "train")
    assert out["target"].tolist() == ["non_fall", "non_fall", "fall", "fall", "non_fall"]
    assert out["subject_id"].unique().tolist() == ["unknown_staged_subject"]
    negative = MODULE.apply_reviewed_target(frame, row("n", "excluded"), 0, "test")
    assert set(negative["target"]) == {"non_fall"}


def test_resample_observed_frame_20_to_10hz_preserves_real_rows():
    frame = pd.DataFrame({
        "timestamp_sec": [0.00, 0.05, 0.10, 0.15, 0.20],
        "frame_idx": [0, 1, 2, 3, 4],
        "sequence_id": [1] * 5,
        "track_id": [7] * 5,
    })
    result = MODULE.resample_observed_frame(frame, 10.0)
    assert result["timestamp_sec"].tolist() == [0.0, 0.1, 0.2]
    assert result["frame_idx"].tolist() == [0, 2, 4]


def test_resample_rejects_non_divisor_rate():
    with pytest.raises(ValueError, match="evenly divide"):
        MODULE.resample_observed_frame(pd.DataFrame(), 12.0)
