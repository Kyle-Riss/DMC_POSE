import pytest

from scripts.reconcile_temporal_annotation_metadata import reconcile


def manifest(frame_count=100):
    return {"items": [{
        "video_id": "v1", "readable": True, "fps": 20.0,
        "frame_count": frame_count, "duration_sec": frame_count / 20,
        "width": 640, "height": 360,
    }]}


def row(end="90"):
    return {
        "video_id": "v1", "decode_ok": "true", "fps": "20.0",
        "frame_count": "120", "duration_sec": "6.0", "width": "640", "height": "360",
        "fall_onset_frame": "10", "impact_frame": "20",
        "post_fall_stable_frame": "30", "fall_end_frame": end,
        "onset_earliest_frame": "9", "onset_latest_frame": "11",
        "annotation_status": "complete",
    }


def test_reconciles_decoded_count_without_losing_completed_boundaries():
    rows, report = reconcile([row()], manifest())
    assert rows[0]["frame_count"] == "100"
    assert rows[0]["fall_end_frame"] == "90"
    assert report["changed_rows"] == 1
    assert report["completed_rows_preserved"] == 1


def test_fails_closed_when_existing_boundary_exceeds_decoded_frames():
    with pytest.raises(ValueError, match="outside decoded frame range"):
        reconcile([row(end="110")], manifest())
