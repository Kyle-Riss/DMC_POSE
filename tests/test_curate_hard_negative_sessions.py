import json
from pathlib import Path

from scripts.curate_hard_negative_sessions import select_rows, summarize


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_select_and_summarize_session(tmp_path):
    session = {
        "session_id": "safe-1", "camera_id": "bed_161",
        "start_utc": "2026-08-07T23:59:59.500Z", "end_utc": "2026-08-08T00:00:00.500Z",
        "event_label": "NO_FALL", "binary_fall_label": 0,
        "actions": ["crouch"], "annotation_precision": "session_level",
    }
    base = {
        "camera_id": "bed_161", "primary_track_id": 7, "primary_track_observed": True,
        "tcn_shadow_ready": True, "tcn_source": "live", "capture_connected": True,
        "capture_decode_error_total": 2, "scheduler_queue_latency_ms": 3.0,
    }
    write_rows(tmp_path / "shadow_features_20260807.jsonl", [
        {**base, "recorded_at": "2026-08-07T23:59:59.750Z", "tcn_fall_probability": .9,
         "tcn_alert_candidate": True, "fusion_phase": "SAFE"},
        {**base, "camera_id": "bed_162", "recorded_at": "2026-08-07T23:59:59.900Z"},
    ])
    write_rows(tmp_path / "shadow_features_20260808.jsonl", [
        {**base, "recorded_at": "2026-08-08T00:00:00.250Z", "tcn_fall_probability": .2,
         "tcn_alert_candidate": False, "fusion_phase": "SAFE"},
    ])
    rows, sources = select_rows(session, tmp_path)
    report = summarize(session, rows, sources)
    assert len(rows) == 2
    assert report["tracking"]["continuous_single_track"] is True
    assert report["tcn"]["candidate_rows"] == 1
    assert report["tcn"]["max_probability"] == .9
    assert report["fusion"]["suppressed_all_tcn_candidates"] is True
    assert report["usage_contract"]["fusion_calibration_eligible"] is True
    assert report["usage_contract"]["temporal_training_eligible"] is False


def test_empty_session_is_not_calibration_eligible(tmp_path):
    session = {
        "session_id": "empty", "camera_id": "bed_161",
        "start_utc": "2026-08-07T00:00:00Z", "end_utc": "2026-08-07T00:00:01Z",
    }
    rows, sources = select_rows(session, tmp_path)
    report = summarize(session, rows, sources)
    assert report["matched_rows"] == 0
    assert report["usage_contract"]["fusion_calibration_eligible"] is False
