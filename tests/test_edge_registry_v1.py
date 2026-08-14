from datetime import datetime, timezone

import pytest

from edge_contract_v1 import EdgeEventEnd, EdgeEventStart, EdgeHeartbeat, EdgeInferenceResult
from edge_registry_v1 import EdgeRegistry, EventLifecycleError, SequenceRegression


NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


def heartbeat(sequence=1):
    return EdgeHeartbeat(
        node_id="rpi-bed-161", camera_id="bed_161", boot_id="boot-a",
        sequence=sequence, sent_at=NOW, software_version="edge-v1",
        uptime_sec=10, capture_connected=True, capture_fps=20, watcher_fps=5,
        runtime_mode="OCCUPIED", roi_state="READY", roi_version=1,
        spool_depth=0, spool_bytes=0, storage_free_mb=4096, capabilities={},
    )


def inference(frame_seq=30):
    return EdgeInferenceResult(
        node_id="rpi-bed-161", camera_id="bed_161", boot_id="boot-a",
        frame_seq=frame_seq, captured_at=NOW,
        model_bundle_version="rpi5-benchmark-required-v1", roi_version=1,
        primary_track_id=3, person_present=True, body_in_bed_ratio=0.7,
        pose_label="lying", pose_confidence=0.9, temporal_ready=True,
        temporal_samples=30, temporal_probability=0.2, temporal_candidate=False,
        fusion_phase="SAFE", fusion_risk=0.1, evidence=["person"],
        quality=0.95, inference_latency_ms=31,
    )


def start_event():
    return EdgeEventStart(
        event_id="evt-1", node_id="rpi-bed-161", camera_id="bed_161",
        boot_id="boot-a", started_at=NOW, start_frame_seq=100,
        event_type="BED_EXIT_FALL", model_bundle_version="bundle-v1",
        roi_version=1, pre_event_frames_available=60, pre_event_coverage_sec=3,
        peak_risk=0.9, evidence=["rapid_descent"],
    )


def end_event(camera_id="bed_161", frame_seq=160):
    return EdgeEventEnd(
        event_id="evt-1", node_id="rpi-bed-161", camera_id=camera_id,
        boot_id="boot-a", ended_at=NOW, end_frame_seq=frame_seq,
        peak_risk=0.95, uploaded_frame_count=5, close_reason="completed",
    )


def test_registry_idempotency_and_monotonic_order(tmp_path):
    registry = EdgeRegistry(tmp_path)
    registry.start()
    try:
        assert registry.accept_heartbeat(heartbeat(2)).duplicate is False
        assert registry.accept_heartbeat(heartbeat(2)).duplicate is True
        with pytest.raises(SequenceRegression):
            registry.accept_heartbeat(heartbeat(1))
        registry.accept_result(inference(30))
        with pytest.raises(SequenceRegression):
            registry.accept_result(inference(29))
    finally:
        registry.stop()


def test_event_lifecycle_checks_owner_and_frame_order(tmp_path):
    registry = EdgeRegistry(tmp_path)
    registry.start()
    try:
        registry.start_event(start_event())
        with pytest.raises(EventLifecycleError):
            registry.end_event(end_event(camera_id="bed_162"))
        with pytest.raises(EventLifecycleError):
            registry.end_event(end_event(frame_seq=99))
        assert registry.end_event(end_event()).duplicate is False
        assert registry.end_event(end_event()).duplicate is True
    finally:
        registry.stop()


def test_writer_persists_without_secrets(tmp_path):
    registry = EdgeRegistry(tmp_path)
    registry.start()
    registry.accept_heartbeat(heartbeat())
    registry.stop()
    content = (tmp_path / "edge_control.jsonl").read_text(encoding="utf-8")
    assert "rpi-bed-161" in content
    assert "rtsp://" not in content
    assert "password" not in content


def test_evidence_upload_requires_matching_open_event(tmp_path):
    registry = EdgeRegistry(tmp_path)
    event = start_event()
    registry.start_event(event)
    assert registry.validate_open_event_owner(event.event_id, event.node_id, event.camera_id) == event
    with pytest.raises(EventLifecycleError, match="owner"):
        registry.validate_open_event_owner(event.event_id, "different", event.camera_id)
    with pytest.raises(EventLifecycleError, match="no open"):
        registry.validate_open_event_owner("missing", event.node_id, event.camera_id)
