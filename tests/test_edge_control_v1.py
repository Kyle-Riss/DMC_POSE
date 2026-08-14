from copy import deepcopy
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from edge_contract_v1 import EdgeHeartbeat, EdgeInferenceResult
from edge_control_server import create_app


NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc).isoformat()


def heartbeat(sequence=1):
    return {
        "node_id": "rpi-bed-161",
        "camera_id": "bed_161",
        "boot_id": "boot-a",
        "sequence": sequence,
        "sent_at": NOW,
        "software_version": "edge-v1",
        "model_bundle_version": None,
        "uptime_sec": 10,
        "capture_connected": True,
        "capture_fps": 20,
        "watcher_fps": 5,
        "runtime_mode": "OCCUPIED",
        "roi_state": "READY",
        "roi_version": 1,
        "spool_depth": 0,
        "spool_bytes": 0,
        "storage_free_mb": 4096,
        "capabilities": {
            "camera_capture": True,
            "rtsp_publish": True,
            "motion_watcher": True,
            "automatic_bed_roi": True,
            "pose_inference": True,
            "temporal_inference": True,
            "fusion": True,
            "event_frame_upload": True,
        },
    }


def inference(frame_seq=30):
    return {
        "node_id": "rpi-bed-161",
        "camera_id": "bed_161",
        "boot_id": "boot-a",
        "frame_seq": frame_seq,
        "captured_at": NOW,
        "model_bundle_version": "rpi5-benchmark-required-v1",
        "roi_version": 1,
        "primary_track_id": 3,
        "person_present": True,
        "body_in_bed_ratio": 0.7,
        "pose_label": "lying",
        "pose_confidence": 0.9,
        "temporal_ready": True,
        "temporal_samples": 30,
        "temporal_probability": 0.2,
        "temporal_candidate": False,
        "fusion_phase": "SAFE",
        "fusion_risk": 0.1,
        "evidence": ["person", "bed_overlap"],
        "quality": 0.95,
        "inference_latency_ms": 31,
    }


def event_start():
    return {
        "event_id": "rpi-bed-161:boot-a:100",
        "node_id": "rpi-bed-161",
        "camera_id": "bed_161",
        "boot_id": "boot-a",
        "started_at": NOW,
        "start_frame_seq": 100,
        "event_type": "BED_EXIT_FALL",
        "model_bundle_version": "rpi5-benchmark-required-v1",
        "roi_version": 1,
        "pre_event_frames_available": 60,
        "pre_event_coverage_sec": 3.0,
        "peak_risk": 0.91,
        "evidence": ["rapid_descent", "left_bed"],
    }


def event_end():
    return {
        "event_id": "rpi-bed-161:boot-a:100",
        "node_id": "rpi-bed-161",
        "camera_id": "bed_161",
        "boot_id": "boot-a",
        "ended_at": NOW,
        "end_frame_seq": 160,
        "peak_risk": 0.96,
        "uploaded_frame_count": 6,
        "close_reason": "completed",
    }


def test_contract_rejects_naive_time_and_rtsp_secret():
    payload = heartbeat()
    payload["sent_at"] = "2026-08-07T12:00:00"
    with pytest.raises(ValidationError):
        EdgeHeartbeat.model_validate(payload)

    payload = heartbeat()
    payload["rtsp_url"] = "rtsp://user:secret@camera/stream"
    with pytest.raises(ValidationError):
        EdgeHeartbeat.model_validate(payload)


def test_temporal_ready_requires_30_observed_samples():
    payload = inference()
    payload["temporal_samples"] = 29
    with pytest.raises(ValidationError):
        EdgeInferenceResult.model_validate(payload)


def test_api_is_idempotent_and_rejects_sequence_regression(tmp_path):
    app = create_app(runtime_dir=tmp_path)
    with TestClient(app) as client:
        first = client.post("/edge/heartbeat", json=heartbeat(2))
        assert first.status_code == 200
        assert first.json()["duplicate"] is False

        duplicate = client.post("/edge/heartbeat", json=heartbeat(2))
        assert duplicate.status_code == 200
        assert duplicate.json()["duplicate"] is True

        regression = client.post("/edge/heartbeat", json=heartbeat(1))
        assert regression.status_code == 409


def test_result_order_and_node_snapshot(tmp_path):
    app = create_app(runtime_dir=tmp_path)
    with TestClient(app) as client:
        assert client.post("/edge/heartbeat", json=heartbeat()).status_code == 200
        assert client.post("/edge/results", json=inference(30)).status_code == 200
        assert client.post("/edge/results", json=inference(29)).status_code == 409
        nodes = client.get("/edge/nodes").json()["nodes"]
        assert nodes[0]["camera_id"] == "bed_161"
        assert nodes[0]["latest_result"]["frame_seq"] == 30


def test_event_owner_lifecycle_and_upload_instruction(tmp_path):
    app = create_app(runtime_dir=tmp_path)
    with TestClient(app) as client:
        started = client.post("/events/start", json=event_start())
        assert started.status_code == 200
        assert started.json()["upload_event_frames"] is True

        wrong_owner = deepcopy(event_end())
        wrong_owner["camera_id"] = "bed_162"
        assert client.post("/events/end", json=wrong_owner).status_code == 409

        ended = client.post("/events/end", json=event_end())
        assert ended.status_code == 200
        assert client.post("/events/end", json=event_end()).json()["duplicate"] is True


def test_manifest_is_not_accidentally_promoted(tmp_path):
    app = create_app(runtime_dir=tmp_path)
    with TestClient(app) as client:
        manifest = client.get("/edge/model-manifest").json()
        assert manifest["status"] == "benchmark_required"
        assert manifest["artifacts"] == []
        assert manifest["temporal_rows"] == 30
