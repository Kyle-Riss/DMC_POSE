import csv
import hashlib
import json

from scripts.audit_ai_runner_event_clips import audit_event


def make_event(tmp_path, *, step=0.05, ground_truth="normal_exit"):
    event = tmp_path / "event"
    frames = event / "frames"
    frames.mkdir(parents=True)
    timeline = []
    for index in range(80):
        image = frames / f"{index:04d}.jpg"
        image.write_bytes(f"frame-{index}".encode())
        timeline.append({
            "timestamp": str(1000.0 + index * step),
            "offset_from_trigger_seconds": str(index * step - 2.0),
            "frame_index": str(index),
            "phase": "pre" if index < 40 else "post",
            "image_path": f"frames/{image.name}",
            "jpeg_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        })
    with (event / "timeline.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=timeline[0])
        writer.writeheader()
        writer.writerows(timeline)
    (event / "event_meta.json").write_text(json.dumps({
        "eventId": "event", "cameraId": "bed_161", "groundTruth": ground_truth,
        "trigger": {"reason": "test"},
    }), encoding="utf-8")
    return event


def test_native_20hz_normal_clip_passes_cadence_but_not_identity(tmp_path):
    result = audit_event(make_event(tmp_path), target_hz=20.0)
    assert result["cadence_eligible"] is True
    assert result["hash_mismatch_count"] == 0
    assert result["production_gru_training_eligible"] is False
    assert "subject_identity_unknown" in result["training_blockers"]


def test_legacy_10hz_clip_is_not_upsampled(tmp_path):
    result = audit_event(make_event(tmp_path, step=0.1), target_hz=20.0)
    assert result["cadence_eligible"] is False
    assert "does_not_satisfy_observed_only_20hz_cadence" in result["training_blockers"]


def test_positive_clip_requires_temporal_boundaries(tmp_path):
    result = audit_event(make_event(tmp_path, ground_truth="simulated_fall"), target_hz=20.0)
    assert "fall_onset_impact_stable_boundaries_missing" in result["training_blockers"]
