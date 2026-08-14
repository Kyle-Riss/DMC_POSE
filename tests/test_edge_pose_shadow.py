import json

import numpy as np

from edge_pose_shadow import EdgePoseShadow


class Watcher:
    def status(self):
        return {"trigger_total": 0}


def make_shadow(tmp_path, **overrides):
    values = {
        "rtsp_url": "rtsp://127.0.0.1:8554/stream",
        "model_path": "pose.onnx",
        "status_path": str(tmp_path / "status.json"),
        "log_path": str(tmp_path / "events.jsonl"),
    }
    values.update(overrides)
    return EdgePoseShadow(Watcher(), **values)


def test_rotation_90_clockwise(tmp_path):
    shadow = make_shadow(tmp_path, rotation_degrees=90)
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert np.array_equal(shadow._rotate(image), np.rot90(image, k=3))


def test_persist_is_atomic_and_appends(tmp_path):
    shadow = make_shadow(tmp_path)
    event = {"trigger_total": 4, "shadow_only": True}
    shadow._persist(event)
    assert json.loads((tmp_path / "status.json").read_text()) == event
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [event]


def test_invalid_rotation_rejected(tmp_path):
    try:
        make_shadow(tmp_path, rotation_degrees=45)
    except ValueError as exc:
        assert "rotation_degrees" in str(exc)
    else:
        raise AssertionError("invalid rotation was accepted")
