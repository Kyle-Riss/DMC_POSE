import json
from pathlib import Path

import edge_signal_client
from edge_signal_client import EdgeSignalClient


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return json.dumps(self.payload).encode()


def test_fresh_quality_person_result_wakes(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("x" * 32)
    payload = {"nodes": [{
        "camera_id": "bed_161",
        "latest_result": {
            "captured_at": "2026-08-10T03:20:44+00:00",
            "person_present": True,
            "pose_confidence": 0.85,
            "quality": 0.8,
            "frame_seq": 12,
            "model_bundle_version": "cm4-onnx-candidate-v1",
        },
    }]}
    monkeypatch.setattr(edge_signal_client.urllib.request, "urlopen", lambda *_a, **_k: Response(payload))
    client = EdgeSignalClient("http://127.0.0.1:8020", token)
    assert client.poll_once() == 1
    status = client.status("bed_161", now_unix=1786332045.0)
    assert status["wake_active"] is True
    assert status["frame_seq"] == 12


def test_stale_or_low_quality_result_does_not_wake(tmp_path):
    token = tmp_path / "token"
    token.write_text("x" * 32)
    client = EdgeSignalClient("http://127.0.0.1:8020", token)
    with client._lock:
        client._results["bed_161"] = {
            "captured_unix": 10.0, "person_present": True,
            "pose_confidence": 0.9, "quality": 0.1,
        }
    assert client.status("bed_161", now_unix=11.0)["wake_active"] is False
    stale = client.status("bed_161", now_unix=20.0)
    assert stale["result_fresh"] is False
    assert stale["person_present"] is False
    assert stale["last_person_present"] is True
