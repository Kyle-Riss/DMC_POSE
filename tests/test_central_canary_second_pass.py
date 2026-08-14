from datetime import datetime, timezone

from central_canary_second_pass import CentralCanarySecondPass
from edge_contract_v1 import EdgeEventStart


def event(event_id="evt-1"):
    return EdgeEventStart(
        event_id=event_id, node_id="node", camera_id="bed_161", boot_id="boot",
        started_at=datetime.now(timezone.utc), start_frame_seq=1, event_type="CANDIDATE",
        model_bundle_version="bundle", roi_version=1, pre_event_frames_available=0,
        pre_event_coverage_sec=0, peak_risk=0.8, evidence=[],
    )


def test_central_reference_confirms_without_opening_stream(tmp_path):
    loader = lambda _url: {"bed_161": {"fusion_phase": "VERIFY", "fusion_risk": 0.8}}
    verifier = CentralCanarySecondPass(tmp_path, frame_wait_sec=0, status_loader=loader)
    result = verifier(event())
    assert result["decision"] == "central_reference_confirmed"
    assert result["continuous_stream_opened"] is False


def test_uploaded_evidence_is_counted_when_reference_unavailable(tmp_path):
    frame_dir = tmp_path / "evt-2"
    frame_dir.mkdir()
    (frame_dir / "node__bed_161__000000000001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    def unavailable(_url): raise OSError("offline")
    verifier = CentralCanarySecondPass(tmp_path, frame_wait_sec=0, status_loader=unavailable)
    result = verifier(event("evt-2"))
    assert result["decision"] == "evidence_received_analyzer_pending"
    assert result["evidence_frame_count"] == 1
