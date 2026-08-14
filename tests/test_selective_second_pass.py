from datetime import datetime, timezone

from edge_contract_v1 import EdgeEventStart
from selective_second_pass import SelectiveSecondPass


def event(event_id="evt-1", event_type="CANDIDATE"):
    return EdgeEventStart(
        event_id=event_id, node_id="node-1", camera_id="bed_161", boot_id="boot-1",
        started_at=datetime.now(timezone.utc), start_frame_seq=10, event_type=event_type,
        model_bundle_version="test", roi_version=1, pre_event_frames_available=100,
        pre_event_coverage_sec=5.0, peak_risk=0.7, evidence=["motion"],
    )


def test_dispatch_is_selective_and_idempotent(tmp_path):
    worker = SelectiveSecondPass(lambda item: {"verified": item.event_id}, tmp_path)
    worker.start()
    assert not worker.submit(event(event_type="BED_EXIT")).accepted
    assert worker.submit(event()).accepted
    assert worker.submit(event()).duplicate
    worker.queue.join()
    assert worker.snapshot()["completed"] == 1
    worker.stop()
