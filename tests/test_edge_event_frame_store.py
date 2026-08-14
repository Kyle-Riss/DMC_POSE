import pytest

from edge_event_frame_store import EventFrameStore, EvidenceFrameError


JPEG = b"\xff\xd8test\xff\xd9"


def test_frame_store_is_bounded_safe_and_idempotent(tmp_path):
    store = EventFrameStore(tmp_path, max_frame_bytes=100, max_frames_per_event=1)
    first = store.put(event_id="evt-1", node_id="node-1", camera_id="bed_161", frame_seq=1, jpeg=JPEG)
    assert not first.duplicate
    assert store.put(event_id="evt-1", node_id="node-1", camera_id="bed_161", frame_seq=1, jpeg=JPEG).duplicate
    with pytest.raises(EvidenceFrameError):
        store.put(event_id="evt-1", node_id="node-1", camera_id="bed_161", frame_seq=2, jpeg=JPEG)
    with pytest.raises(EvidenceFrameError):
        store.put(event_id="../escape", node_id="node-1", camera_id="bed_161", frame_seq=1, jpeg=JPEG)
    with pytest.raises(EvidenceFrameError):
        store.put(event_id="evt-2", node_id="node-1", camera_id="bed_161", frame_seq=1, jpeg=b"not-jpeg")
