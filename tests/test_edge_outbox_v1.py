from datetime import datetime, timezone

from edge_contract_v1 import EdgeHeartbeat
from edge_outbox_v1 import AsyncOutboxWriter, EdgeOutbox, EdgeOutboxSender


def heartbeat(sequence=1):
    return EdgeHeartbeat(
        node_id="rpi-bed-161", camera_id="bed_161", boot_id="boot-a",
        sequence=sequence, sent_at=datetime(2026, 8, 7, tzinfo=timezone.utc),
        software_version="edge-v1", uptime_sec=1, capture_connected=True,
        capture_fps=20, watcher_fps=5, runtime_mode="EMPTY",
        roi_state="READY", roi_version=1, spool_depth=0, spool_bytes=0,
        storage_free_mb=4096, capabilities={},
    )


def test_outbox_is_durable_and_idempotent(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    outbox = EdgeOutbox(path)
    assert outbox.enqueue(heartbeat(), now=10)
    assert not outbox.enqueue(heartbeat(), now=10)
    assert EdgeOutbox(path).stats()["pending"] == 1


def test_failure_backoff_then_acknowledge(tmp_path):
    outbox = EdgeOutbox(tmp_path / "outbox.sqlite3")
    outbox.enqueue(heartbeat(), now=10)
    sender = EdgeOutboxSender(outbox)
    result = sender.flush_once(lambda endpoint, payload: False, now=10)
    assert result["failed"] == 1
    assert result["retrying"] == 1
    assert outbox.due(now=10) == []
    result = sender.flush_once(lambda endpoint, payload: True, now=11)
    assert result["sent"] == 1
    assert result["pending"] == 0


def test_async_writer_keeps_inference_path_off_sqlite_and_compacts_heartbeats(tmp_path):
    outbox = EdgeOutbox(tmp_path / "outbox.sqlite3")
    writer = AsyncOutboxWriter(outbox)
    writer.start()
    for sequence in range(5):
        assert writer.submit(heartbeat(sequence))
    writer.stop()
    assert writer.errors == 0
    assert outbox.stats()["pending"] == 1
    assert outbox.due()[0].payload["sequence"] == 4


def test_new_heartbeat_replaces_failed_older_snapshot(tmp_path):
    outbox = EdgeOutbox(tmp_path / "outbox.sqlite3")
    outbox.enqueue(heartbeat(10), now=10)
    EdgeOutboxSender(outbox).flush_once(lambda endpoint, payload: False, now=10)
    assert outbox.stats() == {"pending": 1, "payload_bytes": outbox.stats()["payload_bytes"], "retrying": 1}

    assert outbox.enqueue(heartbeat(11), now=11)
    pending = outbox.due(now=11)
    assert len(pending) == 1
    assert pending[0].payload["sequence"] == 11
    assert pending[0].attempts == 0


def test_wire_payload_cannot_contain_rtsp_credentials(tmp_path):
    outbox = EdgeOutbox(tmp_path / "outbox.sqlite3")
    outbox.enqueue(heartbeat(), now=10)
    pending = outbox.due(now=10)[0]
    encoded = str(pending.payload)
    assert "rtsp://" not in encoded
    assert "password" not in encoded
