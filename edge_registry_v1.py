"""Thread-safe edge control-plane registry with non-blocking JSONL persistence."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edge_contract_v1 import EdgeEventEnd, EdgeEventStart, EdgeHeartbeat, EdgeInferenceResult


class SequenceRegression(ValueError):
    """A node reused an old sequence number with different content."""


class EventLifecycleError(ValueError):
    """An event end does not match its open event."""


@dataclass(frozen=True)
class AcceptResult:
    accepted: bool
    duplicate: bool = False


class EdgeRegistry:
    """Keep current edge state in memory and persist an append-only audit stream.

    Persistence is deliberately best-effort and asynchronous. A slow disk must not
    hold an edge HTTP request open or slow camera inference.
    """

    def __init__(self, output_dir: str | Path, queue_size: int = 4096):
        self.output_dir = Path(output_dir)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._lock = threading.RLock()
        self._writer: threading.Thread | None = None
        self._heartbeats: dict[tuple[str, str], EdgeHeartbeat] = {}
        self._results: dict[tuple[str, str], EdgeInferenceResult] = {}
        self._open_events: dict[str, EdgeEventStart] = {}
        self._closed_events: dict[str, EdgeEventEnd] = {}
        self._heartbeat_fingerprints: dict[tuple[str, str, str, int], str] = {}
        self._result_fingerprints: dict[tuple[str, str, str, int], str] = {}
        self.written = 0
        self.dropped = 0
        self.write_errors = 0

    def start(self) -> None:
        with self._lock:
            if self._writer and self._writer.is_alive():
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._writer = threading.Thread(target=self._write_loop, name="edge-jsonl-writer", daemon=True)
            self._writer.start()

    def stop(self, timeout: float = 5.0) -> None:
        writer = self._writer
        if not writer:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            self.dropped += 1
        writer.join(timeout=timeout)

    @staticmethod
    def _fingerprint(model: Any) -> str:
        return model.model_dump_json(exclude_none=False)

    def _enqueue(self, kind: str, model: Any) -> None:
        item = {
            "kind": kind,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "payload": model.model_dump(mode="json"),
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.dropped += 1

    def accept_heartbeat(self, heartbeat: EdgeHeartbeat) -> AcceptResult:
        node_key = (heartbeat.node_id, heartbeat.camera_id)
        packet_key = (*node_key, heartbeat.boot_id, heartbeat.sequence)
        fingerprint = self._fingerprint(heartbeat)
        with self._lock:
            old_fingerprint = self._heartbeat_fingerprints.get(packet_key)
            if old_fingerprint is not None:
                if old_fingerprint != fingerprint:
                    raise SequenceRegression("heartbeat sequence reused with different payload")
                return AcceptResult(accepted=True, duplicate=True)
            previous = self._heartbeats.get(node_key)
            if previous and previous.boot_id == heartbeat.boot_id and heartbeat.sequence < previous.sequence:
                raise SequenceRegression("heartbeat sequence moved backwards")
            self._heartbeats[node_key] = heartbeat
            self._heartbeat_fingerprints[packet_key] = fingerprint
        self._enqueue("heartbeat", heartbeat)
        return AcceptResult(accepted=True)

    def accept_result(self, result: EdgeInferenceResult) -> AcceptResult:
        node_key = (result.node_id, result.camera_id)
        packet_key = (*node_key, result.boot_id, result.frame_seq)
        fingerprint = self._fingerprint(result)
        with self._lock:
            old_fingerprint = self._result_fingerprints.get(packet_key)
            if old_fingerprint is not None:
                if old_fingerprint != fingerprint:
                    raise SequenceRegression("frame sequence reused with different payload")
                return AcceptResult(accepted=True, duplicate=True)
            previous = self._results.get(node_key)
            if previous and previous.boot_id == result.boot_id and result.frame_seq < previous.frame_seq:
                raise SequenceRegression("frame sequence moved backwards")
            self._results[node_key] = result
            self._result_fingerprints[packet_key] = fingerprint
        self._enqueue("result", result)
        return AcceptResult(accepted=True)

    def start_event(self, event: EdgeEventStart) -> AcceptResult:
        with self._lock:
            previous = self._open_events.get(event.event_id)
            if previous:
                if self._fingerprint(previous) != self._fingerprint(event):
                    raise EventLifecycleError("event_id already open with different payload")
                return AcceptResult(accepted=True, duplicate=True)
            if event.event_id in self._closed_events:
                raise EventLifecycleError("event_id is already closed")
            self._open_events[event.event_id] = event
        self._enqueue("event_start", event)
        return AcceptResult(accepted=True)

    def end_event(self, event: EdgeEventEnd) -> AcceptResult:
        with self._lock:
            closed = self._closed_events.get(event.event_id)
            if closed:
                if self._fingerprint(closed) != self._fingerprint(event):
                    raise EventLifecycleError("event_id already closed with different payload")
                return AcceptResult(accepted=True, duplicate=True)
            opened = self._open_events.get(event.event_id)
            if not opened:
                raise EventLifecycleError("event_id has no open event")
            if (opened.node_id, opened.camera_id, opened.boot_id) != (
                event.node_id, event.camera_id, event.boot_id
            ):
                raise EventLifecycleError("event owner does not match")
            if event.end_frame_seq < opened.start_frame_seq:
                raise EventLifecycleError("event end precedes event start")
            del self._open_events[event.event_id]
            self._closed_events[event.event_id] = event
        self._enqueue("event_end", event)
        return AcceptResult(accepted=True)

    def validate_open_event_owner(self, event_id: str, node_id: str, camera_id: str) -> EdgeEventStart:
        """Authorize evidence upload only while the matching event is open."""
        with self._lock:
            opened = self._open_events.get(event_id)
            if not opened:
                raise EventLifecycleError("event_id has no open event")
            if (opened.node_id, opened.camera_id) != (node_id, camera_id):
                raise EventLifecycleError("event owner does not match")
            return opened

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            nodes = []
            for key, heartbeat in sorted(self._heartbeats.items()):
                result = self._results.get(key)
                nodes.append({
                    "node_id": key[0],
                    "camera_id": key[1],
                    "heartbeat": heartbeat.model_dump(mode="json"),
                    "latest_result": result.model_dump(mode="json") if result else None,
                })
            return {
                "nodes": nodes,
                "open_events": len(self._open_events),
                "closed_events": len(self._closed_events),
                "writer": {
                    "queue_depth": self._queue.qsize(),
                    "written": self.written,
                    "dropped": self.dropped,
                    "errors": self.write_errors,
                    "alive": bool(self._writer and self._writer.is_alive()),
                },
            }

    def _write_loop(self) -> None:
        path = self.output_dir / "edge_control.jsonl"
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                with self._lock:
                    self.written += 1
            except Exception:
                with self._lock:
                    self.write_errors += 1
            finally:
                self._queue.task_done()
