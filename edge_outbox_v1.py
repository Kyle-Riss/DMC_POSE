"""Durable Pi-side JSON outbox with idempotent messages and retry backoff."""

from __future__ import annotations

import hashlib
import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from edge_contract_v1 import EdgeEventEnd, EdgeEventStart, EdgeHeartbeat, EdgeInferenceResult

WireMessage = EdgeHeartbeat | EdgeInferenceResult | EdgeEventStart | EdgeEventEnd
ENDPOINTS = {
    EdgeHeartbeat: "/edge/heartbeat",
    EdgeInferenceResult: "/edge/results",
    EdgeEventStart: "/events/start",
    EdgeEventEnd: "/events/end",
}


@dataclass(frozen=True)
class PendingMessage:
    message_id: str
    endpoint: str
    payload: dict
    attempts: int


class EdgeOutbox:
    """Small SQLite spool. Call through AsyncOutboxWriter from inference code."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=2.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS outbox (
                    message_id TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    last_error TEXT
                )"""
            )

    @staticmethod
    def encode(message: WireMessage) -> tuple[str, str, str]:
        endpoint = ENDPOINTS.get(type(message))
        if endpoint is None:
            raise TypeError(f"unsupported edge message: {type(message).__name__}")
        payload_json = message.model_dump_json(exclude_none=False)
        digest = hashlib.sha256(f"{endpoint}\n{payload_json}".encode()).hexdigest()
        return digest, endpoint, payload_json

    def enqueue(self, message: WireMessage, *, now: float | None = None) -> bool:
        message_id, endpoint, payload_json = self.encode(message)
        created = time.time() if now is None else float(now)
        with self._lock, self._connect() as connection:
            # A heartbeat is a replaceable state snapshot, not an event. Keeping
            # an older heartbeat after a newer sequence exists only creates a
            # permanently rejected retry backlog at the central monotonic gate.
            # Preserve the same message ID for idempotency, but compact every
            # older heartbeat before inserting the latest snapshot. Inference
            # results and event messages remain durable and are never compacted.
            if endpoint == "/edge/heartbeat":
                connection.execute(
                    "DELETE FROM outbox WHERE endpoint = ? AND message_id <> ?",
                    (endpoint, message_id),
                )
            cursor = connection.execute(
                "INSERT OR IGNORE INTO outbox VALUES (?, ?, ?, ?, 0, ?, NULL)",
                (message_id, endpoint, payload_json, created, created),
            )
            return cursor.rowcount == 1

    def due(self, *, now: float | None = None, limit: int = 32) -> list[PendingMessage]:
        current = time.time() if now is None else float(now)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT message_id, endpoint, payload_json, attempts
                   FROM outbox WHERE next_attempt_at <= ?
                   ORDER BY created_at, message_id LIMIT ?""",
                (current, max(1, int(limit))),
            ).fetchall()
        return [PendingMessage(row[0], row[1], json.loads(row[2]), row[3]) for row in rows]

    def acknowledge(self, message_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM outbox WHERE message_id = ?", (message_id,))

    def fail(self, message_id: str, error: str, *, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT attempts FROM outbox WHERE message_id = ?", (message_id,)
            ).fetchone()
            if not row:
                return
            attempts = int(row[0]) + 1
            delay = min(60.0, 0.5 * (2 ** min(attempts - 1, 7)))
            connection.execute(
                """UPDATE outbox SET attempts = ?, next_attempt_at = ?, last_error = ?
                   WHERE message_id = ?""",
                (attempts, current + delay, str(error)[:500], message_id),
            )

    def stats(self) -> dict:
        with self._lock, self._connect() as connection:
            count, size = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload_json)), 0) FROM outbox"
            ).fetchone()
            failed = connection.execute("SELECT COUNT(*) FROM outbox WHERE attempts > 0").fetchone()[0]
        return {"pending": int(count), "payload_bytes": int(size), "retrying": int(failed)}


class AsyncOutboxWriter:
    """Bounded adapter so SQLite never runs in the camera inference thread."""

    def __init__(self, outbox: EdgeOutbox, *, queue_size: int = 512):
        self.outbox = outbox
        self.queue: queue.Queue[WireMessage | None] = queue.Queue(maxsize=queue_size)
        self.thread: threading.Thread | None = None
        self.accepted = 0
        self.dropped = 0
        self.errors = 0

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, name="edge-outbox-writer", daemon=True)
        self.thread.start()

    def submit(self, message: WireMessage) -> bool:
        try:
            self.queue.put_nowait(message)
            self.accepted += 1
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def stop(self, timeout: float = 5.0) -> None:
        if not self.thread:
            return
        self.queue.put(None, timeout=timeout)
        self.thread.join(timeout=timeout)

    def _loop(self) -> None:
        while True:
            message = self.queue.get()
            try:
                if message is None:
                    return
                self.outbox.enqueue(message)
            except Exception:
                self.errors += 1
            finally:
                self.queue.task_done()


class EdgeOutboxSender:
    """Flush due messages using an injected HTTP-like post callback."""

    def __init__(self, outbox: EdgeOutbox):
        self.outbox = outbox

    def flush_once(
        self,
        post: Callable[[str, dict], bool],
        *,
        now: float | None = None,
        limit: int = 32,
    ) -> dict:
        sent = failed = 0
        for message in self.outbox.due(now=now, limit=limit):
            try:
                if not post(message.endpoint, message.payload):
                    raise RuntimeError("server rejected message")
                self.outbox.acknowledge(message.message_id)
                sent += 1
            except Exception as exc:
                self.outbox.fail(message.message_id, str(exc), now=now)
                failed += 1
        return {"sent": sent, "failed": failed, **self.outbox.stats()}
