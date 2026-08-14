"""Bounded, event-only central verification dispatcher.

This module never consumes RTSP continuously. It accepts only event-start
metadata, de-duplicates event IDs, and invokes a supplied verifier on a bounded
worker queue. The deployment verifier can pull a short RTSP burst or inspect
uploaded event frames without coupling edge request latency to GPU inference.
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from edge_contract_v1 import EdgeEventStart


@dataclass(frozen=True)
class DispatchResult:
    accepted: bool
    duplicate: bool = False
    dropped: bool = False


class SelectiveSecondPass:
    def __init__(
        self,
        verifier: Callable[[EdgeEventStart], dict],
        output_dir: str | Path,
        *,
        queue_size: int = 32,
        workers: int = 1,
    ):
        self.verifier = verifier
        self.output_dir = Path(output_dir)
        self.queue: queue.Queue[EdgeEventStart | None] = queue.Queue(maxsize=queue_size)
        self.workers = max(1, int(workers))
        self._threads: list[threading.Thread] = []
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self.accepted = 0
        self.duplicates = 0
        self.dropped = 0
        self.completed = 0
        self.errors = 0

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self._threads:
            return
        for index in range(self.workers):
            thread = threading.Thread(target=self._loop, name=f"second-pass-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        for _ in self._threads:
            self.queue.put(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

    def submit(self, event: EdgeEventStart) -> DispatchResult:
        if event.event_type not in {"CANDIDATE", "FALL", "BED_EXIT_FALL"}:
            return DispatchResult(accepted=False)
        with self._lock:
            if event.event_id in self._seen:
                self.duplicates += 1
                return DispatchResult(accepted=True, duplicate=True)
            self._seen.add(event.event_id)
        try:
            self.queue.put_nowait(event)
            self.accepted += 1
            return DispatchResult(accepted=True)
        except queue.Full:
            with self._lock:
                self._seen.discard(event.event_id)
            self.dropped += 1
            return DispatchResult(accepted=False, dropped=True)

    def snapshot(self) -> dict:
        return {
            "queue_depth": self.queue.qsize(), "accepted": self.accepted,
            "duplicates": self.duplicates, "dropped": self.dropped,
            "completed": self.completed, "errors": self.errors,
            "workers_alive": sum(thread.is_alive() for thread in self._threads),
        }

    def _write(self, payload: dict) -> None:
        path = self.output_dir / "second_pass.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _loop(self) -> None:
        while True:
            event = self.queue.get()
            try:
                if event is None:
                    return
                started = datetime.now(timezone.utc)
                try:
                    result = self.verifier(event)
                    self.completed += 1
                    status = "completed"
                except Exception as exc:
                    self.errors += 1
                    result = {"error": str(exc)}
                    status = "error"
                self._write({
                    "event_id": event.event_id,
                    "node_id": event.node_id,
                    "camera_id": event.camera_id,
                    "event_type": event.event_type,
                    "status": status,
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                })
            finally:
                self.queue.task_done()
