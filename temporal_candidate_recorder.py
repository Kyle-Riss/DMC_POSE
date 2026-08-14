"""Asynchronous recorder for privacy-preserving 30x109 temporal candidates."""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_SCHEMA = "pose_temporal_109_v1"
SAFE_METADATA = {
    "node_id", "camera_id", "boot_id", "frame_seq", "captured_at",
    "track_id", "model_bundle_version", "roi_version", "trigger",
    "temporal_probability", "fusion_risk", "fusion_phase", "quality",
    "evidence", "sample_timestamps",
}


@dataclass(frozen=True)
class CandidateReceipt:
    accepted: bool
    candidate_id: str | None = None
    reason: str | None = None


class TemporalCandidateRecorder:
    """Persist candidate features without blocking real-time inference.

    Raw frames, raw keypoints, RTSP URLs and credentials are not accepted as
    metadata. A bounded queue makes overload explicit instead of growing RAM.
    """

    def __init__(self, output_dir: str | Path, *, queue_size: int = 128, cooldown_sec: float = 5.0):
        self.output_dir = Path(output_dir)
        self.queue: queue.Queue[tuple[str, np.ndarray, dict[str, Any]] | None] = queue.Queue(maxsize=queue_size)
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self._last_by_key: dict[tuple[str, str, str], float] = {}
        self._lock = threading.Lock()
        self._writer: threading.Thread | None = None
        self.accepted = 0
        self.written = 0
        self.dropped = 0
        self.errors = 0

    def start(self) -> None:
        with self._lock:
            if self._writer and self._writer.is_alive():
                return
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._writer = threading.Thread(target=self._write_loop, name="temporal-candidate-writer", daemon=True)
            self._writer.start()

    def stop(self, timeout: float = 5.0) -> None:
        if not self._writer:
            return
        self.queue.put(None, timeout=timeout)
        self._writer.join(timeout=timeout)

    @staticmethod
    def _validate_window(window: np.ndarray) -> np.ndarray:
        array = np.asarray(window, dtype=np.float32)
        if array.shape != (30, 109):
            raise ValueError(f"candidate window must be (30, 109), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("candidate window contains non-finite values")
        return np.array(array, dtype=np.float32, copy=True)

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        unknown = set(metadata) - SAFE_METADATA
        if unknown:
            raise ValueError(f"unsafe or unknown candidate metadata: {sorted(unknown)}")
        required = {"node_id", "camera_id", "boot_id", "frame_seq", "captured_at", "trigger"}
        missing = required - set(metadata)
        if missing:
            raise ValueError(f"missing candidate metadata: {sorted(missing)}")
        return json.loads(json.dumps(metadata, ensure_ascii=False))

    def submit(self, window: np.ndarray, metadata: dict[str, Any], *, now_mono: float | None = None) -> CandidateReceipt:
        array = self._validate_window(window)
        clean = self._sanitize_metadata(metadata)
        now = time.monotonic() if now_mono is None else float(now_mono)
        key = (str(clean["camera_id"]), str(clean.get("track_id", "none")), str(clean["trigger"]))
        with self._lock:
            last = self._last_by_key.get(key)
            if last is not None and now - last < self.cooldown_sec:
                return CandidateReceipt(False, reason="cooldown")
            digest = hashlib.sha256()
            digest.update(array.tobytes(order="C"))
            digest.update(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode())
            candidate_id = digest.hexdigest()[:24]
            try:
                self.queue.put_nowait((candidate_id, array, clean))
            except queue.Full:
                self.dropped += 1
                return CandidateReceipt(False, reason="queue_full")
            self._last_by_key[key] = now
            self.accepted += 1
        return CandidateReceipt(True, candidate_id=candidate_id)

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "unknown"

    def _write_loop(self) -> None:
        index_path = self.output_dir / "candidates.jsonl"
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                candidate_id, array, metadata = item
                camera = self._safe_name(str(metadata["camera_id"]))
                filename = f"{camera}_{int(metadata['frame_seq']):012d}_{candidate_id}.npz"
                final_path = self.output_dir / filename
                tmp_path = self.output_dir / f".{filename}.tmp"
                with tmp_path.open("wb") as handle:
                    np.savez_compressed(handle, window=array, feature_schema=np.array(FEATURE_SCHEMA))
                tmp_path.replace(final_path)
                checksum = hashlib.sha256(final_path.read_bytes()).hexdigest()
                record = {
                    "candidate_id": candidate_id,
                    "feature_schema": FEATURE_SCHEMA,
                    "shape": [30, 109],
                    "dtype": "float32",
                    "filename": filename,
                    "sha256": checksum,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    **metadata,
                }
                with index_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                with self._lock:
                    self.written += 1
            except Exception:
                with self._lock:
                    self.errors += 1
            finally:
                self.queue.task_done()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "alive": bool(self._writer and self._writer.is_alive()),
                "queue_depth": self.queue.qsize(),
                "accepted": self.accepted,
                "written": self.written,
                "dropped": self.dropped,
                "errors": self.errors,
                "feature_schema": FEATURE_SCHEMA,
            }

