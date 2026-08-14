"""Bounded, idempotent evidence-frame storage for edge events."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


class EvidenceFrameError(ValueError):
    pass


@dataclass(frozen=True)
class StoredFrame:
    path: Path
    sha256: str
    bytes: int
    duplicate: bool


class EventFrameStore:
    def __init__(self, root: str | Path, *, max_frame_bytes: int = 512_000, max_frames_per_event: int = 180):
        self.root = Path(root)
        self.max_frame_bytes = int(max_frame_bytes)
        self.max_frames_per_event = int(max_frames_per_event)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(value: str, field: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value):
            raise EvidenceFrameError(f"unsafe {field}")
        return value

    def put(self, *, event_id: str, node_id: str, camera_id: str, frame_seq: int, jpeg: bytes) -> StoredFrame:
        event_id = self._safe(event_id, "event_id")
        node_id = self._safe(node_id, "node_id")
        camera_id = self._safe(camera_id, "camera_id")
        if frame_seq < 0:
            raise EvidenceFrameError("frame_seq must be non-negative")
        if not jpeg or len(jpeg) > self.max_frame_bytes:
            raise EvidenceFrameError("JPEG size is outside the allowed range")
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise EvidenceFrameError("payload is not a complete JPEG")
        event_dir = self.root / event_id
        event_dir.mkdir(mode=0o700, exist_ok=True)
        existing = list(event_dir.glob("*.jpg"))
        target = event_dir / f"{node_id}__{camera_id}__{frame_seq:012d}.jpg"
        digest = hashlib.sha256(jpeg).hexdigest()
        if target.exists():
            old = target.read_bytes()
            if hashlib.sha256(old).hexdigest() != digest:
                raise EvidenceFrameError("frame sequence reused with different JPEG")
            return StoredFrame(target, digest, len(old), True)
        if len(existing) >= self.max_frames_per_event:
            raise EvidenceFrameError("event frame limit reached")
        temporary = event_dir / f".{target.name}.{os.getpid()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, jpeg)
        finally:
            os.close(descriptor)
        os.replace(temporary, target)
        return StoredFrame(target, digest, len(jpeg), False)
