"""Bounded, compressed camera history for incident pre-roll.

This buffer is deliberately separate from the latest-frame capture slot.  The
viewer always reads the newest frame, while the low-cost watcher retains a
small JPEG history that can later be replayed for temporal inference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np


@dataclass(frozen=True)
class BufferedFrame:
    frame_seq: int
    mono_ts: float
    jpeg: bytes

    def decode(self) -> np.ndarray | None:
        data = np.frombuffer(self.jpeg, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)


class PreEventFrameBuffer:
    """Time-bounded JPEG ring sampled independently from heavy inference."""

    def __init__(
        self,
        *,
        duration_sec: float = 10.0,
        sample_hz: float = 10.0,
        frame_width: int = 320,
        jpeg_quality: int = 70,
    ):
        self.duration_sec = max(1.0, float(duration_sec))
        self.sample_hz = max(0.5, float(sample_hz))
        self.frame_width = max(64, int(frame_width))
        self.jpeg_quality = max(30, min(95, int(jpeg_quality)))
        # A little headroom prevents timestamp jitter from dropping the oldest
        # useful frame before time-based pruning runs.
        capacity = max(2, int(self.duration_sec * self.sample_hz) + 3)
        self._frames: deque[BufferedFrame] = deque(maxlen=capacity)
        self._lock = Lock()
        self._last_sample_ts: float | None = None
        self._encoded_total = 0
        self._encode_error_total = 0

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] <= self.frame_width:
            return frame
        scale = self.frame_width / float(frame.shape[1])
        height = max(1, int(round(frame.shape[0] * scale)))
        return cv2.resize(frame, (self.frame_width, height), interpolation=cv2.INTER_AREA)

    def append(self, frame: np.ndarray, *, frame_seq: int, mono_ts: float) -> bool:
        """Sample and compress a frame; return True only when retained."""
        now = float(mono_ts)
        with self._lock:
            last = self._last_sample_ts
            if last is not None and now <= last:
                return False
            if last is not None and now - last < (1.0 / self.sample_hz) * 0.90:
                return False

        resized = self._resize(frame)
        ok, encoded = cv2.imencode(
            ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            with self._lock:
                self._encode_error_total += 1
            return False

        item = BufferedFrame(int(frame_seq), now, encoded.tobytes())
        with self._lock:
            # Re-check after encoding in case another producer won the cadence.
            last = self._last_sample_ts
            if last is not None and now <= last:
                return False
            self._last_sample_ts = now
            self._frames.append(item)
            cutoff = now - self.duration_sec
            while self._frames and self._frames[0].mono_ts < cutoff:
                self._frames.popleft()
            self._encoded_total += 1
        return True

    def snapshot(
        self,
        *,
        end_mono_ts: float | None = None,
        duration_sec: float | None = None,
        max_frames: int | None = None,
    ) -> tuple[BufferedFrame, ...]:
        """Return an immutable chronological view without decoding images."""
        with self._lock:
            items = tuple(self._frames)
        if end_mono_ts is not None:
            end = float(end_mono_ts)
            items = tuple(item for item in items if item.mono_ts <= end)
        elif items:
            end = items[-1].mono_ts
        else:
            end = 0.0
        if duration_sec is not None:
            start = end - max(0.0, float(duration_sec))
            items = tuple(item for item in items if item.mono_ts >= start)
        if max_frames is not None and len(items) > int(max_frames):
            items = items[-int(max_frames):]
        return items

    def metrics(self) -> dict:
        with self._lock:
            items = tuple(self._frames)
            encoded_total = self._encoded_total
            errors = self._encode_error_total
        coverage = 0.0
        if len(items) >= 2:
            coverage = max(0.0, items[-1].mono_ts - items[0].mono_ts)
        return {
            "pre_event_frames": len(items),
            "pre_event_coverage_sec": coverage,
            "pre_event_bytes": sum(len(item.jpeg) for item in items),
            "pre_event_target_sec": self.duration_sec,
            "pre_event_sample_hz": self.sample_hz,
            "pre_event_encoded_total": encoded_total,
            "pre_event_encode_error_total": errors,
            "pre_event_ready": coverage >= min(3.0, self.duration_sec * 0.8),
        }
