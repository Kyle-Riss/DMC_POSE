"""Low-cost fixed-camera site telemetry for Raspberry Pi camera nodes.

This module never classifies a fall.  It monitors whether the camera scene is
still calibrated, whether a fixed ROI profile remains usable, and whether an
external encoded-segment recorder is healthy.  It consumes the RGB snapshot
already produced by :mod:`edge_motion_watcher` and opens no additional RTSP
connection.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np


SCENE_STATES = {"UNCALIBRATED", "STABLE", "CHANGED"}


def _gray_thumbnail(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("RGB frame must have shape HxWx3")
    rows = np.linspace(0, frame.shape[0] - 1, height).astype(int)
    cols = np.linspace(0, frame.shape[1] - 1, width).astype(int)
    sampled = frame[rows[:, None], cols[None, :]]
    gray = (
        sampled[:, :, 0].astype(np.uint16) * 77
        + sampled[:, :, 1].astype(np.uint16) * 150
        + sampled[:, :, 2].astype(np.uint16) * 29
    ) >> 8
    return gray.astype(np.uint8)


class EdgeSceneGuard:
    """Persistent, debounced camera-relocation detector."""

    def __init__(
        self,
        reference_path: str | Path,
        *,
        width: int = 80,
        height: int = 45,
        pixel_threshold: int = 35,
        change_ratio: float = 0.65,
        persistence: int = 3,
    ) -> None:
        self.reference_path = Path(reference_path).expanduser()
        self.width = max(16, int(width))
        self.height = max(16, int(height))
        self.pixel_threshold = max(1, int(pixel_threshold))
        self.change_ratio = min(1.0, max(0.01, float(change_ratio)))
        self.persistence = max(1, int(persistence))
        self._lock = Lock()
        self._reference: np.ndarray | None = None
        self._state = "UNCALIBRATED"
        self._score = 0.0
        self._change_streak = 0
        self._observed_total = 0
        self._changed_total = 0
        self._last_observed_wall: float | None = None
        self._load()

    def _load(self) -> None:
        try:
            payload = self.reference_path.read_bytes()
        except OSError:
            return
        expected = self.width * self.height
        if len(payload) != expected:
            return
        self._reference = np.frombuffer(payload, dtype=np.uint8).reshape(
            self.height, self.width
        ).copy()
        self._state = "STABLE"

    def calibrate(self, rgb: np.ndarray) -> dict:
        reference = _gray_thumbnail(rgb, self.width, self.height)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.reference_path.with_suffix(self.reference_path.suffix + ".tmp")
        temporary.write_bytes(reference.tobytes())
        os.replace(temporary, self.reference_path)
        with self._lock:
            self._reference = reference
            self._state = "STABLE"
            self._score = 0.0
            self._change_streak = 0
        return self.status()

    def observe(self, rgb: np.ndarray, *, wall_ts: float | None = None) -> dict:
        current = _gray_thumbnail(rgb, self.width, self.height)
        now = time.time() if wall_ts is None else float(wall_ts)
        with self._lock:
            self._observed_total += 1
            self._last_observed_wall = now
            if self._reference is None:
                self._state = "UNCALIBRATED"
                return self._status_locked()
            changed = (
                np.abs(current.astype(np.int16) - self._reference.astype(np.int16))
                >= self.pixel_threshold
            )
            self._score = float(np.count_nonzero(changed)) / float(changed.size)
            if self._score >= self.change_ratio:
                self._change_streak += 1
            else:
                self._change_streak = 0
            if self._change_streak >= self.persistence and self._state != "CHANGED":
                self._state = "CHANGED"
                self._changed_total += 1
            return self._status_locked()

    def _status_locked(self) -> dict:
        return {
            "state": self._state,
            "calibrated": self._reference is not None,
            "change_score": self._score,
            "change_streak": self._change_streak,
            "observed_total": self._observed_total,
            "changed_total": self._changed_total,
            "last_observed_wall": self._last_observed_wall,
        }

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()


class EdgeROIProfile:
    """Validate one normalized, fixed-camera ROI profile."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path).expanduser() if path else None
        self.valid = False
        self.version = 0
        self.source = "none"
        self.polygon: list[list[float]] = []
        self.error = "roi_profile_missing"
        self.reload()

    def reload(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            polygon = payload["polygon_norm"]
            if len(polygon) < 3:
                raise ValueError("polygon needs at least three points")
            normalized = [[float(point[0]), float(point[1])] for point in polygon]
            if any(value < 0.0 or value > 1.0 for point in normalized for value in point):
                raise ValueError("polygon coordinates must be normalized")
            self.polygon = normalized
            self.version = max(1, int(payload.get("version", 1)))
            self.source = str(payload.get("source", "fixed_profile"))
            self.valid = True
            self.error = ""
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.valid = False
            self.error = f"roi_profile_invalid:{type(exc).__name__}"

    def status(self, scene_state: str) -> dict:
        if not self.valid:
            state = "UNAVAILABLE"
        elif scene_state == "CHANGED":
            state = "DEGRADED"
        elif scene_state == "UNCALIBRATED":
            state = "BOOTSTRAP"
        else:
            state = "READY"
        return {
            "state": state,
            "valid": self.valid and scene_state == "STABLE",
            "version": self.version,
            "source": self.source,
            "point_count": len(self.polygon),
            "error": self.error,
        }


class EncodedRingMonitor:
    """Read-only health monitor for MediaMTX/ffmpeg encoded segments."""

    def __init__(
        self,
        directory: str | Path,
        *,
        segment_duration_sec: float = 2.0,
        maximum_segment_age_sec: float = 10.0,
        suffixes: tuple[str, ...] = (".mp4", ".ts", ".mkv"),
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.directory = Path(directory).expanduser()
        self.segment_duration_sec = max(0.1, float(segment_duration_sec))
        self.maximum_segment_age_sec = max(0.1, float(maximum_segment_age_sec))
        self.suffixes = {suffix.lower() for suffix in suffixes}
        self._wall_clock = wall_clock

    def status(self) -> dict:
        try:
            files = [
                path for path in self.directory.iterdir()
                if path.is_file() and path.suffix.lower() in self.suffixes
            ]
        except OSError:
            files = []
        stats = []
        for path in files:
            try:
                item = path.stat()
            except OSError:
                continue
            stats.append(item)
        if not stats:
            return {
                "ready": False,
                "segments": 0,
                "bytes": 0,
                "coverage_sec": 0.0,
                "newest_age_sec": None,
            }
        mtimes = [item.st_mtime for item in stats]
        newest_age = max(0.0, self._wall_clock() - max(mtimes))
        coverage = max(mtimes) - min(mtimes) + self.segment_duration_sec
        return {
            "ready": newest_age <= self.maximum_segment_age_sec,
            "segments": len(stats),
            "bytes": sum(item.st_size for item in stats),
            "coverage_sec": max(0.0, coverage),
            "newest_age_sec": newest_age,
        }


class EdgeSiteRuntime:
    """Join scene, ROI and encoded-ring facts without opening RTSP."""

    def __init__(self, config: dict, motion_watcher) -> None:
        self.motion_watcher = motion_watcher
        scene_config = dict(config.get("scene_guard", {}))
        self.scene_guard = EdgeSceneGuard(**scene_config) if scene_config else None
        self.roi = EdgeROIProfile(config.get("roi_profile_path"))
        ring_config = dict(config.get("encoded_ring", {}))
        self.ring = EncodedRingMonitor(**ring_config) if ring_config else None
        self._last_rgb_wall = 0.0

    def calibrate_from_latest(self) -> dict:
        if self.scene_guard is None:
            raise RuntimeError("scene_guard is disabled")
        snapshot = self.motion_watcher.latest_rgb_snapshot()
        if snapshot is None:
            raise RuntimeError("no RGB snapshot is available")
        rgb, _ = snapshot
        return self.scene_guard.calibrate(rgb)

    def refresh(self) -> dict:
        motion = self.motion_watcher.status()
        if self.scene_guard is None:
            scene = {"state": "UNCALIBRATED", "change_score": 0.0}
        else:
            snapshot = self.motion_watcher.latest_rgb_snapshot()
            if snapshot is not None and snapshot[1] > self._last_rgb_wall:
                rgb, wall_ts = snapshot
                self._last_rgb_wall = wall_ts
                scene = self.scene_guard.observe(rgb, wall_ts=wall_ts)
            else:
                scene = self.scene_guard.status()
        roi = self.roi.status(str(scene["state"]))
        ring = self.ring.status() if self.ring else {
            "ready": False, "segments": 0, "bytes": 0,
            "coverage_sec": 0.0, "newest_age_sec": None,
        }
        return {
            "motion_ratio": float(motion.get("motion_ratio", 0.0)),
            "motion_active": bool(motion.get("burst_active", False)),
            "scene_state": str(scene["state"]),
            "scene_change_score": float(scene.get("change_score", 0.0)),
            "roi_state": roi["state"],
            "roi_version": int(roi["version"]),
            "roi_source": str(roi["source"]),
            "ring_buffer_ready": bool(ring["ready"]),
            "ring_buffer_segments": int(ring["segments"]),
            "ring_buffer_bytes": int(ring["bytes"]),
            "ring_buffer_coverage_sec": float(ring["coverage_sec"]),
        }
