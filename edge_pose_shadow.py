"""Motion-triggered, local-only Pose shadow runner for constrained edge nodes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from threading import Event, Lock, Thread
import time
from typing import Callable

import numpy as np

from edge_pose_onnx import EdgePoseONNX


class EdgePoseShadow:
    """Run one Pose inference whenever the motion watcher enters a new burst."""

    def __init__(self, watcher, *, rtsp_url: str, model_path: str,
                 ffmpeg_path: str = "/usr/bin/ffmpeg", width: int = 640,
                 height: int = 360, input_size: int = 320,
                 confidence: float = 0.25, rotation_degrees: int = 90,
                 capture_timeout_sec: float = 5.0,
                 poll_interval_sec: float = 0.05,
                 status_path: str = "runtime_data/edge_pose_shadow/status.json",
                 log_path: str = "runtime_data/edge_pose_shadow/events.jsonl",
                 on_result: Callable[[dict], None] | None = None,
        wall_clock: Callable[[], float] = time.time):
        if rotation_degrees not in (0, 90, 180, 270):
            raise ValueError("rotation_degrees must be 0, 90, 180, or 270")
        self.watcher = watcher
        self.rtsp_url = str(rtsp_url)
        self.model_path = str(model_path)
        self.ffmpeg_path = str(ffmpeg_path)
        self.width = int(width)
        self.height = int(height)
        self.input_size = int(input_size)
        self.confidence = float(confidence)
        self.rotation_degrees = int(rotation_degrees)
        self.capture_timeout_sec = float(capture_timeout_sec)
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))
        self.status_path = Path(status_path).expanduser()
        self.log_path = Path(log_path).expanduser()
        self._wall_clock = wall_clock
        self.on_result = on_result
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._model: EdgePoseONNX | None = None
        self._status = {
            "enabled": True, "thread_alive": False, "inference_total": 0,
            "error_total": 0, "last_trigger_total": 0,
            "last_detection_count": 0, "last_visible_keypoints": 0,
            "last_person_score": 0.0, "last_total_ms": 0.0,
            "last_error": "", "last_event_unix": None,
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._status["last_trigger_total"] = int(self.watcher.status().get("trigger_total", 0))
        self._thread = Thread(target=self._run, name="edge-pose-shadow", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 6.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        with self._lock:
            result = dict(self._status)
        result["thread_alive"] = self.is_alive()
        return result

    def _capture_rgb(self) -> tuple[np.ndarray, float]:
        latest_snapshot = getattr(self.watcher, "latest_rgb_snapshot", None)
        if callable(latest_snapshot):
            snapshot = latest_snapshot()
            if snapshot is not None:
                return snapshot
        latest_rgb = getattr(self.watcher, "latest_rgb", None)
        if callable(latest_rgb):
            frame = latest_rgb()
            if frame is not None:
                return frame, self._wall_clock()
        command = [self.ffmpeg_path, "-hide_banner", "-loglevel", "error",
                   "-rtsp_transport", "tcp", "-i", self.rtsp_url,
                   "-frames:v", "1", "-vf", f"scale={self.width}:{self.height}",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        completed = subprocess.run(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   timeout=self.capture_timeout_sec, check=False)
        expected = self.width * self.height * 3
        if completed.returncode != 0 or len(completed.stdout) != expected:
            detail = completed.stderr.decode("utf-8", errors="replace")[-240:]
            raise RuntimeError(f"snapshot failed rc={completed.returncode} bytes={len(completed.stdout)}/{expected}: {detail}")
        frame = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(self.height, self.width, 3)
        return frame, self._wall_clock()

    def _rotate(self, rgb: np.ndarray) -> np.ndarray:
        if self.rotation_degrees == 0:
            return rgb
        return np.rot90(rgb, k={90: 3, 180: 2, 270: 1}[self.rotation_degrees]).copy()

    def _persist(self, event: dict) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.status_path)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    def run_once(self, trigger_total: int) -> dict:
        started = time.perf_counter()
        if self._model is None:
            self._model = EdgePoseONNX(self.model_path, input_size=self.input_size)
        captured_rgb, captured_unix = self._capture_rgb()
        rgb = self._rotate(captured_rgb)
        result = self._model.infer(rgb, confidence=self.confidence)
        detections = result["detections"]
        best = max(detections, key=lambda item: item["score"], default=None)
        event = {
            "schema_version": "edge-pose-shadow-v1",
            "event_unix": self._wall_clock(), "captured_unix": captured_unix,
            "trigger_total": int(trigger_total),
            "rotation_degrees": self.rotation_degrees,
            "confidence_threshold": self.confidence,
            "detection_count": len(detections),
            "best_person_score": float(best["score"]) if best else 0.0,
            "best_visible_keypoints": int(best["visible_keypoints"]) if best else 0,
            "pose_inference_ms": float(result["total_ms"]),
            "snapshot_and_pose_ms": (time.perf_counter() - started) * 1000,
            "shadow_only": True,
        }
        self._persist(event)
        if self.on_result is not None:
            self.on_result(event)
        return event

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_sec):
            trigger_total = int(self.watcher.status().get("trigger_total", 0))
            with self._lock:
                previous = int(self._status["last_trigger_total"])
            if trigger_total <= previous:
                continue
            try:
                event = self.run_once(trigger_total)
                update = {
                    "inference_total": int(self._status["inference_total"]) + 1,
                    "last_detection_count": event["detection_count"],
                    "last_visible_keypoints": event["best_visible_keypoints"],
                    "last_person_score": event["best_person_score"],
                    "last_total_ms": event["snapshot_and_pose_ms"],
                    "last_error": "", "last_event_unix": event["event_unix"],
                }
            except Exception as exc:
                update = {"error_total": int(self._status["error_total"]) + 1,
                          "last_error": str(exc)[:500],
                          "last_event_unix": self._wall_clock()}
            with self._lock:
                self._status.update(update)
                self._status["last_trigger_total"] = trigger_total
