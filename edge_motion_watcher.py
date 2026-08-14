"""Low-cost ffmpeg motion watcher for a local RTSP relay."""
from __future__ import annotations

from collections import deque
import subprocess
from threading import Event, Lock, Thread
import time
from typing import Callable

import numpy as np


class EdgeMotionWatcher:
    def __init__(
        self,
        rtsp_url: str,
        *,
        ffmpeg_path: str = "/usr/bin/ffmpeg",
        target_fps: float = 5.0,
        width: int = 160,
        height: int = 90,
        pixel_threshold: int = 22,
        motion_ratio_threshold: float = 0.018,
        max_motion_ratio: float = 0.70,
        consecutive_hits: int = 2,
        burst_hold_sec: float = 3.0,
        retain_rgb: bool = False,
        rgb_width: int = 320,
        rgb_height: int = 180,
        mono_clock: Callable[[], float] = time.monotonic,
    ):
        self.rtsp_url = str(rtsp_url)
        self.ffmpeg_path = str(ffmpeg_path)
        self.target_fps = max(1.0, float(target_fps))
        self.width = max(16, int(width))
        self.height = max(16, int(height))
        self.pixel_threshold = max(1, int(pixel_threshold))
        self.motion_ratio_threshold = max(0.0, float(motion_ratio_threshold))
        self.max_motion_ratio = min(1.0, float(max_motion_ratio))
        self.consecutive_hits = max(1, int(consecutive_hits))
        self.burst_hold_sec = max(0.1, float(burst_hold_sec))
        self.retain_rgb = bool(retain_rgb)
        self.rgb_width = max(self.width, int(rgb_width))
        self.rgb_height = max(self.height, int(rgb_height))
        self._mono_clock = mono_clock
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._process: subprocess.Popen | None = None
        self._previous: np.ndarray | None = None
        self._times = deque(maxlen=180)
        self._hit_streak = 0
        self._motion_ratio = 0.0
        self._burst_until = 0.0
        self._processed_total = 0
        self._trigger_total = 0
        self._restart_total = 0
        self._connected = False
        self._latest_rgb: np.ndarray | None = None
        self._latest_rgb_ts = 0.0
        self._latest_rgb_wall_ts = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="edge-motion-watcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        process = self._process
        if process and process.poll() is None:
            process.terminate()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))
        if process and process.poll() is None:
            process.kill()

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def process_gray(self, frame: np.ndarray, *, mono_ts: float) -> dict:
        gray = np.asarray(frame, dtype=np.uint8)
        if gray.shape != (self.height, self.width):
            raise ValueError(f"expected {(self.height, self.width)}, got {gray.shape}")
        now = float(mono_ts)
        with self._lock:
            previous = self._previous
            self._previous = gray.copy()
            self._processed_total += 1
            self._times.append(now)
            if previous is None:
                self._motion_ratio = 0.0
                self._hit_streak = 0
            else:
                diff = np.abs(gray.astype(np.int16) - previous.astype(np.int16))
                ratio = float(np.count_nonzero(diff >= self.pixel_threshold)) / diff.size
                valid = self.motion_ratio_threshold <= ratio <= self.max_motion_ratio
                self._motion_ratio = ratio
                self._hit_streak = self._hit_streak + 1 if valid else 0
                if self._hit_streak >= self.consecutive_hits:
                    was_active = now < self._burst_until
                    self._burst_until = max(self._burst_until, now + self.burst_hold_sec)
                    if not was_active:
                        self._trigger_total += 1
        return self.status(now=now)

    def status(self, *, now: float | None = None) -> dict:
        check = self._mono_clock() if now is None else float(now)
        with self._lock:
            times = tuple(self._times)
            result = {
                "connected": self._connected,
                "watcher_fps": 0.0,
                "motion_ratio": self._motion_ratio,
                "motion_hit_streak": self._hit_streak,
                "burst_active": check < self._burst_until,
                "processed_total": self._processed_total,
                "trigger_total": self._trigger_total,
                "restart_total": self._restart_total,
                "thread_alive": self.is_alive(),
                "latest_rgb_age_ms": (
                    max(0.0, check - self._latest_rgb_ts) * 1000
                    if self._latest_rgb is not None else None
                ),
            }
        if len(times) >= 2 and times[-1] > times[0]:
            result["watcher_fps"] = (len(times) - 1) / (times[-1] - times[0])
        return result

    def latest_rgb(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_rgb is None else self._latest_rgb.copy()

    def latest_rgb_snapshot(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if self._latest_rgb is None:
                return None
            return self._latest_rgb.copy(), float(self._latest_rgb_wall_ts)

    def _command(self) -> list[str]:
        if self.retain_rgb:
            video_filter = f"fps={self.target_fps},scale={self.rgb_width}:{self.rgb_height},format=rgb24"
            pixel_format = "rgb24"
        else:
            video_filter = f"fps={self.target_fps},scale={self.width}:{self.height},format=gray"
            pixel_format = "gray"
        return [
            self.ffmpeg_path, "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-i", self.rtsp_url,
            "-an", "-vf", video_filter,
            "-f", "rawvideo", "-pix_fmt", pixel_format, "pipe:1",
        ]

    @staticmethod
    def _read_exact(stream, size: int) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = stream.read(size - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks)

    def _run(self) -> None:
        frame_bytes = (
            self.rgb_width * self.rgb_height * 3
            if self.retain_rgb else self.width * self.height
        )
        while not self._stop.is_set():
            process = subprocess.Popen(
                self._command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=frame_bytes * 2,
            )
            self._process = process
            with self._lock:
                self._connected = True
                self._previous = None
            try:
                while not self._stop.is_set() and process.stdout is not None:
                    payload = self._read_exact(process.stdout, frame_bytes)
                    if payload is None:
                        break
                    now = self._mono_clock()
                    if self.retain_rgb:
                        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(
                            self.rgb_height, self.rgb_width, 3
                        )
                        rows = np.linspace(0, self.rgb_height - 1, self.height).astype(int)
                        cols = np.linspace(0, self.rgb_width - 1, self.width).astype(int)
                        sampled = rgb[rows[:, None], cols[None, :]]
                        gray = (
                            sampled[:, :, 0].astype(np.uint16) * 77
                            + sampled[:, :, 1].astype(np.uint16) * 150
                            + sampled[:, :, 2].astype(np.uint16) * 29
                        ) >> 8
                        with self._lock:
                            self._latest_rgb = rgb.copy()
                            self._latest_rgb_ts = now
                            self._latest_rgb_wall_ts = time.time()
                        self.process_gray(gray.astype(np.uint8), mono_ts=now)
                    else:
                        frame = np.frombuffer(payload, dtype=np.uint8).reshape(self.height, self.width)
                        self.process_gray(frame, mono_ts=now)
            finally:
                with self._lock:
                    self._connected = False
                    if not self._stop.is_set():
                        self._restart_total += 1
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                self._process = None
            if not self._stop.wait(1.0):
                continue

