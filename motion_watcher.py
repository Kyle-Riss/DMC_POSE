"""Independent low-cost motion watcher for latest-frame RTSP captures."""

from __future__ import annotations

from collections import deque
from threading import Event, Lock, Thread
import time
from typing import Callable

import cv2
import numpy as np

from latest_frame_capture import LatestFrameCapture
from pre_event_buffer import PreEventFrameBuffer


class MotionWatcher:
    """Watch a camera at high frequency without running a neural network.

    The watcher consumes the capture object's latest-frame slot independently
    from inference. Two consecutive frame-difference hits open a short BURST
    window. The inference loop can wait on ``wait_for_burst`` so it wakes
    immediately instead of finishing an EMPTY-state sleep.
    """

    def __init__(
        self,
        camera_id: str,
        capture: LatestFrameCapture,
        *,
        roi_provider: Callable[[], dict | None] | None = None,
        target_fps: float = 15.0,
        small_width: int = 160,
        small_height: int = 90,
        pixel_threshold: int = 22,
        motion_ratio_threshold: float = 0.018,
        max_motion_ratio: float = 0.70,
        consecutive_hits: int = 2,
        burst_hold_sec: float = 3.0,
        roi_margin_ratio: float = 0.20,
        pre_event_duration_sec: float = 10.0,
        pre_event_sample_hz: float = 10.0,
        pre_event_frame_width: int = 320,
        pre_event_jpeg_quality: int = 70,
        mono_clock: Callable[[], float] = time.monotonic,
    ):
        self.camera_id = str(camera_id)
        self.capture = capture
        self.roi_provider = roi_provider
        self.target_fps = max(1.0, float(target_fps))
        self.small_width = max(16, int(small_width))
        self.small_height = max(16, int(small_height))
        self.pixel_threshold = int(pixel_threshold)
        self.motion_ratio_threshold = float(motion_ratio_threshold)
        self.max_motion_ratio = float(max_motion_ratio)
        self.consecutive_hits = max(1, int(consecutive_hits))
        self.burst_hold_sec = max(0.1, float(burst_hold_sec))
        self.roi_margin_ratio = max(0.0, float(roi_margin_ratio))
        self._mono_clock = mono_clock
        self.pre_event_buffer = PreEventFrameBuffer(
            duration_sec=pre_event_duration_sec,
            sample_hz=pre_event_sample_hz,
            frame_width=pre_event_frame_width,
            jpeg_quality=pre_event_jpeg_quality,
        )

        self._lock = Lock()
        self._stop = Event()
        self._burst_event = Event()
        self._thread: Thread | None = None
        self._previous_gray: np.ndarray | None = None
        self._hit_streak = 0
        self._burst_until = 0.0
        self._motion_ratio = 0.0
        self._motion_detected = False
        self._frame_seq = 0
        self._processed_total = 0
        self._trigger_total = 0
        self._last_trigger_mono_ts: float | None = None
        self._process_times = deque(maxlen=180)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name=f"motion-watcher-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._burst_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def burst_active(self, now: float | None = None) -> bool:
        check_ts = self._mono_clock() if now is None else float(now)
        with self._lock:
            active = check_ts < self._burst_until
        if not active:
            self._burst_event.clear()
        return active

    def wait_for_burst(self, timeout: float) -> bool:
        """Return early when motion opens a BURST window."""
        if self.burst_active():
            return True
        self._burst_event.wait(max(0.0, float(timeout)))
        return self.burst_active()

    def _roi_bbox(self) -> tuple[int, int, int, int] | None:
        if self.roi_provider is None:
            return None
        try:
            roi = self.roi_provider()
        except Exception:
            return None
        if not roi:
            return None
        bbox = roi.get("bbox") if isinstance(roi, dict) else roi
        if bbox is None or len(bbox) != 4:
            return None
        return tuple(int(v) for v in bbox)

    def _prepare_gray(
        self,
        frame: np.ndarray,
        roi_bbox: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        fh, fw = frame.shape[:2]
        view = frame
        if roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            margin_x = int(round((x2 - x1) * self.roi_margin_ratio))
            margin_y = int(round((y2 - y1) * self.roi_margin_ratio))
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(fw, x2 + margin_x)
            y2 = min(fh, y2 + margin_y)
            if x2 > x1 and y2 > y1:
                view = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (self.small_width, self.small_height),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        frame_seq: int,
        mono_ts: float,
        roi_bbox: tuple[int, int, int, int] | None = None,
    ) -> dict:
        """Process one frame. Public for deterministic tests and diagnostics."""
        self.pre_event_buffer.append(
            frame, frame_seq=frame_seq, mono_ts=mono_ts
        )
        gray = self._prepare_gray(frame, roi_bbox)
        now = float(mono_ts)
        with self._lock:
            previous = self._previous_gray
            self._previous_gray = gray
            self._frame_seq = int(frame_seq)
            self._processed_total += 1
            self._process_times.append(now)

            if previous is None or previous.shape != gray.shape:
                self._hit_streak = 0
                self._motion_ratio = 0.0
                self._motion_detected = False
            else:
                diff = cv2.absdiff(previous, gray)
                changed = diff >= self.pixel_threshold
                ratio = float(np.count_nonzero(changed)) / float(changed.size)
                valid_hit = (
                    self.motion_ratio_threshold <= ratio <= self.max_motion_ratio
                )
                self._motion_ratio = ratio
                self._hit_streak = self._hit_streak + 1 if valid_hit else 0
                self._motion_detected = self._hit_streak >= self.consecutive_hits
                if self._motion_detected:
                    was_active = now < self._burst_until
                    self._burst_until = max(
                        self._burst_until, now + self.burst_hold_sec
                    )
                    if not was_active:
                        self._trigger_total += 1
                        self._last_trigger_mono_ts = now
                    self._burst_event.set()

        return self.status(now=now)

    def status(self, *, now: float | None = None) -> dict:
        check_ts = self._mono_clock() if now is None else float(now)
        with self._lock:
            times = tuple(self._process_times)
            burst_until = self._burst_until
            result = {
                "watcher_fps": 0.0,
                "motion_ratio": float(self._motion_ratio),
                "motion_detected": bool(self._motion_detected),
                "motion_hit_streak": int(self._hit_streak),
                "burst_active": bool(check_ts < burst_until),
                "burst_remaining_ms": max(
                    0.0, (burst_until - check_ts) * 1000.0
                ),
                "watcher_frame_seq": int(self._frame_seq),
                "watcher_processed_total": int(self._processed_total),
                "motion_trigger_total": int(self._trigger_total),
                "last_motion_trigger_mono_ts": self._last_trigger_mono_ts,
                "watcher_thread_alive": self.is_alive(),
            }
        if len(times) >= 2 and times[-1] > times[0]:
            result["watcher_fps"] = (len(times) - 1) / (times[-1] - times[0])
        result.update(self.pre_event_buffer.metrics())
        return result

    def pre_event_snapshot(
        self, *, end_mono_ts: float | None = None,
        duration_sec: float | None = None, max_frames: int | None = None,
    ):
        return self.pre_event_buffer.snapshot(
            end_mono_ts=end_mono_ts,
            duration_sec=duration_sec,
            max_frames=max_frames,
        )

    def _run(self) -> None:
        # Process every newly published latest frame. RTSP decoders may deliver
        # frames in short bursts; wall-clock throttling here would discard the
        # very transitions needed for fast fall-motion detection.
        last_seq = 0
        while not self._stop.is_set():
            packet = self.capture.wait_for_frame(
                last_seq, timeout=0.5, copy_frame=False
            )
            if packet is None:
                continue
            last_seq = packet.frame_seq
            self.process_frame(
                packet.frame,
                frame_seq=packet.frame_seq,
                mono_ts=packet.capture_mono_ts,
                roi_bbox=self._roi_bbox(),
            )
