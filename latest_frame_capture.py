"""Continuously drain one RTSP stream and expose only its newest valid frame."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from threading import Condition, Event, Lock, Thread
import time
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    """One immutable frame envelope.

    The ndarray itself is copied when a consumer reads it, so the capture thread
    can replace the latest envelope without coordinating with inference/viewer.
    """

    camera_id: str
    frame_seq: int
    frame: np.ndarray
    capture_mono_ts: float
    capture_wall_ts: float


class LatestFrameCapture:
    """A single-slot, latest-frame-only RTSP capture.

    This class intentionally has no frame queue. Slow consumers skip intermediate
    frames and receive the newest available sequence on their next read.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        *,
        frame_width: int | None = None,
        reconnect_delay_sec: float = 2.0,
        capture_factory: Callable[[str], object] | None = None,
        mono_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.camera_id = str(camera_id)
        self.rtsp_url = str(rtsp_url)
        self.frame_width = int(frame_width) if frame_width else None
        self.reconnect_delay_sec = max(0.01, float(reconnect_delay_sec))
        self._capture_factory = capture_factory or self._default_capture_factory
        self._mono_clock = mono_clock
        self._wall_clock = wall_clock

        self._condition = Condition(Lock())
        self._stop = Event()
        self._thread: Thread | None = None
        self._active_capture = None
        self._latest: CapturedFrame | None = None
        self._frame_seq = 0
        self._frame_times = deque(maxlen=120)
        self._connected = False
        self._decode_error_total = 0
        self._reconnect_total = 0
        self._last_error = ""

    @staticmethod
    def _default_capture_factory(rtsp_url: str):
        return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = Thread(
                target=self._run,
                name=f"capture-{self.camera_id}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        with self._condition:
            active = self._active_capture
            self._condition.notify_all()
        if active is not None:
            try:
                active.release()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))

    def is_alive(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def publish(
        self,
        frame: np.ndarray,
        *,
        capture_mono_ts: float | None = None,
        capture_wall_ts: float | None = None,
    ) -> CapturedFrame:
        """Publish a valid frame into the single latest-frame slot."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        mono_ts = float(self._mono_clock() if capture_mono_ts is None else capture_mono_ts)
        wall_ts = float(self._wall_clock() if capture_wall_ts is None else capture_wall_ts)
        with self._condition:
            self._frame_seq += 1
            envelope = CapturedFrame(
                camera_id=self.camera_id,
                frame_seq=self._frame_seq,
                frame=frame,
                capture_mono_ts=mono_ts,
                capture_wall_ts=wall_ts,
            )
            self._latest = envelope
            self._frame_times.append(mono_ts)
            self._connected = True
            self._last_error = ""
            self._condition.notify_all()
            return envelope

    @staticmethod
    def _copy_envelope(envelope: CapturedFrame) -> CapturedFrame:
        return CapturedFrame(
            camera_id=envelope.camera_id,
            frame_seq=envelope.frame_seq,
            frame=envelope.frame.copy(),
            capture_mono_ts=envelope.capture_mono_ts,
            capture_wall_ts=envelope.capture_wall_ts,
        )

    def latest(self, *, copy_frame: bool = True) -> CapturedFrame | None:
        with self._condition:
            envelope = self._latest
            if envelope is None:
                return None
            return self._copy_envelope(envelope) if copy_frame else envelope

    def wait_for_frame(
        self,
        after_seq: int = 0,
        *,
        timeout: float | None = None,
        copy_frame: bool = True,
    ) -> CapturedFrame | None:
        """Wait until a sequence newer than ``after_seq`` exists.

        If several frames arrived while the consumer was busy, only the newest
        sequence is returned.
        """
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._latest is not None
                    and self._latest.frame_seq > int(after_seq)
                )
                or self._stop.is_set(),
                timeout=timeout,
            )
            if not ready or self._latest is None or self._latest.frame_seq <= int(after_seq):
                return None
            return self._copy_envelope(self._latest) if copy_frame else self._latest

    def metrics(self) -> dict:
        now = self._mono_clock()
        with self._condition:
            times = tuple(self._frame_times)
            latest = self._latest
            connected = self._connected
            decode_error_total = self._decode_error_total
            reconnect_total = self._reconnect_total
            last_error = self._last_error
        capture_fps = 0.0
        if len(times) >= 2 and times[-1] > times[0]:
            capture_fps = (len(times) - 1) / (times[-1] - times[0])
        frame_age_ms = None
        frame_seq = 0
        if latest is not None:
            frame_seq = latest.frame_seq
            frame_age_ms = max(0.0, (now - latest.capture_mono_ts) * 1000.0)
        return {
            "connected": connected,
            "capture_fps": float(capture_fps),
            "frame_seq": int(frame_seq),
            "frame_age_ms": frame_age_ms,
            "decode_error_total": int(decode_error_total),
            "reconnect_total": int(reconnect_total),
            "last_error": last_error,
            "thread_alive": self.is_alive(),
        }

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if self.frame_width is None or frame.shape[1] == self.frame_width:
            return frame
        scale = self.frame_width / float(frame.shape[1])
        height = max(1, int(round(frame.shape[0] * scale)))
        return cv2.resize(frame, (self.frame_width, height), interpolation=cv2.INTER_AREA)

    def _mark_disconnected(self, error: str) -> None:
        with self._condition:
            self._connected = False
            self._last_error = str(error)
            self._condition.notify_all()

    def _run(self) -> None:
        first_open = True
        while not self._stop.is_set():
            try:
                capture = self._capture_factory(self.rtsp_url)
            except Exception as exc:
                self._mark_disconnected(f"open failed: {exc}")
                if not first_open:
                    with self._condition:
                        self._reconnect_total += 1
                first_open = False
                self._stop.wait(self.reconnect_delay_sec)
                continue

            with self._condition:
                self._active_capture = capture
                if not first_open:
                    self._reconnect_total += 1
            first_open = False

            try:
                if capture is None or not capture.isOpened():
                    self._mark_disconnected("RTSP open failed")
                    self._stop.wait(self.reconnect_delay_sec)
                    continue

                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.size == 0:
                        with self._condition:
                            self._decode_error_total += 1
                        self._mark_disconnected("RTSP read failed")
                        break
                    self.publish(self._resize(frame))
            except Exception as exc:
                logging.warning("[%s] capture error: %s", self.camera_id, exc)
                with self._condition:
                    self._decode_error_total += 1
                self._mark_disconnected(str(exc))
            finally:
                try:
                    if capture is not None:
                        capture.release()
                except Exception:
                    pass
                with self._condition:
                    if self._active_capture is capture:
                        self._active_capture = None

            if not self._stop.is_set():
                self._stop.wait(self.reconnect_delay_sec)

        self._mark_disconnected("stopped")

