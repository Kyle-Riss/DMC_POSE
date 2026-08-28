"""Bounded candidate-only RGB verifier runtime; never opens a camera stream."""

from __future__ import annotations

import time
from threading import Lock, Thread

from pre_event_buffer import PreEventFrameBuffer
from swin3d_verifier import Swin3DVerifierService, progressive_decision


def verifier_trigger_active(*, rapid_motion: bool, fusion_phase: str) -> bool:
    """Candidate RGB verification never depends on temporal-model readiness."""
    return bool(rapid_motion) or str(fusion_phase) in {
        "CANDIDATE", "VERIFY", "SHADOW_ALERT"
    }


class CandidateVideoVerifierRuntime:
    def __init__(
        self,
        camera_id: str,
        service: Swin3DVerifierService,
        *,
        duration_sec: float = 10.0,
        sample_hz: float = 5.0,
        frame_width: int = 320,
        absolute_threshold: float = 0.439,
        delta_threshold: float = 0.10,
        rearm_sec: float = 10.0,
    ):
        self.camera_id = camera_id
        self.service = service
        self.buffer = PreEventFrameBuffer(
            duration_sec=duration_sec, sample_hz=sample_hz,
            frame_width=frame_width, jpeg_quality=70,
        )
        self.absolute_threshold = float(absolute_threshold)
        self.delta_threshold = float(delta_threshold)
        self.rearm_sec = float(rearm_sec)
        self._lock = Lock()
        self._thread: Thread | None = None
        self._result = {
            "enabled": True, "running": False, "ready": False,
            "candidate": False, "baseline": None, "post_max": None,
            "delta": None, "pair_probability": None,
            "decision_mode": str(getattr(service, "feature_mode", "single_embedding_v1")),
            "threshold": float(getattr(service, "threshold", absolute_threshold)),
            "latency_ms": 0.0, "trigger_total": 0,
            "completed_total": 0, "error_total": 0, "last_error": None,
        }
        self._previous_trigger = False
        self._next_allowed_at = 0.0

    def observe(self, frame, *, frame_seq: int, mono_ts: float) -> bool:
        return self.buffer.append(frame, frame_seq=frame_seq, mono_ts=mono_ts)

    def _snapshot_frames(self, end_ts: float, duration_sec: float = 4.0):
        items = self.buffer.snapshot(end_mono_ts=end_ts, duration_sec=duration_sec)
        return [frame for item in items if (frame := item.decode()) is not None]

    def _run(self, trigger_ts: float) -> None:
        started = time.perf_counter()
        try:
            baseline_frames = self._snapshot_frames(trigger_ts - 4.0)
            post_frames = self._snapshot_frames(trigger_ts)
            if getattr(self.service, "feature_mode", "single_embedding_v1") == "delta_embedding_v1":
                pair = self.service.predict_pair(baseline_frames, post_frames)
                decision = {
                    "ready": True,
                    "candidate": pair.probability >= self.service.threshold,
                    "baseline": None,
                    "post_max": None,
                    "delta": None,
                    "pair_probability": pair.probability,
                    "decision_mode": "delta_embedding_v1",
                    "threshold": self.service.threshold,
                }
            else:
                baseline = self.service.predict(baseline_frames)
                post = self.service.predict(post_frames)
                decision = progressive_decision(
                    baseline.probability, [post.probability],
                    absolute_threshold=self.absolute_threshold,
                    delta_threshold=self.delta_threshold,
                )
                decision.update({
                    "pair_probability": post.probability,
                    "decision_mode": "single_embedding_v1",
                    "threshold": self.absolute_threshold,
                })
            update = {
                **decision,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "baseline_frames": len(baseline_frames),
                "post_frames": len(post_frames),
                "last_error": None,
            }
            with self._lock:
                self._result.update(update)
                self._result["completed_total"] += 1
        except Exception as exc:
            with self._lock:
                self._result.update({
                    "ready": False, "candidate": False,
                    "pair_probability": None,
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "error_total": self._result["error_total"] + 1,
                    "last_error": f"{type(exc).__name__}: {exc}",
                })
        finally:
            with self._lock:
                self._result["running"] = False

    def update_trigger(self, active: bool, *, mono_ts: float) -> bool:
        active = bool(active)
        if not active:
            self._previous_trigger = False
            return False
        rising = not self._previous_trigger
        now = time.monotonic()
        with self._lock:
            busy = bool(self._result["running"])
        if not rising or busy or now < self._next_allowed_at:
            return False
        if not self.buffer.metrics()["pre_event_ready"]:
            return False
        self._previous_trigger = True
        with self._lock:
            self._result["running"] = True
            self._result["trigger_total"] += 1
        self._next_allowed_at = now + self.rearm_sec
        self._thread = Thread(target=self._run, args=(float(mono_ts),), daemon=True, name=f"video-verify-{self.camera_id}")
        self._thread.start()
        return True

    def status(self) -> dict:
        with self._lock:
            result = dict(self._result)
        result.update({f"ring_{key}": value for key, value in self.buffer.metrics().items()})
        result["authority"] = "telemetry_only"
        result["absolute_threshold"] = self.absolute_threshold
        result["delta_threshold"] = self.delta_threshold
        return result

    def close(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
