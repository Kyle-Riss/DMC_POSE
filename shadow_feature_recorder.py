"""Asynchronous feature-only runtime recorder.

The recorder intentionally rejects images, keypoints, and RTSP URLs. It stores
only the small status fields needed to measure false alarms per bed-hour.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
import time
from typing import Any

ALLOWED_FIELDS = (
    "camera_id", "timestamp", "runtime_mode", "bed_roi_ready",
    "body_in_bed_ratio", "motion_ratio", "motion_detected", "burst_active",
    "person_count", "track_count", "primary_track_id",
    "primary_track_observed", "primary_track_bed_overlap",
    "primary_track_confidence", "pose", "pose_conf", "fall_score",
    "tcn_shadow_ready", "tcn_fall_probability", "tcn_alert_candidate",
    "tcn_missing_samples_window", "tcn_track_id", "tcn_source",
    "tcn_replay_ready", "tcn_replay_probability", "tcn_replay_candidate",
    "tcn_replay_samples", "tcn_replay_requested_frames",
    "tcn_replay_observed_frames", "tcn_replay_attempt_total",
    "tcn_replay_completed_total", "tcn_replay_error_total",
    "tcn_replay_elapsed_ms", "tcn_replay_reason", "fusion_phase",
    "fusion_risk", "fusion_evidence", "fusion_safe_evidence",
    "fusion_quality", "fusion_track_id", "fusion_policy_version",
    "capture_connected", "capture_fps",
    "capture_decode_error_total", "scheduler_queue_latency_ms",
    "scheduler_stale_drop_total", "scheduler_superseded_drop_total",
    "pre_event_frames", "pre_event_coverage_sec", "pre_event_bytes",
    "pre_event_ready",
)


class ShadowFeatureRecorder:
    """Write throttled status snapshots without blocking inference threads."""

    def __init__(self, output_dir: Path, *, sample_interval_sec: float = 0.5,
                 queue_size: int = 2048, flush_interval_sec: float = 1.0):
        self.output_dir = Path(output_dir)
        self.sample_interval_sec = max(0.1, float(sample_interval_sec))
        self.flush_interval_sec = max(0.1, float(flush_interval_sec))
        self.queue: Queue[dict[str, Any] | None] = Queue(maxsize=max(16, int(queue_size)))
        self.lock = Lock()
        self.last_submit_mono: dict[str, float] = {}
        self.last_phase: dict[str, str] = {}
        self.submitted_total = 0
        self.written_total = 0
        self.dropped_total = 0
        self.error_total = 0
        self.current_file = ""
        self.current_file_bytes = 0
        self.running = False
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.running:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.running = True
        self.thread = Thread(target=self._writer_loop, name="shadow-feature-recorder", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if not self.running:
            return
        self.running = False
        try:
            self.queue.put_nowait(None)
        except Full:
            pass
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def submit(self, camera_id: str, snapshot: dict[str, Any], *,
               mono_ts: float | None = None) -> bool:
        now_mono = time.monotonic() if mono_ts is None else float(mono_ts)
        phase = str(snapshot.get("fusion_phase", "NO_PERSON"))
        with self.lock:
            last_ts = self.last_submit_mono.get(camera_id)
            phase_changed = self.last_phase.get(camera_id) != phase
            if last_ts is not None and now_mono - last_ts < self.sample_interval_sec and not phase_changed:
                return False
            self.last_submit_mono[camera_id] = now_mono
            self.last_phase[camera_id] = phase
        row = {field: snapshot.get(field) for field in ALLOWED_FIELDS}
        row["camera_id"] = camera_id
        row["schema_version"] = 1
        row["recorded_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            self.queue.put_nowait(row)
        except Full:
            with self.lock:
                self.dropped_total += 1
            return False
        with self.lock:
            self.submitted_total += 1
        return True

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "enabled": True,
                "sample_interval_sec": self.sample_interval_sec,
                "submitted_total": self.submitted_total,
                "written_total": self.written_total,
                "dropped_total": self.dropped_total,
                "error_total": self.error_total,
                "queue_depth": self.queue.qsize(),
                "queue_capacity": self.queue.maxsize,
                "current_file": self.current_file,
                "current_file_bytes": self.current_file_bytes,
                "thread_alive": bool(self.thread and self.thread.is_alive()),
            }

    def _writer_loop(self) -> None:
        handle = None
        active_day = ""
        last_flush = time.monotonic()
        try:
            while self.running or not self.queue.empty():
                try:
                    row = self.queue.get(timeout=0.25)
                except Empty:
                    row = None
                if row is None:
                    if handle is not None and time.monotonic() - last_flush >= self.flush_interval_sec:
                        handle.flush()
                        last_flush = time.monotonic()
                    if not self.running:
                        break
                    continue
                try:
                    day = str(row["recorded_at"])[:10].replace("-", "")
                    if day != active_day:
                        if handle is not None:
                            handle.flush()
                            handle.close()
                        path = self.output_dir / f"shadow_features_{day}.jsonl"
                        handle = path.open("a", encoding="utf-8", buffering=1)
                        active_day = day
                        with self.lock:
                            self.current_file = str(path.resolve())
                    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    handle.write(encoded + "\n")
                    with self.lock:
                        self.written_total += 1
                        self.current_file_bytes = handle.tell()
                except Exception:
                    with self.lock:
                        self.error_total += 1
                finally:
                    self.queue.task_done()
        finally:
            if handle is not None:
                handle.flush()
                handle.close()
