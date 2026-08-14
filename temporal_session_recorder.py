"""Automatic event-triggered recorder for observed-only 109-D TCN rows.

The recorder never accepts images, raw keypoints, or RTSP URLs. Camera threads
only update small in-memory rings; completed sessions are written by one daemon
thread so compressed NPZ I/O cannot block live inference.
"""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Iterable

import numpy as np

from temporal_features import FEATURE_COUNT, FEATURE_SCHEMA_VERSION


FUSION_TRIGGER_PHASES = {"CANDIDATE", "VERIFY", "SHADOW_ALERT"}
MODEL_ONLY_TRIGGERS = {"TCN_CANDIDATE_RISE"}
CONTEXT_MIN_DT_SEC = 0.070
CONTEXT_MAX_DT_SEC = 0.250
TCN_CONTEXT_SAMPLES = 30
LONG_CONTEXT_SEC = 8.0
LONG_CONTEXT_MIN_SAMPLES = 80


def derive_temporal_session_triggers(previous: dict, current: dict) -> set[str]:
    """Return rising/transition triggers without assigning a ground-truth label."""
    triggers: set[str] = set()
    previous_person = bool(previous.get("person_observed", False))
    current_person = bool(current.get("person_observed", False))
    if current_person and not previous_person:
        triggers.add("PERSON_ENTER")
    elif previous_person and not current_person:
        triggers.add("PERSON_EXIT")

    if bool(current.get("edge_wake", False)) and not bool(
        previous.get("edge_wake", False)
    ):
        triggers.add("EDGE_WAKE_RISE")
    if bool(current.get("local_motion", False)) and not bool(
        previous.get("local_motion", False)
    ):
        triggers.add("LOCAL_MOTION_RISE")
    if bool(current.get("tcn_candidate", False)) and not bool(
        previous.get("tcn_candidate", False)
    ):
        triggers.add("TCN_CANDIDATE_RISE")

    previous_phase = str(previous.get("fusion_phase", "NO_PERSON"))
    current_phase = str(current.get("fusion_phase", "NO_PERSON"))
    if current_phase in FUSION_TRIGGER_PHASES and previous_phase not in FUSION_TRIGGER_PHASES:
        triggers.add("FUSION_CANDIDATE_RISE")

    previous_track = previous.get("track_id")
    current_track = current.get("track_id")
    if (
        previous_track is not None
        and current_track is not None
        and int(previous_track) != int(current_track)
    ):
        triggers.add("PRIMARY_TRACK_CHANGE")
    return triggers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TemporalEventSessionRecorder:
    """Keep pre-roll in RAM and persist only automatically triggered sessions."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        pre_roll_sec: float = 10.0,
        post_roll_sec: float = 10.0,
        max_session_sec: float = 180.0,
        model_trigger_rearm_sec: float = 60.0,
        queue_size: int = 32,
    ):
        self.output_dir = Path(output_dir)
        self.pre_roll_sec = max(0.0, float(pre_roll_sec))
        self.post_roll_sec = max(0.0, float(post_roll_sec))
        self.max_session_sec = max(self.post_roll_sec, float(max_session_sec))
        self.model_trigger_rearm_sec = max(0.0, float(model_trigger_rearm_sec))
        self.queue: Queue[tuple[dict, str] | None] = Queue(maxsize=max(2, int(queue_size)))
        self.lock = Lock()
        self.rings: dict[str, deque[dict]] = {}
        self.active: dict[str, dict] = {}
        self.last_finished_capture_ts: dict[str, float] = {}
        self.running = False
        self.thread: Thread | None = None
        self.completed_total = 0
        self.written_total = 0
        self.dropped_total = 0
        self.invalid_total = 0
        self.skipped_empty_total = 0
        self.error_total = 0
        self.trigger_counts: Counter[str] = Counter()
        self.suppressed_model_trigger_total = 0
        self.last_session_dir = ""
        self.tcn_context_ready_total = 0
        self.long_pre_context_ready_total = 0
        self.last_session_context: dict | None = None

    def start(self) -> None:
        if self.running:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.running = True
        self.thread = Thread(
            target=self._writer_loop,
            name="temporal-session-writer",
            daemon=True,
        )
        self.thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        if not self.running:
            return
        with self.lock:
            for camera_id in list(self.active):
                self._finish_locked(camera_id, "shutdown")
        self.running = False
        try:
            self.queue.put(None, timeout=max(0.1, timeout))
        except Full:
            self.dropped_total += 1
        if self.thread is not None:
            self.thread.join(timeout=max(0.0, timeout))

    def observe(
        self,
        camera_id: str,
        capture_ts: float,
        feature: np.ndarray,
        *,
        track_id: int,
        quality: float,
        triggers: Iterable[str] = (),
    ) -> bool:
        vector = np.asarray(feature, dtype=np.float32)
        if vector.shape != (FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            with self.lock:
                self.invalid_total += 1
            return False
        sample = {
            "capture_ts": float(capture_ts),
            "feature": vector.copy(),
            "track_id": int(track_id),
            "quality": float(np.clip(quality, 0.0, 1.0)),
        }
        self._process(str(camera_id), float(capture_ts), sample, triggers)
        return True

    def tick(
        self,
        camera_id: str,
        capture_ts: float,
        *,
        triggers: Iterable[str] = (),
    ) -> None:
        self._process(str(camera_id), float(capture_ts), None, triggers)

    def _process(
        self,
        camera_id: str,
        capture_ts: float,
        sample: dict | None,
        triggers: Iterable[str],
    ) -> None:
        trigger_names = sorted({str(item) for item in triggers if str(item)})
        with self.lock:
            active = self.active.get(camera_id)
            if active is not None and (
                capture_ts >= active["max_deadline"]
                or (capture_ts >= active["deadline"] and not trigger_names)
            ):
                self._finish_locked(
                    camera_id,
                    "max_duration" if capture_ts >= active["max_deadline"] else "post_roll",
                    capture_ts=capture_ts,
                )
                active = None

            # A noisy shadow checkpoint may repeatedly fall below and rise
            # above threshold after one captured episode. It may extend an
            # active session, but cannot by itself reopen another disk session
            # during the bounded re-arm interval. Edge/person/Fusion triggers
            # remain authoritative and are never suppressed here.
            if trigger_names and active is None and set(trigger_names).issubset(
                MODEL_ONLY_TRIGGERS
            ):
                last_finished = self.last_finished_capture_ts.get(camera_id)
                if (
                    last_finished is not None
                    and capture_ts - last_finished < self.model_trigger_rearm_sec
                ):
                    self.suppressed_model_trigger_total += len(trigger_names)
                    trigger_names = []

            ring = self.rings.setdefault(camera_id, deque())
            cutoff = capture_ts - self.pre_roll_sec
            while ring and ring[0]["capture_ts"] < cutoff:
                ring.popleft()
            if sample is not None:
                ring.append(sample)
                if active is not None:
                    active["samples"].append(sample)

            if trigger_names:
                if active is None:
                    active = self._start_locked(camera_id, capture_ts, list(ring))
                active["deadline"] = min(
                    active["max_deadline"], capture_ts + self.post_roll_sec
                )
                for name in trigger_names:
                    active["triggers"].append({
                        "name": name,
                        # Internal only. It is converted to an offset and
                        # per-trigger context metrics before serialization.
                        "capture_ts": float(capture_ts),
                        "capture_offset_sec": round(
                            capture_ts - active["trigger_capture_ts"], 6
                        ),
                        "observed_at": _utc_now(),
                    })
                    self.trigger_counts[name] += 1

    def _start_locked(self, camera_id: str, capture_ts: float, samples: list[dict]) -> dict:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        session = {
            "session_id": f"{camera_id}_{stamp}",
            "camera_id": camera_id,
            "started_at": _utc_now(),
            "trigger_capture_ts": float(capture_ts),
            "deadline": float(capture_ts + self.post_roll_sec),
            "max_deadline": float(capture_ts + self.max_session_sec),
            "samples": list(samples),
            "triggers": [],
        }
        self.active[camera_id] = session
        return session

    def _finish_locked(
        self,
        camera_id: str,
        reason: str,
        *,
        capture_ts: float | None = None,
    ) -> None:
        session = self.active.pop(camera_id, None)
        if session is None:
            return
        session["finished_at"] = _utc_now()
        session["close_reason"] = str(reason)
        if capture_ts is not None:
            self.last_finished_capture_ts[camera_id] = float(capture_ts)
        self.completed_total += 1
        try:
            self.queue.put_nowait((session, str(reason)))
        except Full:
            self.dropped_total += 1

    def status(self) -> dict:
        with self.lock:
            return {
                "enabled": True,
                "thread_alive": bool(self.thread and self.thread.is_alive()),
                "pre_roll_sec": self.pre_roll_sec,
                "post_roll_sec": self.post_roll_sec,
                "max_session_sec": self.max_session_sec,
                "model_trigger_rearm_sec": self.model_trigger_rearm_sec,
                "active_cameras": sorted(self.active),
                "ring_samples": {
                    camera_id: len(rows) for camera_id, rows in self.rings.items()
                },
                "completed_total": self.completed_total,
                "written_total": self.written_total,
                "dropped_total": self.dropped_total,
                "invalid_total": self.invalid_total,
                "skipped_empty_total": self.skipped_empty_total,
                "error_total": self.error_total,
                "queue_depth": self.queue.qsize(),
                "trigger_counts": dict(self.trigger_counts),
                "suppressed_model_trigger_total": self.suppressed_model_trigger_total,
                "last_session_dir": self.last_session_dir,
                "tcn_context_ready_total": self.tcn_context_ready_total,
                "long_pre_context_ready_total": self.long_pre_context_ready_total,
                "last_session_context": self.last_session_context,
                "contains_images": False,
                "contains_keypoints": False,
                "feature_schema": FEATURE_SCHEMA_VERSION,
            }

    def _writer_loop(self) -> None:
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.25)
            except Empty:
                continue
            if item is None:
                self.queue.task_done()
                if not self.running:
                    break
                continue
            session, _reason = item
            try:
                written = self._write_session(session)
                with self.lock:
                    if written:
                        self.written_total += 1
                    else:
                        self.skipped_empty_total += 1
            except Exception:
                with self.lock:
                    self.error_total += 1
            finally:
                self.queue.task_done()

    def _write_session(self, session: dict) -> bool:
        samples = session["samples"]
        if not samples:
            # A trigger without one observed Pose row is useful operationally
            # but cannot train the temporal model, so do not emit an empty NPZ.
            return False
        target = self.output_dir / session["session_id"]
        target.mkdir(parents=True, exist_ok=False)
        capture_ts = np.asarray([row["capture_ts"] for row in samples], dtype=np.float64)
        features = np.stack([row["feature"] for row in samples]).astype(np.float32)
        track_ids = np.asarray([row["track_id"] for row in samples], dtype=np.int64)
        quality = np.asarray([row["quality"] for row in samples], dtype=np.float32)
        relative_ts = capture_ts - float(capture_ts[0])

        trigger_contexts = []
        for trigger in session["triggers"]:
            trigger_ts = float(trigger.get("capture_ts", session["trigger_capture_ts"]))
            eligible = np.flatnonzero(capture_ts <= trigger_ts + 1e-6)
            contiguous_count = 0
            coverage_to_trigger = 0.0
            context_track_id = None
            latest_gap = None
            if len(eligible):
                end_index = int(eligible[-1])
                latest_gap = max(0.0, trigger_ts - float(capture_ts[end_index]))
                context_track_id = int(track_ids[end_index])
                if latest_gap <= CONTEXT_MAX_DT_SEC:
                    start_index = end_index
                    while start_index > 0:
                        dt = float(capture_ts[start_index] - capture_ts[start_index - 1])
                        if (
                            int(track_ids[start_index - 1]) != context_track_id
                            or dt < CONTEXT_MIN_DT_SEC
                            or dt > CONTEXT_MAX_DT_SEC
                        ):
                            break
                        start_index -= 1
                    contiguous_count = end_index - start_index + 1
                    coverage_to_trigger = max(
                        0.0, trigger_ts - float(capture_ts[start_index])
                    )
            trigger_contexts.append({
                "name": str(trigger["name"]),
                "capture_offset_sec": float(trigger["capture_offset_sec"]),
                "observed_at": str(trigger["observed_at"]),
                "track_id": context_track_id,
                "latest_observation_gap_sec": (
                    round(float(latest_gap), 6) if latest_gap is not None else None
                ),
                "contiguous_observed_samples": int(contiguous_count),
                "coverage_to_trigger_sec": round(float(coverage_to_trigger), 6),
                "tcn_context_ready": bool(contiguous_count >= TCN_CONTEXT_SAMPLES),
                "long_pre_context_ready": bool(
                    contiguous_count >= LONG_CONTEXT_MIN_SAMPLES
                    and coverage_to_trigger >= LONG_CONTEXT_SEC
                ),
            })

        best_context = max(
            trigger_contexts,
            key=lambda item: (
                int(item["long_pre_context_ready"]),
                int(item["tcn_context_ready"]),
                float(item["coverage_to_trigger_sec"]),
                int(item["contiguous_observed_samples"]),
            ),
            default=None,
        )

        npz_path = target / "features.npz"
        temporary_npz = target / "features.npz.tmp"
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=features,
                relative_timestamps_sec=relative_ts,
                track_ids=track_ids,
                pose_quality=quality,
            )
        os.replace(temporary_npz, npz_path)

        manifest = {
            "schema_version": "temporal_event_session_v1",
            "session_id": session["session_id"],
            "camera_id": session["camera_id"],
            "started_at": session["started_at"],
            "finished_at": session["finished_at"],
            "close_reason": session["close_reason"],
            "label": "UNREVIEWED",
            "binary_fall_label": None,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_count": FEATURE_COUNT,
            "sample_count": int(features.shape[0]),
            "duration_sec": float(relative_ts[-1]) if len(relative_ts) else 0.0,
            "track_ids": sorted({int(value) for value in track_ids}),
            "triggers": [
                {key: value for key, value in item.items() if key != "capture_ts"}
                for item in session["triggers"]
            ],
            "trigger_counts": dict(Counter(
                item["name"] for item in session["triggers"]
            )),
            "trigger_contexts": trigger_contexts,
            "best_pre_trigger_context": best_context,
            "tcn_context_ready": bool(
                best_context and best_context["tcn_context_ready"]
            ),
            "long_pre_context_ready": bool(
                best_context and best_context["long_pre_context_ready"]
            ),
            "context_contract": {
                "observed_only": True,
                "same_track_required": True,
                "min_dt_sec": CONTEXT_MIN_DT_SEC,
                "max_dt_sec": CONTEXT_MAX_DT_SEC,
                "tcn_samples": TCN_CONTEXT_SAMPLES,
                "long_context_sec": LONG_CONTEXT_SEC,
                "long_context_min_samples": LONG_CONTEXT_MIN_SAMPLES,
            },
            "contains_video": False,
            "contains_images": False,
            "contains_raw_keypoints": False,
            "training_eligible": False,
            "training_blockers": ["label_unreviewed"],
        }
        manifest_tmp = target / "manifest.json.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, target / "manifest.json")
        with self.lock:
            self.last_session_dir = str(target.resolve())
            self.tcn_context_ready_total += int(manifest["tcn_context_ready"])
            self.long_pre_context_ready_total += int(
                manifest["long_pre_context_ready"]
            )
            self.last_session_context = {
                "session_id": manifest["session_id"],
                "tcn_context_ready": manifest["tcn_context_ready"],
                "long_pre_context_ready": manifest["long_pre_context_ready"],
                "best_pre_trigger_context": manifest["best_pre_trigger_context"],
            }
        return True
