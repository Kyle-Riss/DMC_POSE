"""Read-only operational health evaluation for the multi-camera runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping


HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
OFFLINE = "OFFLINE"


@dataclass(frozen=True)
class HealthThresholds:
    min_capture_fps: float = 5.0
    max_capture_age_ms: float = 1000.0
    min_watcher_fps: float = 5.0
    max_analysis_age_ms: float = 2500.0
    max_scheduler_pending: int = 1
    max_scheduler_queue_ms: float = 1000.0


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def evaluate_camera(
    state: Mapping[str, Any],
    thresholds: HealthThresholds | None = None,
) -> dict[str, Any]:
    """Classify one camera without mutating its runtime state."""
    cfg = thresholds or HealthThresholds()
    critical: list[str] = []
    warnings: list[str] = []

    connected = bool(state.get("capture_connected", False))
    capture_age = state.get("capture_frame_age_ms")
    capture_fps = _finite(state.get("capture_fps"))
    analysis_age = state.get("analysis_frame_age_ms")
    watcher_fps = _finite(state.get("watcher_fps"))

    if not connected:
        critical.append("capture_disconnected")
    if not bool(state.get("watcher_thread_alive", False)):
        critical.append("watcher_thread_stopped")
    if not bool(state.get("scheduler_thread_alive", False)):
        critical.append("scheduler_thread_stopped")

    if connected and capture_age is None:
        critical.append("capture_has_no_frame")
    elif capture_age is not None and _finite(capture_age) > cfg.max_capture_age_ms:
        critical.append("capture_frame_stale")

    if connected and capture_fps < cfg.min_capture_fps:
        warnings.append("capture_fps_low")
    if watcher_fps < cfg.min_watcher_fps:
        warnings.append("watcher_fps_low")
    if analysis_age is None:
        warnings.append("analysis_has_no_frame")
    elif _finite(analysis_age) > cfg.max_analysis_age_ms:
        warnings.append("analysis_result_stale")
    if not bool(state.get("bed_roi_ready", False)):
        warnings.append("bed_roi_not_ready")
    if int(state.get("scheduler_pending", 0) or 0) > cfg.max_scheduler_pending:
        warnings.append("scheduler_backlog")
    if _finite(state.get("scheduler_queue_latency_ms")) > cfg.max_scheduler_queue_ms:
        warnings.append("scheduler_queue_slow")
    if int(state.get("scheduler_error_total", 0) or 0) > 0:
        warnings.append("scheduler_errors_seen")

    if critical:
        status = OFFLINE
    elif warnings:
        status = DEGRADED
    else:
        status = HEALTHY
    return {
        "camera_id": state.get("camera_id"),
        "status": status,
        "ready": status != OFFLINE,
        "critical": critical,
        "warnings": warnings,
        "capture_fps": capture_fps,
        "capture_frame_age_ms": capture_age,
        "analysis_frame_age_ms": analysis_age,
        "watcher_fps": watcher_fps,
        "scheduler_pending": int(state.get("scheduler_pending", 0) or 0),
    }


def evaluate_fleet(
    states: Mapping[str, Mapping[str, Any]],
    thresholds: HealthThresholds | None = None,
) -> dict[str, Any]:
    cameras = {
        camera_id: evaluate_camera(state, thresholds)
        for camera_id, state in states.items()
    }
    counts = {HEALTHY: 0, DEGRADED: 0, OFFLINE: 0}
    for result in cameras.values():
        counts[result["status"]] += 1
    if counts[OFFLINE]:
        status = OFFLINE
    elif counts[DEGRADED]:
        status = DEGRADED
    else:
        status = HEALTHY
    return {
        "status": status,
        "camera_count": len(cameras),
        "counts": counts,
        "cameras": cameras,
        "generated_unix": time.time(),
    }


def render_prometheus(
    states: Mapping[str, Mapping[str, Any]],
    *,
    process_ready: bool,
    recorder: Mapping[str, Any] | None = None,
) -> str:
    """Render a compact dependency-free Prometheus exposition."""
    lines = [
        "# HELP dmc_pose_process_ready Central inference process readiness.",
        "# TYPE dmc_pose_process_ready gauge",
        f"dmc_pose_process_ready {1 if process_ready else 0}",
    ]
    fields = {
        "capture_connected": "capture_connected",
        "capture_fps": "capture_fps",
        "capture_frame_age_ms": "capture_frame_age_ms",
        "capture_decode_error_total": "capture_decode_error_total",
        "capture_reconnect_total": "capture_reconnect_total",
        "watcher_fps": "watcher_fps",
        "scheduler_completed_hz": "scheduler_completed_hz",
        "scheduler_queue_latency_ms": "scheduler_queue_latency_ms",
        "scheduler_pending": "scheduler_pending",
        "scheduler_stale_drop_total": "scheduler_stale_drop_total",
        "scheduler_superseded_drop_total": "scheduler_superseded_drop_total",
        "scheduler_timeout_total": "scheduler_timeout_total",
        "scheduler_error_total": "scheduler_error_total",
        "pose_inference_fps": "pose_inference_fps",
        "motion_trigger_total": "motion_trigger_total",
        "pre_event_frames": "pre_event_frames",
        "pre_event_coverage_sec": "pre_event_coverage_sec",
        "pre_event_bytes": "pre_event_bytes",
        "pre_event_ready": "pre_event_ready",
        "tcn_replay_attempt_total": "tcn_replay_attempt_total",
        "tcn_replay_completed_total": "tcn_replay_completed_total",
        "tcn_replay_error_total": "tcn_replay_error_total",
        "tcn_replay_elapsed_ms": "tcn_replay_elapsed_ms",
        "track_switch_total": "track_switch_total",
        "tcn_gap_reset_total": "tcn_gap_reset_total",
    }
    for camera_id, state in sorted(states.items()):
        label = str(camera_id).replace("\\", "\\\\").replace('"', '\\"')
        for source, metric in fields.items():
            value = state.get(source, 0)
            if isinstance(value, bool):
                value = 1 if value else 0
            lines.append(
                f'dmc_pose_{metric}{{camera_id="{label}"}} {_finite(value)}'
            )
    recorder = recorder or {}
    for source in ("written_total", "dropped_total", "error_total", "queue_depth"):
        lines.append(
            f"dmc_pose_recorder_{source} {_finite(recorder.get(source, 0))}"
        )
    return "\n".join(lines) + "\n"
