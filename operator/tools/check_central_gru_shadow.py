#!/usr/bin/env python3
"""Summarize protected core GRU/cadence state without exposing raw frames."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

CURL = "/home/dmc/anaconda3/bin/curl"
SOCKET = "/run/company-core/core.sock"
CAMERAS = ("bed_161", "bed_162", "bed_174", "bed_175", "bed_178", "bed_179")


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def main() -> int:
    if os.geteuid() != 0:
        raise PermissionError("run with sudo")
    raw = run(
        CURL, "--noproxy", "*", "--unix-socket", SOCKET,
        "-sS", "--max-time", "15", "http://localhost/api/v2/status",
    )
    status = json.loads(raw)
    rows = {}
    for camera in CAMERAS:
        item = status[camera]
        rows[camera] = {
            "capture_fps": round(float(item["capture_fps"]), 3),
            "capture_age_ms": (
                None if item.get("capture_frame_age_ms") is None
                else round(float(item["capture_frame_age_ms"]), 3)
            ),
            "pipeline_fps": round(float(item["pipeline_fps"]), 3),
            "pose_inference_fps": round(float(item["pose_inference_fps"]), 3),
            "scheduler_hz": round(float(item["scheduler_completed_hz"]), 3),
            "scheduler_completed": int(item["scheduler_completed_total"]),
            "scheduler_inference_ms": round(float(item["scheduler_inference_ms"]), 3),
            "queue_latency_ms": round(float(item["scheduler_queue_latency_ms"]), 3),
            "scheduler_stale_drops": int(item["scheduler_stale_drop_total"]),
            "scheduler_superseded_drops": int(
                item["scheduler_superseded_drop_total"]
            ),
            "scheduler_pending": int(item["scheduler_pending"]),
            "scheduler_errors": int(item["scheduler_error_total"]),
            "scheduler_timeouts": int(item["scheduler_timeout_total"]),
            "watcher_thread_alive": bool(item["watcher_thread_alive"]),
            "watcher_fps": round(float(item["watcher_fps"]), 3),
            "pre_event_frames": int(item["pre_event_frames"]),
            "tcn_enabled": bool(item["tcn_shadow_enabled"]),
            "tcn_ready": bool(item["tcn_shadow_ready"]),
            "tcn_samples": int(item["tcn_samples"]),
            "tcn_window_rows": int(item["tcn_window_rows"]),
            "tcn_sample_hz": float(item["tcn_sample_hz"]),
            "tcn_predictions": int(item["tcn_prediction_count"]),
            "tcn_gap_resets": int(item["tcn_gap_reset_total"]),
            "tcn_track_resets": int(item["tcn_track_reset_total"]),
            "tcn_fusion_enabled": bool(item["tcn_fusion_enabled"]),
            "video_verifier_enabled": bool(item.get("video_verifier_enabled", False)),
            "video_verifier_running": bool(item.get("video_verifier_running", False)),
            "video_verifier_ready": bool(item.get("video_verifier_ready", False)),
            "video_verifier_candidate": bool(item.get("video_verifier_candidate", False)),
            "video_verifier_baseline": item.get("video_verifier_baseline"),
            "video_verifier_post_max": item.get("video_verifier_post_max"),
            "video_verifier_delta": item.get("video_verifier_delta"),
            "video_verifier_pair_probability": item.get("video_verifier_pair_probability"),
            "video_verifier_decision_mode": str(item.get("video_verifier_decision_mode", "missing")),
            "video_verifier_threshold": float(item.get("video_verifier_threshold", 0.0)),
            "video_verifier_latency_ms": round(float(item.get("video_verifier_latency_ms", 0.0)), 3),
            "video_verifier_triggers": int(item.get("video_verifier_trigger_total", 0)),
            "video_verifier_completed": int(item.get("video_verifier_completed_total", 0)),
            "video_verifier_errors": int(item.get("video_verifier_error_total", 0)),
            "video_verifier_ring_frames": int(item.get("video_verifier_ring_frames", 0)),
            "video_verifier_ring_coverage_sec": round(float(item.get("video_verifier_ring_coverage_sec", 0.0)), 3),
            "video_verifier_authority": str(item.get("video_verifier_authority", "missing")),
            "person_count": int(item["person_count"]),
        }
    show = run(
        "systemctl", "show", "company-core.service",
        "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts",
        "-p", "MainPID", "-p", "MemoryCurrent", "-p", "TasksCurrent",
    )
    service = dict(line.split("=", 1) for line in show.splitlines() if "=" in line)
    ss = run("ss", "-H", "-tn", "state", "established")
    rtsp = [
        line.strip() for line in ss.splitlines()
        if ":8554" in line and any(f"192.168.0.{ip}:8554" in line for ip in ("161", "162", "174", "175", "178", "179"))
    ]
    result = {
        "schema_version": "dmc_central_gru_shadow_status_v1",
        "service": {
            "active": service.get("ActiveState"),
            "substate": service.get("SubState"),
            "restarts": int(service.get("NRestarts", -1)),
            "pid": int(service.get("MainPID", 0)),
            "memory_mib": round(int(service.get("MemoryCurrent", 0)) / 1048576, 1),
            "tasks": int(service.get("TasksCurrent", 0)),
        },
        "rtsp_connections": len(rtsp),
        "cameras": rows,
        "checks": {
            "service_stable": service.get("ActiveState") == "active" and int(service.get("NRestarts", -1)) == 0,
            "one_rtsp_per_camera": len(rtsp) == 6,
            "contract_10hz_40_rows": all(row["tcn_sample_hz"] == 10.0 and row["tcn_window_rows"] == 40 for row in rows.values()),
            "fusion_disabled": all(not row["tcn_fusion_enabled"] for row in rows.values()),
            "watchers_enabled": all(
                row["watcher_thread_alive"] for row in rows.values()
            ),
            "pose_at_least_9hz": all(row["pose_inference_fps"] >= 9.0 for row in rows.values()),
            "video_verifier_enabled": all(
                row["video_verifier_enabled"] for row in rows.values()
            ),
            "video_verifier_ring_ready": all(
                row["video_verifier_ring_coverage_sec"] >= 8.0
                for row in rows.values()
            ),
            "video_verifier_telemetry_only": all(
                row["video_verifier_authority"] == "telemetry_only"
                for row in rows.values()
            ),
            "video_verifier_delta_probe": all(
                row["video_verifier_decision_mode"] == "delta_embedding_v1"
                for row in rows.values()
            ),
            "video_verifier_clean": all(
                row["video_verifier_errors"] == 0 for row in rows.values()
            ),
            "scheduler_clean": all(row["scheduler_errors"] == 0 and row["scheduler_timeouts"] == 0 for row in rows.values()),
        },
        "note": "GRU requires 40 observed rows; motion/fusion may independently trigger the telemetry-only RGB verifier",
    }
    print(json.dumps(result, indent=2))
    return 0 if all(result["checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
