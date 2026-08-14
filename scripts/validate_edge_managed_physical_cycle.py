#!/usr/bin/env python3
"""Record and evaluate one physical edge-managed enter/exit cycle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def fetch_json(url: str, token: str | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urlopen(Request(url, headers=headers), timeout=3.0) as response:
        return json.load(response)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", default="bed_161")
    parser.add_argument("--node-id", default="rpi-bed-161")
    parser.add_argument("--status-url", default="http://127.0.0.1:8000/status")
    parser.add_argument("--edge-url", default="http://127.0.0.1:8020/edge/nodes")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    started = time.monotonic()
    samples: list[dict] = []
    transitions: list[dict] = []
    errors: list[str] = []
    previous_signature = None

    while time.monotonic() - started < args.duration:
        tick = time.monotonic()
        try:
            central = fetch_json(args.status_url)[args.camera_id]
            registry = fetch_json(args.edge_url, token)
            node = next(item for item in registry["nodes"] if item["node_id"] == args.node_id)
            heartbeat = node["heartbeat"]
            row = {
                "observed_at": utc_now(),
                "elapsed_sec": round(time.monotonic() - started, 3),
                "runtime_mode": central.get("runtime_mode"),
                "person_count": int(central.get("person_count", 0)),
                "pose": central.get("pose"),
                "in_bed": central.get("in_bed"),
                "scheduler_priority": central.get("scheduler_last_priority"),
                "fusion_risk": float(central.get("fusion_risk", 0.0)),
                "fusion_phase": central.get("fusion_phase"),
                "fall_score": float(central.get("fall_score", 0.0)),
                "fall_level": central.get("fall_level"),
                "fall_status": central.get("fall_status"),
                "tcn_ready": bool(central.get("tcn_shadow_ready", False)),
                "tcn_probability": float(central.get("tcn_fall_probability", 0.0)),
                "tcn_candidate": bool(central.get("tcn_alert_candidate", False)),
                "edge_result_fresh": bool(central.get("edge_signal_result_fresh", False)),
                "edge_person_present": bool(central.get("edge_signal_person_present", False)),
                "edge_frame_seq": int(central.get("edge_signal_frame_seq", -1)),
                "central_watcher_fps": float(central.get("watcher_fps", 0.0)),
                "edge_fallback_active": bool(central.get("edge_fallback_active", False)),
                "edge_heartbeat_sequence": int(heartbeat["sequence"]),
                "edge_spool_depth": int(heartbeat["spool_depth"]),
                "edge_watcher_fps": float(heartbeat["watcher_fps"]),
            }
            samples.append(row)
            signature = (
                row["runtime_mode"], row["person_count"], row["pose"], row["in_bed"],
                row["scheduler_priority"], row["edge_result_fresh"],
                row["edge_person_present"], row["edge_frame_seq"],
                row["fusion_phase"], row["tcn_candidate"],
            )
            if signature != previous_signature:
                transitions.append(row)
                previous_signature = signature
        except Exception as exc:
            errors.append(f"{utc_now()} {type(exc).__name__}: {exc}")
        remaining = args.interval - (time.monotonic() - tick)
        if remaining > 0:
            time.sleep(remaining)

    baseline_seq = samples[0]["edge_frame_seq"] if samples else -1
    edge_person_indices = [i for i, row in enumerate(samples) if row["edge_person_present"]]
    central_person_indices = [i for i, row in enumerate(samples) if row["person_count"] > 0]
    activation_indices = [
        i for i, row in enumerate(samples)
        if row["runtime_mode"] in {"BURST", "OCCUPIED"} or row["person_count"] > 0
    ]
    activated_at = min(activation_indices) if activation_indices else None
    returned_empty = bool(
        activated_at is not None
        and any(
            row["runtime_mode"] == "EMPTY" and row["person_count"] == 0
            for row in samples[activated_at + 1:]
        )
    )
    checks = {
        "samples_collected": len(samples) > 0,
        "no_poll_errors": len(errors) == 0,
        "edge_result_sequence_advanced": any(
            row["edge_frame_seq"] > baseline_seq for row in samples
        ),
        "fresh_edge_person_observed": bool(edge_person_indices),
        "central_person_observed": bool(central_person_indices),
        "central_analysis_activated": bool(activation_indices),
        "returned_to_empty_after_activation": returned_empty,
        "central_watcher_remained_suppressed": all(
            row["central_watcher_fps"] == 0.0 for row in samples
        ),
        "edge_fallback_never_activated": not any(
            row["edge_fallback_active"] for row in samples
        ),
        "edge_spool_never_backlogged": max(
            (row["edge_spool_depth"] for row in samples), default=1
        ) == 0,
        "fusion_never_raised_alert_phase": not any(
            row["fusion_phase"] in {"CANDIDATE", "VERIFY", "SHADOW_ALERT"}
            for row in samples
        ),
        "frame_fall_score_remained_zero": max(
            (row["fall_score"] for row in samples), default=1.0
        ) == 0.0,
    }
    report = {
        "schema_version": "edge-managed-physical-cycle-v1",
        "camera_id": args.camera_id,
        "node_id": args.node_id,
        "duration_sec": args.duration,
        "interval_sec": args.interval,
        "sample_count": len(samples),
        "error_count": len(errors),
        "errors": errors,
        "checks": checks,
        "passed": all(checks.values()),
        "summary": {
            "baseline_edge_frame_seq": baseline_seq,
            "max_edge_frame_seq": max((row["edge_frame_seq"] for row in samples), default=-1),
            "max_person_count": max((row["person_count"] for row in samples), default=0),
            "max_edge_spool_depth": max((row["edge_spool_depth"] for row in samples), default=None),
            "edge_watcher_fps_mean": (
                sum(row["edge_watcher_fps"] for row in samples) / len(samples)
                if samples else None
            ),
            "max_fusion_risk": max((row["fusion_risk"] for row in samples), default=0.0),
            "fusion_phases": sorted({row["fusion_phase"] for row in samples}),
            "max_tcn_probability": max((row["tcn_probability"] for row in samples), default=0.0),
            "tcn_candidate_sample_count": sum(row["tcn_candidate"] for row in samples),
            "max_fall_score": max((row["fall_score"] for row in samples), default=0.0),
            "runtime_modes": sorted({row["runtime_mode"] for row in samples}),
        },
        "transitions": transitions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
