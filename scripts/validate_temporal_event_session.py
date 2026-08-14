#!/usr/bin/env python3
"""Wait for one automatic 109-D event session and validate its artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--poll-sec", type=float, default=0.5)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--require-tcn-context", action="store_true",
        help="Wait for a newly written session with 30 contiguous observed rows.",
    )
    parser.add_argument(
        "--require-long-context", action="store_true",
        help="Wait for a newly written session with at least 8 seconds/80 rows.",
    )
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + "/temporal-recorder/status"
    initial = fetch_json(endpoint)
    baseline_written = int(initial.get("written_total", 0))
    baseline_tcn_ready = int(initial.get("tcn_context_ready_total", 0))
    baseline_long_ready = int(initial.get("long_pre_context_ready_total", 0))
    saw_active = False
    deadline = time.monotonic() + args.timeout_sec
    final = initial
    while time.monotonic() < deadline:
        final = fetch_json(endpoint)
        saw_active = saw_active or bool(final.get("active_cameras"))
        wrote = int(final.get("written_total", 0)) > baseline_written
        tcn_ready = int(final.get("tcn_context_ready_total", 0)) > baseline_tcn_ready
        long_ready = (
            int(final.get("long_pre_context_ready_total", 0))
            > baseline_long_ready
        )
        if (
            wrote
            and (not args.require_tcn_context or tcn_ready)
            and (not args.require_long_context or long_ready)
        ):
            break
        time.sleep(args.poll_sec)

    checks = {
        "recorder_enabled": bool(final.get("enabled")),
        "writer_thread_alive": bool(final.get("thread_alive")),
        "saw_active_session": saw_active,
        "session_written": int(final.get("written_total", 0)) > baseline_written,
        "no_recorder_errors": int(final.get("error_total", 0)) == 0,
        "no_queue_drops": int(final.get("dropped_total", 0)) == 0,
    }
    if args.require_tcn_context:
        checks["new_tcn_context_ready_session"] = (
            int(final.get("tcn_context_ready_total", 0)) > baseline_tcn_ready
        )
    if args.require_long_context:
        checks["new_long_context_ready_session"] = (
            int(final.get("long_pre_context_ready_total", 0)) > baseline_long_ready
        )
    artifact: dict = {}
    session_dir_text = str(final.get("last_session_dir", ""))
    if checks["session_written"] and session_dir_text:
        session_dir = Path(session_dir_text)
        manifest_path = session_dir / "manifest.json"
        npz_path = session_dir / "features.npz"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as arrays:
            features = arrays["features"]
            timestamps = arrays["relative_timestamps_sec"]
            track_ids = arrays["track_ids"]
            quality = arrays["pose_quality"]
        checks.update({
            "feature_shape_109": features.ndim == 2 and features.shape[1] == 109,
            "features_finite": bool(np.all(np.isfinite(features))),
            "timestamps_monotonic": bool(
                len(timestamps) < 2 or np.all(np.diff(timestamps) > 0)
            ),
            "array_lengths_match": len({
                len(features), len(timestamps), len(track_ids), len(quality)
            }) == 1,
            "label_is_unreviewed": manifest.get("label") == "UNREVIEWED",
            "training_blocked": manifest.get("training_eligible") is False,
            "contains_no_images": not bool(manifest.get("contains_images")),
            "contains_no_video": not bool(manifest.get("contains_video")),
            "contains_no_raw_keypoints": not bool(
                manifest.get("contains_raw_keypoints")
            ),
            "trigger_context_metrics_present": bool(
                manifest.get("trigger_contexts")
            ),
        })
        artifact = {
            "session_dir": str(session_dir),
            "sample_count": int(features.shape[0]),
            "duration_sec": float(timestamps[-1]) if len(timestamps) else 0.0,
            "track_ids": sorted({int(value) for value in track_ids}),
            "trigger_counts": manifest.get("trigger_counts", {}),
            "tcn_context_ready": bool(manifest.get("tcn_context_ready", False)),
            "long_pre_context_ready": bool(
                manifest.get("long_pre_context_ready", False)
            ),
            "best_pre_trigger_context": manifest.get(
                "best_pre_trigger_context"
            ),
        }

    report = {
        "schema_version": "temporal_session_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "initial_status": initial,
        "final_status": final,
        "artifact": artifact,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
