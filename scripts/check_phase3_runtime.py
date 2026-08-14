#!/usr/bin/env python3
"""Validate Phase 3 live runtime rates through the local status API."""

from __future__ import annotations

import argparse
import json
import time
from urllib.request import urlopen


def load_status(url: str) -> dict:
    with urlopen(url, timeout=5.0) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/status")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--min-capture-fps", type=float, default=15.0)
    parser.add_argument("--min-watcher-fps", type=float, default=15.0)
    parser.add_argument("--max-empty-pose-fps", type=float, default=2.0)
    args = parser.parse_args()

    before = load_status(args.url)
    started = time.monotonic()
    time.sleep(max(1.0, args.seconds))
    after = load_status(args.url)
    elapsed = time.monotonic() - started
    failures = []
    report = {}

    for camera_id, state in after.items():
        previous = before.get(camera_id, {})
        watcher_delta = (
            int(state.get("watcher_processed_total", 0))
            - int(previous.get("watcher_processed_total", 0))
        ) / elapsed
        pose_delta_fps = (
            int(state.get("pose_inference_total", 0))
            - int(previous.get("pose_inference_total", 0))
        ) / elapsed
        stable_empty = (
            previous.get("runtime_mode") == "EMPTY"
            and state.get("runtime_mode") == "EMPTY"
            and previous.get("primary_track_id") is None
            and state.get("primary_track_id") is None
            and int(state.get("motion_trigger_total", 0))
            == int(previous.get("motion_trigger_total", 0))
        )
        item = {
            "capture_fps": round(float(state.get("capture_fps", 0.0)), 2),
            "watcher_delta_fps": round(watcher_delta, 2),
            "pose_delta_fps": round(pose_delta_fps, 2),
            "stable_empty_window": stable_empty,
            "runtime_mode": state.get("runtime_mode"),
            "bed_roi_ready": bool(state.get("bed_roi_ready")),
            "watcher_thread_alive": bool(state.get("watcher_thread_alive")),
        }
        report[camera_id] = item
        if item["capture_fps"] < args.min_capture_fps:
            failures.append(f"{camera_id}: capture FPS too low")
        if item["watcher_delta_fps"] < args.min_watcher_fps:
            failures.append(f"{camera_id}: watcher FPS too low")
        if not item["bed_roi_ready"]:
            failures.append(f"{camera_id}: automatic bed ROI not ready")
        if not item["watcher_thread_alive"]:
            failures.append(f"{camera_id}: watcher thread is not alive")
        if (
            item["stable_empty_window"]
            and item["pose_delta_fps"] > args.max_empty_pose_fps
        ):
            failures.append(f"{camera_id}: EMPTY pose rate too high")

    print(json.dumps(report, indent=2))
    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: Phase 3 live capture, watcher, ROI, and EMPTY load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
