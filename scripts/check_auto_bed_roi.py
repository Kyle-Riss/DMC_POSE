#!/usr/bin/env python3
"""Check automatic bed ROI readiness and segmentation throttling."""

from __future__ import annotations

import argparse
import json
import time
from urllib.request import urlopen


def fetch_status(url: str) -> dict:
    with urlopen(url, timeout=5.0) as response:
        return json.load(response)


def compact(status: dict) -> dict:
    return {
        camera_id: {
            "ready": bool(item.get("bed_roi_ready", False)),
            "source": item.get("bed_roi_source", "missing"),
            "version": int(item.get("bed_roi_version", 0)),
            "agreement_iou": round(float(item.get("bed_roi_agreement_iou", 0.0)), 3),
            "candidates": int(item.get("bed_roi_candidate_count", 0)),
            "seg_runs": int(item.get("bed_seg_run_count", 0)),
            "reason": item.get("bed_roi_invalid_reason", "field_missing"),
            "capture_fps": round(float(item.get("capture_fps", 0.0)), 2),
            "pipeline_fps": round(float(item.get("pipeline_fps", 0.0)), 2),
        }
        for camera_id, item in status.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/status")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    latest = {}
    while time.monotonic() < deadline:
        latest = fetch_status(args.url)
        if latest and all(item.get("bed_roi_ready") is True for item in latest.values()):
            break
        time.sleep(1.0)

    # A restored cache is READY before its first background refresh finishes.
    # Give that one boot-time validation pass time to complete.
    time.sleep(2.0)
    latest = fetch_status(args.url)
    first = compact(latest)
    print(json.dumps({"after_bootstrap": first}, indent=2, ensure_ascii=False))
    if not latest or not all(item.get("bed_roi_ready") is True for item in latest.values()):
        print("FAIL: one or more cameras are ROI_NOT_READY")
        return 1

    first_runs = {
        camera_id: int(item.get("bed_seg_run_count", 0))
        for camera_id, item in latest.items()
    }
    time.sleep(max(0.0, args.settle_seconds))
    settled = fetch_status(args.url)
    second_runs = {
        camera_id: int(item.get("bed_seg_run_count", 0))
        for camera_id, item in settled.items()
    }
    unexpected = {
        camera_id: second_runs[camera_id] - first_runs[camera_id]
        for camera_id in first_runs
        if second_runs.get(camera_id, -1) != first_runs[camera_id]
    }
    print(json.dumps({"after_settle": compact(settled)}, indent=2, ensure_ascii=False))
    if unexpected:
        print(f"FAIL: segmentation did not throttle after consensus: {unexpected}")
        return 2
    print("PASS: every camera has an automatic ROI and segmentation is throttled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
