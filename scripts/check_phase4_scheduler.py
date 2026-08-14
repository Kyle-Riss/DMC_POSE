#!/usr/bin/env python3
"""Validate the live central inference scheduler through /status."""

import argparse
import json
import time
from urllib.request import urlopen


def read_status(url):
    with urlopen(url, timeout=3.0) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/status")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    first = read_status(args.url)
    time.sleep(args.seconds)
    second = read_status(args.url)
    failures = []
    report = {}

    for camera_id, current in second.items():
        previous = first.get(camera_id, {})
        completed_delta = int(current.get("scheduler_completed_total", 0)) - int(
            previous.get("scheduler_completed_total", 0)
        )
        drop_delta = sum(
            int(current.get(name, 0)) - int(previous.get(name, 0))
            for name in (
                "scheduler_stale_drop_total",
                "scheduler_superseded_drop_total",
                "scheduler_timeout_total",
                "scheduler_error_total",
            )
        )
        item = {
            "capture_fps": round(float(current.get("capture_fps", 0)), 2),
            "runtime_mode": current.get("runtime_mode"),
            "completed_delta": completed_delta,
            "completed_hz": round(float(current.get("scheduler_completed_hz", 0)), 2),
            "queue_latency_ms": round(float(current.get("scheduler_queue_latency_ms", 0)), 1),
            "inference_ms": round(float(current.get("scheduler_inference_ms", 0)), 1),
            "last_priority": current.get("scheduler_last_priority"),
            "pending": int(current.get("scheduler_pending", 0)),
            "drop_delta": drop_delta,
            "thread_alive": bool(current.get("scheduler_thread_alive")),
        }
        report[camera_id] = item
        if item["capture_fps"] < 15.0:
            failures.append(f"{camera_id}: capture_fps < 15")
        if not item["thread_alive"]:
            failures.append(f"{camera_id}: scheduler thread not alive")
        if item["pending"] > 1:
            failures.append(f"{camera_id}: mailbox backlog > 1")
        if item["completed_delta"] < 1:
            failures.append(f"{camera_id}: no inference completed during sample")
        if item["drop_delta"] != 0 and item["runtime_mode"] == "EMPTY":
            failures.append(f"{camera_id}: unexpected EMPTY drop delta {drop_delta}")

    print(json.dumps(report, indent=2))
    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASS: Phase 4 central scheduler is live, bounded, and completing work")


if __name__ == "__main__":
    main()
