#!/usr/bin/env python3
"""Check live primary-track and TCN ownership invariants."""

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
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    first = read_status(args.url)
    time.sleep(args.seconds)
    second = read_status(args.url)
    failures = []
    report = {}
    for camera_id, state in second.items():
        primary = state.get("primary_track_id")
        owner = state.get("tcn_track_id")
        item = {
            "capture_fps": round(float(state.get("capture_fps", 0)), 2),
            "person_count": int(state.get("person_count", 0)),
            "track_count": int(state.get("track_count", 0)),
            "primary_track_id": primary,
            "track_switch_total": int(state.get("track_switch_total", 0)),
            "track_created_total": int(state.get("track_created_total", 0)),
            "track_expired_total": int(state.get("track_expired_total", 0)),
            "primary_observed": bool(state.get("primary_track_observed", False)),
            "tcn_track_id": owner,
            "tcn_samples": int(state.get("tcn_samples", 0)),
            "tcn_gap_reset_total": int(state.get("tcn_gap_reset_total", 0)),
            "tcn_track_reset_total": int(state.get("tcn_track_reset_total", 0)),
            "scheduler_alive": bool(state.get("scheduler_thread_alive")),
        }
        report[camera_id] = item
        if item["capture_fps"] < 15:
            failures.append(f"{camera_id}: capture below 15FPS")
        if not item["scheduler_alive"]:
            failures.append(f"{camera_id}: scheduler not alive")
        if primary is None and owner is not None:
            failures.append(f"{camera_id}: TCN owner exists without primary")
        if primary is not None:
            if item["track_count"] < 1:
                failures.append(f"{camera_id}: primary exists without track")
            if owner != primary:
                failures.append(f"{camera_id}: TCN owner differs from primary")
        if not 0 <= item["tcn_samples"] <= 30:
            failures.append(f"{camera_id}: invalid TCN sample count")
        before_switches = int(first.get(camera_id, {}).get("track_switch_total", 0))
        if item["track_switch_total"] < before_switches:
            failures.append(f"{camera_id}: switch counter decreased")

    print(json.dumps(report, indent=2))
    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PASS: Phase 5 primary-track and TCN ownership invariants")


if __name__ == "__main__":
    main()
