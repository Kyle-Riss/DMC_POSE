#!/usr/bin/env python3
import argparse
import json
import time
from urllib.request import urlopen


def read_status(url):
    with urlopen(url, timeout=3) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/status")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    start = read_status(args.url)
    time.sleep(args.seconds)
    end = read_status(args.url)
    report = {}
    errors = []
    valid_phases = {
        "NO_PERSON", "INSUFFICIENT", "WARMING", "SAFE",
        "CANDIDATE", "VERIFY", "SHADOW_ALERT",
    }
    for camera_id, item in end.items():
        phase = item.get("fusion_phase")
        risk = float(item.get("fusion_risk", -1.0))
        primary = item.get("primary_track_id")
        owner = item.get("fusion_track_id")
        evidence = item.get("fusion_evidence") or []
        created_delta = int(item.get("track_created_total", 0)) - int(
            start.get(camera_id, {}).get("track_created_total", 0)
        )
        report[camera_id] = {
            "capture_fps": round(float(item.get("capture_fps", 0.0)), 2),
            "runtime_mode": item.get("runtime_mode"),
            "primary_track_id": primary,
            "fusion_track_id": owner,
            "fusion_phase": phase,
            "fusion_risk": round(risk, 4),
            "fusion_quality": round(float(item.get("fusion_quality", 0.0)), 4),
            "fusion_evidence": evidence,
            "fusion_safe_evidence": item.get("fusion_safe_evidence") or [],
            "created_delta": created_delta,
        }
        if phase not in valid_phases:
            errors.append(f"{camera_id}: invalid fusion phase {phase!r}")
        if not 0.0 <= risk <= 1.0:
            errors.append(f"{camera_id}: fusion risk outside [0,1]: {risk}")
        if primary != owner:
            errors.append(f"{camera_id}: primary {primary} != fusion owner {owner}")
        if primary is None and phase != "NO_PERSON":
            errors.append(f"{camera_id}: no primary but phase={phase}")
        if phase in {"VERIFY", "SHADOW_ALERT"} and not evidence:
            errors.append(f"{camera_id}: {phase} without evidence")
        if float(item.get("capture_fps", 0.0)) < 10.0:
            errors.append(f"{camera_id}: capture FPS too low")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        print("FAIL:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)
    print("PASS: Phase 6 fusion contract and ownership invariants")


if __name__ == "__main__":
    main()
