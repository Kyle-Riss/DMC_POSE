#!/usr/bin/env python3
"""Summarize feature-only runtime logs into bed-hour and review candidates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def summarize_rows(rows: list[dict], *, max_sample_gap_sec: float = 2.0) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        camera_id = row.get("camera_id")
        timestamp = row.get("recorded_at")
        if camera_id and timestamp:
            grouped[str(camera_id)].append((parse_time(str(timestamp)), row))

    cameras = {}
    all_candidates = []
    for camera_id, samples in sorted(grouped.items()):
        samples.sort(key=lambda item: item[0])
        phases = Counter()
        policy_seconds = Counter()
        recorded_sec = occupied_sec = 0.0
        motion_rows = tcn_candidate_rows = 0
        current_event = None
        candidates = []
        for index, (timestamp, row) in enumerate(samples):
            phase = str(row.get("fusion_phase") or "UNKNOWN")
            phases[phase] += 1
            if row.get("motion_detected"):
                motion_rows += 1
            if row.get("tcn_alert_candidate"):
                tcn_candidate_rows += 1
            if index + 1 < len(samples):
                dt = max(0.0, min(max_sample_gap_sec, samples[index + 1][0] - timestamp))
                recorded_sec += dt
                policy = str(row.get("fusion_policy_version") or "legacy_unknown")
                policy_seconds[policy] += dt
                if row.get("primary_track_id") is not None:
                    occupied_sec += dt

            if phase == "SHADOW_ALERT":
                if current_event is None:
                    current_event = {
                        "camera_id": camera_id,
                        "started_at": row["recorded_at"],
                        "ended_at": row["recorded_at"],
                        "peak_risk": float(row.get("fusion_risk") or 0.0),
                        "evidence": set(row.get("fusion_evidence") or []),
                        "track_ids": set(),
                        "policy_versions": set(),
                    }
                current_event["ended_at"] = row["recorded_at"]
                current_event["peak_risk"] = max(
                    current_event["peak_risk"], float(row.get("fusion_risk") or 0.0)
                )
                current_event["evidence"].update(row.get("fusion_evidence") or [])
                if row.get("fusion_track_id") is not None:
                    current_event["track_ids"].add(row["fusion_track_id"])
                current_event["policy_versions"].add(
                    str(row.get("fusion_policy_version") or "legacy_unknown")
                )
            elif current_event is not None:
                candidates.append(current_event)
                current_event = None
        if current_event is not None:
            candidates.append(current_event)

        serialized_candidates = []
        for event in candidates:
            item = dict(event)
            item["evidence"] = sorted(item["evidence"])
            item["track_ids"] = sorted(item["track_ids"])
            item["policy_versions"] = sorted(item["policy_versions"])
            serialized_candidates.append(item)
            all_candidates.append(item)

        bed_hours = recorded_sec / 3600.0
        cameras[camera_id] = {
            "rows": len(samples),
            "recorded_bed_hours": round(bed_hours, 6),
            "policy_bed_hours": {
                key: round(value / 3600.0, 6)
                for key, value in sorted(policy_seconds.items())
            },
            "occupied_hours": round(occupied_sec / 3600.0, 6),
            "phase_rows": dict(sorted(phases.items())),
            "motion_rows": motion_rows,
            "tcn_candidate_rows": tcn_candidate_rows,
            "shadow_alert_events": len(candidates),
            "shadow_alerts_per_bed_hour": (
                round(len(candidates) / bed_hours, 4) if bed_hours > 0 else None
            ),
            "review_candidates": serialized_candidates,
        }

    total_bed_hours = sum(item["recorded_bed_hours"] for item in cameras.values())
    total_policy_hours = Counter()
    for item in cameras.values():
        total_policy_hours.update(item["policy_bed_hours"])
    return {
        "schema_version": 1,
        "camera_count": len(cameras),
        "total_rows": sum(item["rows"] for item in cameras.values()),
        "total_bed_hours": round(total_bed_hours, 6),
        "policy_bed_hours": {
            key: round(value, 6) for key, value in sorted(total_policy_hours.items())
        },
        "total_shadow_alert_events": len(all_candidates),
        "shadow_alerts_per_bed_hour": (
            round(len(all_candidates) / total_bed_hours, 4)
            if total_bed_hours > 0 else None
        ),
        "cameras": cameras,
        "review_candidates": all_candidates,
        "interpretation": (
            "Shadow alerts are review candidates, not false alarms, until a human label is attached."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=list(Path("runtime_data/shadow_features").glob("*.jsonl")),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runtime_data/shadow_summary.json"),
    )
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    report = summarize_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
