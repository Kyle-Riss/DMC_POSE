#!/usr/bin/env python3
"""Replay recorded shadow observations through the current fusion policy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_fusion import FusionInput, FusionPhase, HybridFusion


def epoch_seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    args = parser.parse_args()

    fusion = HybridFusion()
    counts = {phase.value: 0 for phase in FusionPhase}
    alerts: list[dict] = []
    rows = 0

    with args.jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            timestamp = row.get("timestamp", "")
            if row.get("camera_id") != args.camera:
                continue
            if args.start and timestamp < args.start:
                continue
            if args.end and timestamp > args.end:
                continue
            result = fusion.update(FusionInput(
                timestamp=epoch_seconds(timestamp),
                track_id=row.get("primary_track_id"),
                primary_observed=bool(row.get("primary_track_observed")),
                bed_roi_ready=bool(row.get("bed_roi_ready")),
                body_in_bed_ratio=float(row.get("body_in_bed_ratio") or 0.0),
                pose_class=str(row.get("pose") or "None"),
                pose_confidence=float(row.get("pose_conf") or 0.0),
                legacy_fall_score=float(row.get("fall_score") or 0.0),
                # Live fusion receives the held burst state, not only the
                # single-frame motion detector edge.
                rapid_motion=bool(row.get("burst_active")),
                motion_ratio=float(row.get("motion_ratio") or 0.0),
                tcn_ready=bool(row.get("tcn_shadow_ready")),
                tcn_probability=float(row.get("tcn_fall_probability") or 0.0),
                tcn_threshold=0.5565,
                tcn_candidate=bool(row.get("tcn_alert_candidate")),
                missing_samples=int(row.get("tcn_missing_samples_window") or 0),
            ))
            rows += 1
            counts[result.phase.value] += 1
            if result.phase == FusionPhase.SHADOW_ALERT:
                alerts.append({
                    "timestamp": timestamp,
                    "track_id": result.track_id,
                    "risk": round(result.risk, 6),
                    "evidence": list(result.evidence),
                    "body_in_bed_ratio": row.get("body_in_bed_ratio"),
                    "pose": row.get("pose"),
                })

    print(json.dumps({
        "camera_id": args.camera,
        "rows": rows,
        "phase_counts": counts,
        "alert_count": len(alerts),
        "alerts": alerts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
