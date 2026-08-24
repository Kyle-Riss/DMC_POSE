#!/usr/bin/env python3
"""Audit legacy AI_runner event clips against the frozen DMC 20 Hz contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from temporal_sequence import cadence_interval_bounds, observed_sequence_contract


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_event(event_dir: Path, *, target_hz: float = 20.0, verify_hashes: bool = True) -> dict:
    timeline_path = event_dir / "timeline.csv"
    meta_path = event_dir / "event_meta.json"
    rows = list(csv.DictReader(timeline_path.open(encoding="utf-8-sig", newline="")))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    timestamps = [float(row["timestamp"]) for row in rows]
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    min_interval, max_interval = cadence_interval_bounds(target_hz)
    cadence_ok = bool(intervals) and all(
        min_interval - 1e-9 <= value <= max_interval + 1e-9 for value in intervals
    )
    missing_images = []
    hash_mismatches = []
    for row in rows:
        image_path = event_dir / row["image_path"]
        if not image_path.is_file():
            missing_images.append(str(image_path))
        elif verify_hashes and row.get("jpeg_sha256") and file_sha256(image_path) != row["jpeg_sha256"]:
            hash_mismatches.append(str(image_path))

    ground_truth = str(meta.get("groundTruth") or "unknown")
    blockers = []
    if not cadence_ok:
        blockers.append("does_not_satisfy_observed_only_20hz_cadence")
    if len(rows) < int(round(4.0 * target_hz)):
        blockers.append("fewer_than_80_observations")
    if ground_truth != "normal_exit":
        blockers.append("fall_onset_impact_stable_boundaries_missing")
    blockers.extend(["subject_identity_unknown", "session_split_identity_unknown"])
    if missing_images:
        blockers.append("missing_images")
    if hash_mismatches:
        blockers.append("image_hash_mismatch")

    span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    return {
        "event_id": meta.get("eventId") or event_dir.name,
        "event_dir": str(event_dir.resolve()),
        "camera_id": meta.get("cameraId"),
        "ground_truth": ground_truth,
        "trigger_reason": (meta.get("trigger") or {}).get("reason"),
        "row_count": len(rows),
        "span_sec": round(span, 6),
        "effective_hz": round((len(rows) - 1) / span, 6) if span > 0.0 else None,
        "interval_sec": {
            "minimum": round(min(intervals), 6) if intervals else None,
            "median": round(statistics.median(intervals), 6) if intervals else None,
            "maximum": round(max(intervals), 6) if intervals else None,
        },
        "phase_counts": dict(Counter(row.get("phase") or "unknown" for row in rows)),
        "target_hz": target_hz,
        "sequence_contract_version": observed_sequence_contract(target_hz),
        "cadence_bounds_sec": [min_interval, max_interval],
        "cadence_eligible": cadence_ok,
        "missing_image_count": len(missing_images),
        "hash_mismatch_count": len(hash_mismatches),
        "production_gru_training_eligible": not blockers,
        "training_blockers": blockers,
        "allowed_uses": ["fusion_replay", "runtime_regression", "visual_failure_review"],
    }


def audit_root(root: Path, *, target_hz: float = 20.0, verify_hashes: bool = True) -> dict:
    event_dirs = sorted(path.parent for path in root.glob("*/*/timeline.csv"))
    events = [audit_event(path, target_hz=target_hz, verify_hashes=verify_hashes) for path in event_dirs]
    return {
        "schema_version": "dmc_legacy_event_clip_audit_v1",
        "source_root": str(root.resolve()),
        "target_hz": target_hz,
        "sequence_contract_version": observed_sequence_contract(target_hz),
        "event_count": len(events),
        "ground_truth_counts": dict(Counter(event["ground_truth"] for event in events)),
        "training_eligible_count": sum(event["production_gru_training_eligible"] for event in events),
        "cadence_eligible_count": sum(event["cadence_eligible"] for event in events),
        "integrity_error_count": sum(event["missing_image_count"] + event["hash_mismatch_count"] for event in events),
        "decision": "fusion/replay evidence only; never upsample into the 20 Hz production GRU corpus",
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dmc/AI/AI_runner/data/research/fall_v4e_post_trigger"),
    )
    parser.add_argument("--target-hz", type=float, default=20.0)
    parser.add_argument("--skip-hash-verification", action="store_true")
    parser.add_argument("--out", type=Path, default=project / "docs/ai_runner_event_clip_audit_20260824.json")
    args = parser.parse_args()
    report = audit_root(
        args.root.resolve(), target_hz=args.target_hz,
        verify_hashes=not args.skip_hash_verification,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "event_count", "ground_truth_counts", "training_eligible_count",
        "cadence_eligible_count", "integrity_error_count", "decision",
    )}, ensure_ascii=False, indent=2))
    print(f"report: {args.out.resolve()}")
    return 2 if report["integrity_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
