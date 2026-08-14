#!/usr/bin/env python3
"""Evaluate reviewed shadow alerts against bed-hours and actual-event logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadow_review import evaluate_operations, read_csv_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("runtime_data/shadow_summary.json"))
    parser.add_argument("--review", type=Path, default=Path("runtime_data/shadow_review.csv"))
    parser.add_argument("--events", type=Path, default=Path("runtime_data/actual_events.csv"))
    parser.add_argument("--out", type=Path, default=Path("runtime_data/operational_report.json"))
    parser.add_argument("--min-bed-hours", type=float, default=168.0)
    parser.add_argument("--max-false-alarms-per-bed-hour", type=float, default=0.01)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument(
        "--policy-version", default="hybrid_v2_structural_confirm",
        help="Fusion policy to evaluate; use 'all' only for historical inspection.",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    report = evaluate_operations(
        summary,
        read_csv_rows(args.review),
        read_csv_rows(args.events),
        min_bed_hours=args.min_bed_hours,
        max_false_alarms_per_bed_hour=args.max_false_alarms_per_bed_hour,
        min_sensitivity=args.min_sensitivity,
        policy_version=None if args.policy_version == "all" else args.policy_version,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

