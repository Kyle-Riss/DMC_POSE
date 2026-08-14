#!/usr/bin/env python3
"""Create or update label-preserving shadow review and actual-event CSV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from shadow_review import (
    EVENT_FIELDS,
    REVIEW_FIELDS,
    prepare_review_rows,
    read_csv_rows,
    write_csv_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("runtime_data/shadow_summary.json"))
    parser.add_argument("--review", type=Path, default=Path("runtime_data/shadow_review.csv"))
    parser.add_argument("--events", type=Path, default=Path("runtime_data/actual_events.csv"))
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = prepare_review_rows(
        list(summary.get("review_candidates") or []),
        read_csv_rows(args.review),
    )
    write_csv_rows(args.review, REVIEW_FIELDS, rows)
    existing_events = read_csv_rows(args.events)
    write_csv_rows(args.events, EVENT_FIELDS, existing_events)
    print(json.dumps({
        "review_file": str(args.review),
        "candidate_count": len(rows),
        "pending_count": sum(row["label"] == "pending" for row in rows),
        "events_file": str(args.events),
        "existing_event_count": len(existing_events),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

