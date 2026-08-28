#!/usr/bin/env python3
"""Deduplicate held-session predictions and recalibrate a shadow probe threshold."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from scripts.train_swin3d_verifier_probe import metrics, operating_point


EVENT_RE = re.compile(r"^bed_[^_]+_fall_(\d{8})_(\d{2})\d{4}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--min-recall", type=float, default=0.8)
    args = parser.parse_args()

    source_report = json.loads(
        (args.source_dir / "report.json").read_text(encoding="utf-8")
    )
    candidates: dict[str, list[dict]] = {}
    for row in source_report["held_session_predictions"]:
        candidates.setdefault(row["event_id"], []).append(row)
    rows = []
    for event_id, values in candidates.items():
        match = EVENT_RE.match(event_id)
        session_group = f"{match.group(1)}_{match.group(2)}" if match else None
        preferred = [
            row for row in values if row["held_positive_group"] == session_group
        ]
        rows.append(preferred[0] if preferred else values[0])

    y = np.asarray([row["target"] for row in rows], dtype=np.int64)
    probability = np.asarray([row["probability"] for row in rows], dtype=np.float64)
    point = operating_point(y, probability, args.min_recall)
    threshold = float(point["threshold"])
    probe = np.load(args.source_dir / (
        "linear_probe.npz" if (args.source_dir / "linear_probe.npz").exists()
        else "delta_probe.npz"
    ))
    probe_name = (
        "linear_probe.npz" if "single" in source_report["architecture"]
        else "delta_probe.npz"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / probe_name,
        feature_mode=probe["feature_mode"],
        mean=probe["mean"],
        scale=probe["scale"],
        coefficient=probe["coefficient"],
        intercept=probe["intercept"],
        threshold=np.asarray([threshold], dtype=np.float32),
    )
    report = {
        **source_report,
        "schema_version": source_report["schema_version"] + "_dedup_v1",
        "threshold": threshold,
        "threshold_policy": (
            f"max precision with deduplicated held-session recall >= {args.min_recall}"
        ),
        "held_session_cv": metrics(y, probability, threshold),
        "held_session_predictions": rows,
        "deduplication": {
            "input_rows": int(sum(len(value) for value in candidates.values())),
            "unique_events": len(rows),
            "duplicate_rows_removed": int(
                sum(len(value) for value in candidates.values()) - len(rows)
            ),
        },
        "promotion_eligible": False,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "threshold": threshold,
        "held_session_cv": report["held_session_cv"],
        "deduplication": report["deduplication"],
        "output": str((args.out_dir / probe_name).resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
