#!/usr/bin/env python3
"""Rank unreviewed AI_runner fall candidates using corrected temporal sample fields."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


EVENT_RE = re.compile(r"^(bed_[^_]+)_fall_(\d{8})_(\d{6})$")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def reviewed_ids(review_dir: Path) -> set[str]:
    result: set[str] = set()
    for path in review_dir.glob("reviewed_labels*.json"):
        result.update(read_json(path).get("labels", {}))
    return result


def record(path: Path) -> dict | None:
    match = EVENT_RE.match(path.name)
    if not match:
        return None
    payload = read_json(path / "fall_result.json")
    result = payload.get("result") or {}
    samples = payload.get("samples") or result.get("samples") or []
    positives = [row for row in samples if row.get("positive")]
    confidences = [
        float(row["confidence"]) for row in positives if row.get("confidence") is not None
    ]
    boxes = [
        row["boundingBox"] for row in positives
        if row.get("boundingBox") and len(row["boundingBox"]) == 4
    ]
    centers = np.asarray(
        [(float(box[1]) + float(box[3])) / 2.0 for box in boxes],
        dtype=np.float32,
    )
    heights = np.asarray(
        [float(box[3]) - float(box[1]) for box in boxes],
        dtype=np.float32,
    )
    last = result.get("lastResult") or {}
    return {
        "event_id": path.name,
        "date": match.group(2),
        "time": match.group(3),
        "bed_id": match.group(1),
        "samples": len(samples),
        "positive_samples": len(positives),
        "positive_ratio": round(len(positives) / max(1, len(samples)), 4),
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "last_confidence": round(float(last.get("confidence") or 0.0), 4),
        "center_drop": round(float(centers[-1] - centers[0]), 2) if len(centers) >= 2 else 0.0,
        "center_range": round(float(np.ptp(centers)), 2) if len(centers) >= 2 else 0.0,
        "height_change": round(float(heights[-1] - heights[0]), 2) if len(heights) >= 2 else 0.0,
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/dmc/AI/AI_runner/data/events/fall"))
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=project / "runtime_data/ai_runner_fall_review_v1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project / "runtime_data/ai_runner_fall_review_v1/unreviewed_ranked.csv",
    )
    args = parser.parse_args()

    reviewed = reviewed_ids(args.review_dir)
    rows = [
        row for path in args.root.iterdir()
        if path.is_dir() and path.name not in reviewed
        if (row := record(path)) is not None
    ]
    rows.sort(
        key=lambda row: (
            -row["center_drop"],
            -row["center_range"],
            -row["positive_ratio"],
            -row["mean_confidence"],
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "total_events": len(rows) + len(reviewed),
        "reviewed_events": len(reviewed),
        "unreviewed_events": len(rows),
        "output": str(args.out.resolve()),
        "top": rows[:20],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
