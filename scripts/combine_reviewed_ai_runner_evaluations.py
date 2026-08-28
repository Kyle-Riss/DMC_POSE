#!/usr/bin/env python3
"""Combine disjoint reviewed-event evaluation reports without rerunning the backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def metrics(rows: list[dict], threshold: float) -> dict:
    y = np.asarray([row["target"] for row in rows], dtype=np.int64)
    probability = np.asarray([row["probability"] for row in rows], dtype=np.float64)
    prediction = (probability >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "events": int(len(rows)),
        "negative": int((y == 0).sum()),
        "positive": int((y == 1).sum()),
        "accuracy": round(float(accuracy_score(y, prediction)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument(
        "--out",
        type=Path,
        default=project / "runs/performance/current_20260828/ai_runner_reviewed_combined.json",
    )
    args = parser.parse_args()

    rows_by_id: dict[str, dict] = {}
    threshold = None
    sources = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        current_threshold = float(report["threshold"])
        if threshold is None:
            threshold = current_threshold
        elif not np.isclose(threshold, current_threshold):
            raise ValueError(f"threshold mismatch: {threshold} != {current_threshold}")
        sources.append(str(path.resolve()))
        for row in report["events"]:
            if "target" not in row:
                continue
            event_id = row["event_id"]
            previous = rows_by_id.get(event_id)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting event result: {event_id}")
            rows_by_id[event_id] = row

    rows = list(rows_by_id.values())
    high = [row for row in rows if row["review_confidence"] == "high"]
    pilot_ids = {
        "bed_001_fall_20260814_091245",
        "bed_001_fall_20260814_091316",
        "bed_001_fall_20260814_091342",
    }
    novel = [row for row in rows if row["event_id"] not in pilot_ids]
    result = {
        "schema_version": "dmc_ai_runner_reviewed_swin3d_combined_v1",
        "model": "swin3d_b_frozen_delta_embedding_logistic_v1",
        "threshold": threshold,
        "sources": sources,
        "all_reviewed": metrics(rows, threshold),
        "high_confidence_review_only": metrics(high, threshold),
        "excluding_known_pilot_duplicates": metrics(novel, threshold),
        "events": rows,
        "promotion_eligible": False,
        "warnings": [
            "events were selected from automatic fall candidates",
            "same-session and cross-camera correlation is present",
            "retrospective first-to-last frame scoring differs from live post-trigger scoring",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "all_reviewed": result["all_reviewed"],
        "high_confidence_review_only": result["high_confidence_review_only"],
        "excluding_known_pilot_duplicates": result["excluding_known_pilot_duplicates"],
    }, ensure_ascii=False, indent=2))
    print(f"report: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
