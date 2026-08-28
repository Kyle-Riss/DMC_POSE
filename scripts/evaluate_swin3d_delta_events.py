#!/usr/bin/env python3
"""Evaluate the staged Swin3D delta probe without counting correlated pairs as events."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-value))


def metric_row(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "events": int(len(labels)),
        "accuracy": round((tp + tn) / len(labels), 4) if len(labels) else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall
        else 0.0,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def annotation_statuses(path: Path) -> dict[str, str]:
    statuses: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            statuses[row["recording_id"]].add(row["annotation_status"])
    result = {}
    for recording, values in statuses.items():
        if len(values) != 1:
            raise ValueError(f"annotation disagreement for {recording}: {sorted(values)}")
        result[recording] = next(iter(values))
    return result


def paired_rows(x: np.ndarray, metadata: list[dict]) -> list[dict]:
    groups: dict[tuple[str, bool], list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[(str(row["video_id"]), bool(row.get("horizontal_flip", False)))].append((index, row))
    output = []
    for (video_id, horizontal_flip), items in groups.items():
        prefall = [item for item in items if item[1]["label_source"] == "same_recording_prefall"]
        falls = [item for item in items if int(item[1]["label"]) == 1]
        hardneg = sorted(
            [item for item in items if item[1]["label_source"] == "reviewed_hard_negative"],
            key=lambda item: int(item[1]["start_frame"]),
        )
        pairs: list[tuple[int, int, int]] = []
        if prefall and falls:
            pairs = [(left[0], right[0], 1) for left in prefall for right in falls]
        elif len(hardneg) >= 3:
            pairs = [
                (hardneg[left][0], hardneg[right][0], 0)
                for left, right in ((0, 1), (1, 2), (0, 2))
            ]
        for left, right, label in pairs:
            row = items[0][1]
            output.append({
                "feature": x[right] - x[left],
                "label": label,
                "recording_id": str(row["recording_id"]),
                "camera_id": str(row["camera_id"]),
                "video_id": video_id,
                "horizontal_flip": horizontal_flip,
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=PROJECT_ROOT / "external_datasets/features/swin3d_b_verifier/staged_v1",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=PROJECT_ROOT / "runs/video_verifier/swin3d_b_staged_delta_v2_20260828/delta_probe.npz",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "runs/video_verifier/swin3d_b_staged_delta_v2_20260828/event_test_report.json",
    )
    args = parser.parse_args()

    arrays = np.load(args.embeddings_dir / f"{args.split}.npz")
    metadata = json.loads(
        (args.embeddings_dir / f"{args.split}_metadata.json").read_text(encoding="utf-8")
    )
    pairs = paired_rows(arrays["x"], metadata)
    if not pairs:
        raise ValueError("no evaluable embedding pairs")

    probe = np.load(args.probe)
    features = np.asarray([row["feature"] for row in pairs], dtype=np.float32)
    scaled = (features - probe["mean"]) / probe["scale"]
    probabilities = sigmoid(scaled @ probe["coefficient"].reshape(-1) + float(probe["intercept"][0]))
    threshold = float(probe["threshold"][0])
    for row, probability in zip(pairs, probabilities, strict=True):
        row["probability"] = float(probability)
        del row["feature"]

    camera_pairs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in pairs:
        camera_pairs[(row["recording_id"], row["camera_id"])].append(row)
    camera_scores: dict[str, dict[str, float]] = defaultdict(dict)
    labels = {}
    for (recording, camera), rows in camera_pairs.items():
        camera_scores[recording][camera] = max(float(row["probability"]) for row in rows)
        values = {int(row["label"]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"label disagreement for {recording}/{camera}")
        labels[recording] = next(iter(values))

    statuses = annotation_statuses(args.annotations)
    event_rows = []
    for recording in sorted(camera_scores):
        values = np.asarray(list(camera_scores[recording].values()), dtype=np.float64)
        event_rows.append({
            "recording_id": recording,
            "label": labels[recording],
            "annotation_status": statuses.get(recording, "unknown"),
            "camera_scores": dict(sorted(camera_scores[recording].items())),
            "any_view_probability": float(np.max(values)),
            "median_view_probability": float(np.median(values)),
            "all_views_probability": float(np.min(values)),
        })

    event_labels = np.asarray([row["label"] for row in event_rows], dtype=np.int64)
    aggregations = {}
    for name in ("any_view", "median_view", "all_views"):
        event_probability = np.asarray(
            [row[f"{name}_probability"] for row in event_rows], dtype=np.float64
        )
        aggregations[name] = metric_row(event_labels, event_probability, threshold)

    strict_rows = [
        row for row in event_rows
        if row["annotation_status"] in {"complete", "excluded"}
    ]
    strict = {}
    if strict_rows:
        strict_labels = np.asarray([row["label"] for row in strict_rows], dtype=np.int64)
        for name in ("any_view", "median_view", "all_views"):
            values = np.asarray([row[f"{name}_probability"] for row in strict_rows])
            strict[name] = metric_row(strict_labels, values, threshold)

    report = {
        "schema_version": "dmc_swin3d_delta_event_evaluation_v1",
        "split": args.split,
        "threshold": threshold,
        "pair_count": len(pairs),
        "camera_count": len(camera_pairs),
        "event_count": len(event_rows),
        "aggregation_contract": "max pair within camera; then max/median/min across synchronized views",
        "all_reviewed_events": aggregations,
        "strict_complete_or_hard_negative_events": strict,
        "events": event_rows,
        "promotion_eligible": False,
        "warnings": [
            "recording-disjoint but subject identity is unknown",
            "test event count is small",
            "engineering regression metric only",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
