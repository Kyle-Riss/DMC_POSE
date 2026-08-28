#!/usr/bin/env python3
"""Evaluate a single-clip Swin3D probe on all manually reviewed AI_runner events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.combine_reviewed_ai_runner_evaluations import metrics
from swin3d_verifier import Swin3DVerifierService


def load_labels(paths: list[Path]) -> dict[str, list[str]]:
    labels = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for event_id, value in document["labels"].items():
            previous = labels.get(event_id)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting labels for {event_id}")
            labels[event_id] = value
    return labels


def read_frames(path: Path) -> list[np.ndarray]:
    frames = []
    for frame_path in sorted((path / "frames").glob("*.jpg")):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            frames.append(frame)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    review_dir = PROJECT_ROOT / "runtime_data/ai_runner_fall_review_v1"
    parser.add_argument(
        "--labels", type=Path, nargs="+", default=sorted(review_dir.glob("reviewed_labels*.json"))
    )
    parser.add_argument(
        "--events-root", type=Path, default=Path("/home/dmc/AI/AI_runner/data/events/fall")
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=PROJECT_ROOT / "external_models/torchvision/swin3d_b_22k-7c6ae6fa.pth",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=PROJECT_ROOT
        / "runs/video_verifier/swin3d_b_staged_linear_v1_20260828/linear_probe.npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "runs/performance/current_20260828/ai_runner_reviewed_single_probe_71.json",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels)
    service = Swin3DVerifierService(args.weight, args.probe, device=args.device)
    rows = []
    for event_id, (label, confidence) in labels.items():
        frames = read_frames(args.events_root / event_id)
        if len(frames) < 16:
            continue
        result = service.predict(frames)
        rows.append(
            {
                "event_id": event_id,
                "target": int(label == "fall"),
                "review_label": label,
                "review_confidence": confidence,
                "probability": round(float(result.probability), 6),
                "predicted": bool(result.probability >= service.threshold),
                "latency_ms": round(float(result.latency_ms), 3),
                "decoded_frames": len(frames),
            }
        )
    high = [row for row in rows if row["review_confidence"] == "high"]
    report = {
        "schema_version": "dmc_ai_runner_reviewed_swin3d_single_probe_v1",
        "model": "swin3d_b_frozen_single_embedding_logistic_v1",
        "threshold": service.threshold,
        "evaluation_contract": "16 frames sampled across the entire stored event",
        "all_reviewed": metrics(rows, service.threshold),
        "high_confidence_review_only": metrics(high, service.threshold),
        "events": rows,
        "promotion_eligible": False,
        "warnings": [
            "same-session and cross-camera correlation is present",
            "events were selected from automatic fall candidates",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "threshold": service.threshold,
        "all_reviewed": report["all_reviewed"],
        "high_confidence_review_only": report["high_confidence_review_only"],
        "output": str(args.out.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
