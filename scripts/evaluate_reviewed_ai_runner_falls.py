#!/usr/bin/env python3
"""Evaluate manually reviewed AI_runner event frames with the current Swin3D delta probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from swin3d_verifier import Swin3DVerifierService


def read_frames(event_dir: Path) -> list[np.ndarray]:
    frames = []
    for path in sorted((event_dir / "frames").glob("*.jpg")):
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
    return frames


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
        "roc_auc": round(float(roc_auc_score(y, probability)), 4) if len(np.unique(y)) == 2 else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path,
                        default=project / "runtime_data/ai_runner_fall_review_v1/reviewed_labels_v1.json")
    parser.add_argument("--events-root", type=Path,
                        default=Path("/home/dmc/AI/AI_runner/data/events/fall"))
    parser.add_argument("--weight", type=Path,
                        default=project / "external_models/torchvision/swin3d_b_22k-7c6ae6fa.pth")
    parser.add_argument("--probe", type=Path,
                        default=project / "runs/video_verifier/swin3d_b_staged_delta_v2_20260828/delta_probe.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path,
                        default=project / "runs/performance/current_20260828/ai_runner_reviewed_events.json")
    args = parser.parse_args()

    label_doc = json.loads(args.labels.read_text(encoding="utf-8"))
    service = Swin3DVerifierService(args.weight, args.probe, device=args.device)
    rows = []
    for event_id, label_value in label_doc["labels"].items():
        label, review_confidence = label_value
        frames = read_frames(args.events_root / event_id)
        if len(frames) < 32:
            rows.append({"event_id": event_id, "excluded": "fewer_than_32_decoded_frames"})
            continue
        prediction = service.predict_pair(frames[:16], frames[-16:])
        rows.append({
            "event_id": event_id,
            "target": int(label == "fall"),
            "review_label": label,
            "review_confidence": review_confidence,
            "probability": round(float(prediction.probability), 6),
            "predicted": bool(prediction.probability >= service.threshold),
            "latency_ms": round(float(prediction.latency_ms), 3),
            "decoded_frames": len(frames),
        })
    evaluated = [row for row in rows if "target" in row]
    high = [row for row in evaluated if row["review_confidence"] == "high"]
    pilot_ids = {
        "bed_001_fall_20260814_091245",
        "bed_001_fall_20260814_091316",
        "bed_001_fall_20260814_091342",
    }
    novel = [row for row in evaluated if row["event_id"] not in pilot_ids]
    report = {
        "schema_version": "dmc_ai_runner_reviewed_swin3d_evaluation_v1",
        "model": "swin3d_b_frozen_delta_embedding_logistic_v1",
        "threshold": service.threshold,
        "device": args.device,
        "evaluation_contract": "retrospective first 16 stored frames versus last 16 stored frames",
        "all_reviewed": metrics(evaluated, service.threshold),
        "high_confidence_review_only": metrics(high, service.threshold),
        "excluding_known_pilot_duplicates": metrics(novel, service.threshold),
        "events": rows,
        "promotion_eligible": False,
        "warnings": [
            "review sample was selected from automatic fall candidates and is prevalence-biased",
            "two positive labels have medium visual-review confidence",
            "the retrospective first-to-last frame contract differs from the intended live post-trigger contract",
            "subject/session identity and camera independence are not fully known",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "threshold", "evaluation_contract", "all_reviewed",
        "high_confidence_review_only", "excluding_known_pilot_duplicates",
    )}, ensure_ascii=False, indent=2))
    print(f"report: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
