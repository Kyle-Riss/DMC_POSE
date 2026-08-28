#!/usr/bin/env python3
"""Train a regularized binary probe over frozen Swin3D-B embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


def operating_point(labels: np.ndarray, probability: np.ndarray, minimum_recall: float = 0.9) -> dict:
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    rows = []
    for threshold in candidates:
        prediction = (probability >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
        if recall >= minimum_recall:
            rows.append((float(precision), float(f1), float(threshold), float(recall)))
    if not rows:
        raise ValueError("no threshold satisfies minimum recall")
    precision, f1, threshold, recall = max(rows)
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}


def metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(labels, prediction)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(labels, probability)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, default=project / "external_datasets/features/swin3d_b_verifier/staged_v1")
    parser.add_argument("--out-dir", type=Path, default=project / "runs/video_verifier/swin3d_b_staged_linear_v1_20260828")
    parser.add_argument("--min-val-recall", type=float, default=0.9)
    args = parser.parse_args()
    arrays = {split: np.load(args.embeddings_dir / f"{split}.npz") for split in ("train", "val", "test")}
    scaler = StandardScaler().fit(arrays["train"]["x"])
    x = {split: scaler.transform(arrays[split]["x"]) for split in arrays}
    y = {split: arrays[split]["y"] for split in arrays}
    searches = []
    best = None
    for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=20260828)
        model.fit(x["train"], y["train"])
        probability = model.predict_proba(x["val"])[:, 1]
        point = operating_point(y["val"], probability, args.min_val_recall)
        auc = float(roc_auc_score(y["val"], probability))
        row = {"c": c, "val_auc": auc, **point}
        searches.append(row)
        key = (auc, point["precision"], point["f1"], -c)
        if best is None or key > best[0]:
            best = (key, model, row)
    _, model, selected = best
    threshold = selected["threshold"]
    probabilities = {split: model.predict_proba(x[split])[:, 1] for split in arrays}
    report = {
        "schema_version": "dmc_swin3d_verifier_probe_v1",
        "architecture": "swin3d_b_frozen_linear_probe_v1",
        "backbone_weights": "KINETICS400_IMAGENET22K_V1",
        "parameter_count_trainable": int(model.coef_.size + model.intercept_.size),
        "selected_c": selected["c"],
        "threshold_policy": f"max precision with validation recall >= {args.min_val_recall}",
        "validation": metrics(y["val"], probabilities["val"], threshold),
        "test": metrics(y["test"], probabilities["test"], threshold),
        "search": searches,
        "promotion_eligible": False,
        "authority": "telemetry_only",
        "warnings": ["subject identity unknown", "multiview clips correlated", "engineering diagnostic only"],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "linear_probe.npz",
        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
        coefficient=model.coef_.astype(np.float32), intercept=model.intercept_.astype(np.float32),
        threshold=np.asarray([threshold], dtype=np.float32),
    )
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for split in ("val", "test"):
        np.savez_compressed(args.out_dir / f"{split}_predictions.npz", y=y[split], probability=probabilities[split])
    print(json.dumps({key: report[key] for key in ("architecture", "selected_c", "validation", "test", "promotion_eligible")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
