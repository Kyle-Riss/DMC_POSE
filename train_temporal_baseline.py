#!/usr/bin/env python3
"""Train a leakage-safe logistic baseline for temporal windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def summarize_windows(x: np.ndarray) -> np.ndarray:
    """Convert (N,T,F) windows to deterministic non-neural statistics."""
    return np.concatenate(
        [x[:, -1], x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1)],
        axis=1,
    ).astype(np.float32)


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    prediction = (probability >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y, prediction)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def select_threshold(y: np.ndarray, probability: np.ndarray, min_recall: float) -> float:
    candidates = np.unique(np.concatenate(([0.0], probability, [1.0])))
    feasible = []
    for threshold in candidates:
        result = metrics(y, probability, float(threshold))
        if result["recall"] >= min_recall:
            feasible.append((result["precision"], result["f1"], float(threshold)))
    if feasible:
        return max(feasible)[2]
    return 0.5


def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(root / f"{split}.npz")
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--windows-dir", type=Path, default=project_root / "external_datasets/windows/gmdcsa24_3s")
    parser.add_argument("--out-dir", type=Path, default=project_root / "runs/temporal_baseline/gmdcsa24_logistic")
    parser.add_argument("--min-val-recall", type=float, default=0.90)
    args = parser.parse_args()

    train_x, train_y = load_split(args.windows_dir, "train")
    val_x, val_y = load_split(args.windows_dir, "val")
    test_x, test_y = load_split(args.windows_dir, "test")
    train_summary = summarize_windows(train_x)
    val_summary = summarize_windows(val_x)
    test_summary = summarize_windows(test_x)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, random_state=42)),
        ]
    )
    model.fit(train_summary, train_y)
    val_probability = model.predict_proba(val_summary)[:, 1]
    threshold = select_threshold(val_y, val_probability, args.min_val_recall)
    test_probability = model.predict_proba(test_summary)[:, 1]

    report = {
        "model": "logistic_window_statistics_v1",
        "windows_dir": str(args.windows_dir.resolve()),
        "input_window_shape": list(train_x.shape[1:]),
        "summary_feature_count": int(train_summary.shape[1]),
        "threshold_policy": f"max precision on validation with recall >= {args.min_val_recall}",
        "validation": metrics(val_y, val_probability, threshold),
        "test": metrics(test_y, test_probability, threshold),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out_dir / "model.joblib")
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
