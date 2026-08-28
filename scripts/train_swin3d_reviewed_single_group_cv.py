#!/usr/bin/env python3
"""Train a single-clip Swin3D probe with held-out positive-session CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_swin3d_verifier_probe import metrics, operating_point


C_VALUES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03)


def fit(x: np.ndarray, y: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=c, class_weight="balanced", max_iter=20000, random_state=20260828
    ).fit(scaler.transform(x), y)
    return scaler, model


def probability(
    scaler: StandardScaler, model: LogisticRegression, x: np.ndarray
) -> np.ndarray:
    return model.predict_proba(scaler.transform(x))[:, 1]


def load_staged(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(root / f"{split}.npz")
    return arrays["x"].astype(np.float32), arrays["y"].astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=PROJECT_ROOT
        / "external_datasets/features/swin3d_b_verifier/ai_runner_reviewed_single_v1.npz",
    )
    parser.add_argument(
        "--staged",
        type=Path,
        default=PROJECT_ROOT
        / "external_datasets/features/swin3d_b_verifier/staged_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/video_verifier/swin3d_b_reviewed_single_group_cv_v1_20260828",
    )
    parser.add_argument("--min-cv-recall", type=float, default=0.8)
    args = parser.parse_args()

    reviewed = np.load(args.reviewed)
    x_reviewed = reviewed["x"].astype(np.float32)
    y_reviewed = reviewed["y"].astype(np.int64)
    groups = reviewed["session_group"].astype(str)
    event_ids = reviewed["event_id"].astype(str)
    staged_train_x, staged_train_y = load_staged(args.staged, "train")
    staged_val_x, staged_val_y = load_staged(args.staged, "val")
    staged_test_x, staged_test_y = load_staged(args.staged, "test")

    positive_groups = sorted(set(groups[y_reviewed == 1]))
    if len(positive_groups) != 2:
        raise ValueError(f"expected two positive session groups, got {positive_groups}")
    negative_groups = sorted(set(groups[y_reviewed == 0]))
    negative_fold = {group: index % 2 for index, group in enumerate(negative_groups)}

    best = None
    search = []
    for c in C_VALUES:
        held_rows = []
        for fold, positive_group in enumerate(positive_groups):
            test_mask = (groups == positive_group) | np.asarray(
                [
                    target == 0 and negative_fold.get(group) == fold
                    for target, group in zip(y_reviewed, groups)
                ],
                dtype=bool,
            )
            train_mask = ~test_mask
            train_x = np.concatenate([staged_train_x, x_reviewed[train_mask]])
            train_y = np.concatenate([staged_train_y, y_reviewed[train_mask]])
            scaler, model = fit(train_x, train_y, c)
            scores = probability(scaler, model, x_reviewed[test_mask])
            for event_id, target, score in zip(
                event_ids[test_mask], y_reviewed[test_mask], scores
            ):
                held_rows.append(
                    {
                        "fold": fold,
                        "held_positive_group": positive_group,
                        "event_id": event_id,
                        "target": int(target),
                        "probability": float(score),
                    }
                )
        held_y = np.asarray([row["target"] for row in held_rows], dtype=np.int64)
        held_probability = np.asarray(
            [row["probability"] for row in held_rows], dtype=np.float64
        )
        point = operating_point(held_y, held_probability, args.min_cv_recall)
        auc = float(roc_auc_score(held_y, held_probability))
        row = {"c": c, "roc_auc": auc, **point}
        search.append(row)
        key = (point["f1"], point["precision"], auc, -c)
        if best is None or key > best[0]:
            best = (key, c, point, held_rows)

    assert best is not None
    _, selected_c, selected_point, selected_rows = best
    threshold = float(selected_point["threshold"])
    final_x = np.concatenate([staged_train_x, x_reviewed])
    final_y = np.concatenate([staged_train_y, y_reviewed])
    scaler, model = fit(final_x, final_y, selected_c)
    held_y = np.asarray([row["target"] for row in selected_rows], dtype=np.int64)
    held_probability = np.asarray(
        [row["probability"] for row in selected_rows], dtype=np.float64
    )
    val_probability = probability(scaler, model, staged_val_x)
    test_probability = probability(scaler, model, staged_test_x)
    report = {
        "schema_version": "dmc_swin3d_reviewed_single_group_cv_probe_v1",
        "architecture": "swin3d_b_frozen_single_embedding_logistic_v1",
        "selected_c": selected_c,
        "threshold": threshold,
        "threshold_policy": (
            f"max precision with held-session CV recall >= {args.min_cv_recall}"
        ),
        "reviewed_counts": {
            "events": int(len(y_reviewed)),
            "positive": int(y_reviewed.sum()),
            "negative": int(len(y_reviewed) - y_reviewed.sum()),
            "positive_session_groups": positive_groups,
        },
        "held_session_cv": metrics(held_y, held_probability, threshold),
        "staged_validation": metrics(staged_val_y, val_probability, threshold),
        "staged_test": metrics(staged_test_y, test_probability, threshold),
        "search": search,
        "held_session_predictions": selected_rows,
        "promotion_eligible": False,
        "authority": "telemetry_only",
        "warnings": [
            "only two reviewed positive recording sessions are available",
            "final model requires a new untouched positive session",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "linear_probe.npz",
        feature_mode=np.asarray(["single_embedding_v1"]),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coefficient=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float32),
        threshold=np.asarray([threshold], dtype=np.float32),
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected_c": selected_c,
        "threshold": threshold,
        "held_session_cv": report["held_session_cv"],
        "staged_validation": report["staged_validation"],
        "staged_test": report["staged_test"],
        "promotion_eligible": False,
        "output": str((args.out_dir / "linear_probe.npz").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
