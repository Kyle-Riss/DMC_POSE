#!/usr/bin/env python3
"""Benchmark post-clip plus delta Swin3D embeddings with held-session CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_swin3d_reviewed_single_group_cv import fit, probability
from scripts.train_swin3d_verifier_probe import metrics, operating_point


C_VALUES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01)


def staged_hybrid(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(root / f"{split}.npz")
    metadata = json.loads((root / f"{split}_metadata.json").read_text(encoding="utf-8"))
    x = arrays["x"].astype(np.float32)
    groups: dict[tuple[str, bool], list[tuple[int, dict]]] = {}
    for index, row in enumerate(metadata):
        key = (str(row["video_id"]), bool(row.get("horizontal_flip", False)))
        groups.setdefault(key, []).append((index, row))
    features, labels = [], []
    for items in groups.values():
        prefall = [item for item in items if item[1]["label_source"] == "same_recording_prefall"]
        falls = [item for item in items if int(item[1]["label"]) == 1]
        hardneg = sorted(
            [item for item in items if item[1]["label_source"] == "reviewed_hard_negative"],
            key=lambda item: int(item[1]["start_frame"]),
        )
        if prefall and falls:
            for baseline_index, _ in prefall:
                for post_index, _ in falls:
                    features.append(
                        np.concatenate([x[post_index], x[post_index] - x[baseline_index]])
                    )
                    labels.append(1)
        elif len(hardneg) >= 3:
            for left, right in ((0, 1), (1, 2), (0, 2)):
                left_index, right_index = hardneg[left][0], hardneg[right][0]
                features.append(
                    np.concatenate([x[right_index], x[right_index] - x[left_index]])
                )
                labels.append(0)
    return np.asarray(features, np.float32), np.asarray(labels, np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single",
        type=Path,
        default=PROJECT_ROOT
        / "external_datasets/features/swin3d_b_verifier/ai_runner_reviewed_single_v1.npz",
    )
    parser.add_argument(
        "--delta",
        type=Path,
        default=PROJECT_ROOT
        / "external_datasets/features/swin3d_b_verifier/ai_runner_reviewed_v1.npz",
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
        / "runs/video_verifier/swin3d_b_reviewed_hybrid_group_cv_v1_20260828",
    )
    parser.add_argument("--min-cv-recall", type=float, default=0.8)
    args = parser.parse_args()

    single, delta = np.load(args.single), np.load(args.delta)
    if not np.array_equal(single["event_id"], delta["event_id"]):
        raise ValueError("reviewed single/delta event order mismatch")
    x_reviewed = np.concatenate([single["x"], delta["x"]], axis=1).astype(np.float32)
    y_reviewed = single["y"].astype(np.int64)
    groups = single["session_group"].astype(str)
    event_ids = single["event_id"].astype(str)
    staged = {
        split: staged_hybrid(args.staged, split)
        for split in ("train", "val", "test")
    }
    positive_groups = sorted(set(groups[y_reviewed == 1]))
    negative_groups = sorted(set(groups[y_reviewed == 0]) - set(positive_groups))
    negative_fold = {group: index % 2 for index, group in enumerate(negative_groups)}

    best, search = None, []
    for c in C_VALUES:
        held_rows = []
        for fold, positive_group in enumerate(positive_groups):
            test_mask = (groups == positive_group) | np.asarray(
                [
                    target == 0
                    and group not in positive_groups
                    and negative_fold.get(group) == fold
                    for target, group in zip(y_reviewed, groups)
                ],
                dtype=bool,
            )
            train_mask = ~test_mask
            train_x = np.concatenate([staged["train"][0], x_reviewed[train_mask]])
            train_y = np.concatenate([staged["train"][1], y_reviewed[train_mask]])
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
        held_y = np.asarray([row["target"] for row in held_rows], np.int64)
        held_probability = np.asarray([row["probability"] for row in held_rows])
        point = operating_point(held_y, held_probability, args.min_cv_recall)
        auc = float(roc_auc_score(held_y, held_probability))
        search.append({"c": c, "roc_auc": auc, **point})
        key = (point["f1"], point["precision"], auc, -c)
        if best is None or key > best[0]:
            best = (key, c, point, held_rows)

    assert best is not None
    _, selected_c, selected_point, selected_rows = best
    threshold = float(selected_point["threshold"])
    final_x = np.concatenate([staged["train"][0], x_reviewed])
    final_y = np.concatenate([staged["train"][1], y_reviewed])
    scaler, model = fit(final_x, final_y, selected_c)
    held_y = np.asarray([row["target"] for row in selected_rows], np.int64)
    held_probability = np.asarray([row["probability"] for row in selected_rows])
    val_probability = probability(scaler, model, staged["val"][0])
    test_probability = probability(scaler, model, staged["test"][0])
    report = {
        "schema_version": "dmc_swin3d_reviewed_hybrid_group_cv_v1",
        "architecture": "swin3d_b_frozen_post_plus_delta_logistic_v1",
        "selected_c": selected_c,
        "threshold": threshold,
        "held_session_cv": metrics(held_y, held_probability, threshold),
        "staged_validation": metrics(staged["val"][1], val_probability, threshold),
        "staged_test": metrics(staged["test"][1], test_probability, threshold),
        "search": search,
        "held_session_predictions": selected_rows,
        "promotion_eligible": False,
        "runtime_supported": False,
        "warnings": [
            "only two reviewed positive recording sessions are available",
            "hybrid feature mode is diagnostic and is not wired into runtime",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "hybrid_probe.npz",
        feature_mode=np.asarray(["hybrid_embedding_v1"]),
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
        "runtime_supported": False,
        "output": str((args.out_dir / "hybrid_probe.npz").resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
