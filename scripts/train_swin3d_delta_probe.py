#!/usr/bin/env python3
"""Train an event-level fall probe from post-minus-baseline Swin3D embeddings."""

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


def paired_embeddings(x: np.ndarray, metadata: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[str, bool], list[tuple[int, dict]]] = {}
    for index, row in enumerate(metadata):
        key = (str(row["video_id"]), bool(row.get("horizontal_flip", False)))
        groups.setdefault(key, []).append((index, row))
    features: list[np.ndarray] = []
    labels: list[int] = []
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
                    features.append(x[post_index] - x[baseline_index])
                    labels.append(1)
        elif len(hardneg) >= 3:
            for left, right in ((0, 1), (1, 2), (0, 2)):
                features.append(x[hardneg[right][0]] - x[hardneg[left][0]])
                labels.append(0)
    if not features:
        raise ValueError("no event-level embedding pairs were built")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def load_pairs(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(root / f"{split}.npz")
    metadata = json.loads((root / f"{split}_metadata.json").read_text(encoding="utf-8"))
    return paired_embeddings(arrays["x"], metadata)


def main() -> int:
    project = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, default=project / "external_datasets/features/swin3d_b_verifier/staged_v1")
    parser.add_argument("--out-dir", type=Path, default=project / "runs/video_verifier/swin3d_b_staged_delta_v2_20260828")
    parser.add_argument("--min-val-recall", type=float, default=0.9)
    args = parser.parse_args()
    arrays = {split: load_pairs(args.embeddings_dir, split) for split in ("train", "val", "test")}
    scaler = StandardScaler().fit(arrays["train"][0])
    scaled = {split: scaler.transform(value[0]) for split, value in arrays.items()}
    search = []
    best = None
    for c in (1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        model = LogisticRegression(
            C=c, class_weight="balanced", max_iter=10000, random_state=20260828,
        ).fit(scaled["train"], arrays["train"][1])
        probability = model.predict_proba(scaled["val"])[:, 1]
        point = operating_point(arrays["val"][1], probability, args.min_val_recall)
        auc = float(roc_auc_score(arrays["val"][1], probability))
        row = {"c": c, "val_auc": auc, **point}
        search.append(row)
        key = (auc, point["precision"], point["f1"], -c)
        if best is None or key > best[0]:
            best = (key, model, row)
    _, model, selected = best
    threshold = float(selected["threshold"])
    probability = {
        split: model.predict_proba(scaled[split])[:, 1]
        for split in arrays
    }
    report = {
        "schema_version": "dmc_swin3d_delta_probe_v1",
        "architecture": "swin3d_b_frozen_delta_embedding_logistic_v1",
        "backbone_weights": "KINETICS400_IMAGENET22K_V1",
        "feature_mode": "delta_embedding_v1",
        "parameter_count_trainable": int(model.coef_.size + model.intercept_.size),
        "selected_c": selected["c"],
        "threshold_policy": f"max precision with validation recall >= {args.min_val_recall}",
        "pair_counts": {split: int(len(arrays[split][1])) for split in arrays},
        "validation": metrics(arrays["val"][1], probability["val"], threshold),
        "test": metrics(arrays["test"][1], probability["test"], threshold),
        "search": search,
        "promotion_eligible": False,
        "authority": "telemetry_only",
        "warnings": [
            "subject identity unknown",
            "multiview pairs are correlated",
            "engineering diagnostic only",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "delta_probe.npz",
        feature_mode=np.asarray(["delta_embedding_v1"]),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coefficient=model.coef_.astype(np.float32),
        intercept=model.intercept_.astype(np.float32),
        threshold=np.asarray([threshold], dtype=np.float32),
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for split in ("val", "test"):
        np.savez_compressed(
            args.out_dir / f"{split}_predictions.npz",
            y=arrays[split][1], probability=probability[split],
        )
    print(json.dumps({
        "architecture": report["architecture"],
        "selected_c": report["selected_c"],
        "pair_counts": report["pair_counts"],
        "validation": report["validation"],
        "test": report["test"],
        "promotion_eligible": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
