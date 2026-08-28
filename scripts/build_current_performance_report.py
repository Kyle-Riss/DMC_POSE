#!/usr/bin/env python3
"""Build reproducible diagnostic performance tables and plots for current models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_swin3d_delta_probe import load_pairs
from temporal_model import architecture_from_checkpoint, build_temporal_model


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "count": int(len(y)),
        "negative": int((y == 0).sum()),
        "positive": int((y == 1).sum()),
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y, prediction)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4) if len(np.unique(y)) == 2 else None,
        "average_precision": round(float(average_precision_score(y, probability)), 4) if len(np.unique(y)) == 2 else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def swin_predictions(embeddings_dir: Path, probe_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    probe = np.load(probe_path)
    mean = probe["mean"].astype(np.float64)
    scale = probe["scale"].astype(np.float64)
    coefficient = probe["coefficient"].astype(np.float64).reshape(-1)
    intercept = float(probe["intercept"].reshape(-1)[0])
    result = {}
    for split in ("train", "val", "test"):
        x, y = load_pairs(embeddings_dir, split)
        z = ((x.astype(np.float64) - mean) / scale) @ coefficient + intercept
        probability = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        result[split] = (y.astype(np.int64), probability)
    return result


def gru_predictions(windows_dir: Path, checkpoint_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = architecture_from_checkpoint(checkpoint)
    model = build_temporal_model(architecture, int(checkpoint["feature_count"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    result = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            arrays = np.load(windows_dir / f"{split}.npz")
            x = (arrays["x"].astype(np.float32) - mean) / std
            y = arrays["y"].astype(np.int64)
            outputs = []
            for start in range(0, len(x), 256):
                logits = model(torch.from_numpy(x[start : start + 256]))
                outputs.append(torch.sigmoid(logits).numpy())
            result[split] = (y, np.concatenate(outputs))
    return result


def plot_curves(predictions: dict, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for split, (y, probability) in predictions.items():
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        axes[0].plot(fpr, tpr, label=f"{split} AUC={roc_auc_score(y, probability):.3f}")
        axes[1].plot(recall, precision, label=f"{split} AP={average_precision_score(y, probability):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_metric_bars(model_results: dict, out_path: Path) -> None:
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    fig, axes = plt.subplots(1, len(model_results), figsize=(14, 5), sharey=True)
    if len(model_results) == 1:
        axes = [axes]
    width = 0.24
    x = np.arange(len(metric_names))
    for axis, (model_name, splits) in zip(axes, model_results.items()):
        for offset, split in enumerate(("train", "val", "test")):
            values = [splits[split].get(name) or 0.0 for name in metric_names]
            axis.bar(x + (offset - 1) * width, values, width, label=split)
        axis.set_xticks(x, ["ACC", "Precision", "Recall", "F1", "AUC"], rotation=25)
        axis.set_ylim(0, 1.05)
        axis.set_title(model_name)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Diagnostic split metrics (not a clinical claim)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_gru_history(history: list[dict], out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    best_index = int(np.argmin(val_loss))
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_loss, marker="o", label="train loss")
    axis.plot(epochs, val_loss, marker="o", label="validation loss")
    axis.scatter([epochs[best_index]], [val_loss[best_index]], color="red", zorder=3,
                 label=f"best epoch={epochs[best_index]}")
    axis.set(xlabel="Epoch", ylabel="Binary cross entropy", title="10 Hz GRU learning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def markdown_table(results: dict) -> str:
    lines = [
        "| Model | Split | N | Neg/Pos | ACC | Precision | Recall | F1 | ROC-AUC | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, splits in results.items():
        for split in ("train", "val", "test"):
            row = splits[split]
            lines.append(
                f"| {model_name} | {split} | {row['count']} | {row['negative']}/{row['positive']} "
                f"| {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} "
                f"| {row['f1']:.4f} | {row['roc_auc']:.4f} | {row['average_precision']:.4f} |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_ROOT / "runs/performance/current_20260828")
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    swin_run = PROJECT_ROOT / "runs/video_verifier/swin3d_b_staged_delta_v2_20260828"
    embeddings = PROJECT_ROOT / "external_datasets/features/swin3d_b_verifier/staged_v1"
    swin_probe = np.load(swin_run / "delta_probe.npz")
    swin_threshold = float(swin_probe["threshold"].reshape(-1)[0])
    swin = swin_predictions(embeddings, swin_run / "delta_probe.npz")

    gru_run = PROJECT_ROOT / "runs/temporal_gru/usb_reviewed_10hz_small_gru_v1_20260828"
    gru_report = json.loads((gru_run / "report.json").read_text(encoding="utf-8"))
    windows = PROJECT_ROOT / "external_datasets/windows/pose_gru_109_observed_only_10hz/usb_reviewed_staged_shadow_v2_4s"
    gru = gru_predictions(windows, gru_run / "model.pt")
    gru_threshold = float(gru_report["validation"]["threshold"])

    predictions = {"Swin3D-B delta": swin, "10 Hz small GRU": gru}
    thresholds = {"Swin3D-B delta": swin_threshold, "10 Hz small GRU": gru_threshold}
    results = {
        name: {split: metrics(y, probability, thresholds[name])
               for split, (y, probability) in split_predictions.items()}
        for name, split_predictions in predictions.items()
    }

    plot_curves(swin, out_dir / "swin3d_roc_pr.png", "Swin3D-B delta verifier")
    plot_curves(gru, out_dir / "gru_roc_pr.png", "10 Hz small GRU")
    plot_metric_bars(results, out_dir / "split_metrics.png")
    plot_gru_history(gru_report["history"], out_dir / "gru_learning_curve.png")

    event_report = json.loads((swin_run / "event_test_report.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "dmc_current_performance_report_v1",
        "scope": "engineering diagnostic only",
        "models": results,
        "event_test": event_report["all_reviewed_events"]["any_view"],
        "independent_pilot": {
            "event_count": 6,
            "falls": 3,
            "normal_exits": 3,
            "accuracy": 0.8333,
            "precision": 0.75,
            "recall": 1.0,
            "confusion": {"tn": 2, "fp": 1, "fn": 0, "tp": 3},
            "contract": "early stable baseline to first 16 actual post-trigger frames",
        },
        "warnings": [
            "recording-disjoint splits are used, but subject/session identity is unknown",
            "multiview observations of one event are correlated",
            "window/pair metrics are not independent clinical event accuracy",
            "the independent pilot contains only six events",
            "all deployed outputs remain telemetry-only shadow authority",
        ],
        "artifacts": [
            "split_metrics.png", "gru_learning_curve.png", "swin3d_roc_pr.png", "gru_roc_pr.png",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = "\n".join([
        "# Current fall-model performance (engineering diagnostic)",
        "",
        markdown_table(results),
        "",
        "## Event checks",
        "",
        f"- Correlated staged test: ACC {report['event_test']['accuracy']:.4f}, "
        f"precision {report['event_test']['precision']:.4f}, recall {report['event_test']['recall']:.4f} "
        f"over {report['event_test']['events']} recording events.",
        f"- Independent pilot: ACC 0.8333, precision 0.7500, recall 1.0000 over 6 events.",
        "",
        "## Interpretation",
        "",
        "The Swin3D verifier is the stronger current candidate. The GRU learning curve shows immediate "
        "overfitting: validation loss is best at epoch 1 while training loss continues downward. These "
        "numbers are engineering diagnostics, not a clinical or promotion claim.",
    ])
    (out_dir / "REPORT.md").write_text(markdown + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "models": results,
                      "event_test": report["event_test"],
                      "independent_pilot": report["independent_pilot"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
