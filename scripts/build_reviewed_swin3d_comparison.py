#!/usr/bin/env python3
"""Build a compact comparison chart for reviewed AI_runner Swin3D experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project / "runs/performance/current_20260828/reviewed_swin3d_comparison",
    )
    args = parser.parse_args()
    sources = {
        "Delta (old)": (
            project / "runs/performance/current_20260828/ai_runner_reviewed_combined_71.json",
            "all_reviewed",
        ),
        "Single-clip (new)": (
            project
            / "runs/video_verifier/swin3d_b_reviewed_single_group_cv_dedup_v1_20260828/report.json",
            "held_session_cv",
        ),
        "Hybrid (diagnostic)": (
            project
            / "runs/video_verifier/swin3d_b_reviewed_hybrid_group_cv_v1_20260828/report.json",
            "held_session_cv",
        ),
    }
    rows = []
    for name, (path, key) in sources.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        metric = document[key]
        rows.append({"name": name, "source": str(path.resolve()), **metric})

    labels = [row["name"] for row in rows]
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    x = range(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for index, metric_name in enumerate(metric_names):
        offset = (index - 2) * width
        values = [row[metric_name] for row in rows]
        bars = ax.bar(
            [value + offset for value in x],
            values,
            width,
            label=metric_name,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Reviewed AI_runner events — held-session Swin3D comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "model_comparison.png", dpi=170)
    plt.close(fig)

    result = {
        "schema_version": "dmc_reviewed_swin3d_comparison_v1",
        "rows": rows,
        "selected": "Single-clip (new)",
        "selection_reason": (
            "best held-session F1 and ROC-AUC while retaining >=0.8 fall recall"
        ),
        "promotion_eligible": False,
        "next_gate": "new untouched positive recording session",
    }
    (args.out_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"chart: {(args.out_dir / 'model_comparison.png').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
