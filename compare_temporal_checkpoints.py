#!/usr/bin/env python3
"""Compare legacy and v2 checkpoints on identical observed-only windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_temporal_events import evaluate_split, predict_split
from train_tcn import report_metrics


def event_f1(report: dict) -> float:
    precision = float(report["event_precision"])
    recall = float(report["end_to_end_event_recall"])
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def concise(report: dict) -> dict:
    return {key: value for key, value in report.items() if key != "per_video"}


def choose_event_operating_point(windows_dir: Path, manifest: dict, checkpoint: Path, metadata: list[dict], probability: np.ndarray, device: torch.device, merge_gap_sec: float, report_threshold: float | None) -> tuple[dict, list[dict]]:
    thresholds = {round(float(value), 6) for value in np.arange(0.05, 0.951, 0.01)}
    if report_threshold is not None:
        thresholds.add(round(float(report_threshold), 6))
    candidates = []
    for persistence in (1, 2, 3):
        for threshold in sorted(thresholds):
            result = evaluate_split(
                "val", windows_dir, manifest, checkpoint, threshold,
                persistence, device, merge_gap_sec,
                precomputed_metadata=metadata,
                precomputed_probability=probability,
            )
            summary = concise(result)
            summary["event_f1"] = round(event_f1(result), 4)
            candidates.append(summary)
    selected = max(
        candidates,
        key=lambda row: (
            row["event_f1"],
            row["end_to_end_event_recall"],
            -row["false_events"],
            row["event_precision"],
            -row["persistence_windows"],
        ),
    )
    return selected, candidates


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, default=root / "external_datasets/windows/tcn_109_v2_no_missing/gmdcsa24_3s")
    parser.add_argument("--manifest", type=Path, default=root / "external_datasets/manifests/gmdcsa24.json")
    parser.add_argument("--legacy-dir", type=Path, default=root / "runs/temporal_tcn/gmdcsa24_tcn")
    parser.add_argument("--v2-dir", type=Path, default=root / "runs/temporal_tcn/gmdcsa24_tcn_v2_observed_only")
    parser.add_argument("--legacy-name", default="legacy_checkpoint")
    parser.add_argument("--v2-name", default="v2_observed_only_checkpoint")
    parser.add_argument("--out", type=Path, default=root / "runs/temporal_tcn/observed_only_checkpoint_comparison.json")
    parser.add_argument("--merge-gap-sec", type=float, default=3.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    split_targets = {
        split: np.load(args.windows_dir / f"{split}.npz")["y"].astype(np.int64)
        for split in ("val", "test")
    }
    if args.legacy_name == args.v2_name:
        raise ValueError("checkpoint names must differ")
    models = {args.legacy_name: args.legacy_dir, args.v2_name: args.v2_dir}
    results = {}
    for name, model_dir in models.items():
        checkpoint_path = model_dir / "model.pt"
        training_report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        probabilities_by_split = {}
        metadata_by_split = {}
        for split in ("val", "test"):
            metadata_by_split[split], probabilities_by_split[split] = predict_split(args.windows_dir, split, checkpoint_path, device)

        selected, candidates = choose_event_operating_point(
            args.windows_dir, manifest, checkpoint_path,
            metadata_by_split["val"], probabilities_by_split["val"],
            device, args.merge_gap_sec,
            float(training_report["validation"]["threshold"]),
        )
        threshold = float(selected["threshold"])
        persistence = int(selected["persistence_windows"])
        test_event = evaluate_split(
            "test", args.windows_dir, manifest, checkpoint_path,
            threshold, persistence, device, args.merge_gap_sec,
            precomputed_metadata=metadata_by_split["test"],
            precomputed_probability=probabilities_by_split["test"],
        )
        results[name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "training_input_contract": checkpoint_payload.get("sequence_contract_version", "historical_unknown"),
            "normalization": "checkpoint-owned mean/std",
            "validation_selection_policy": "max event F1; ties: recall, fewer false events, precision, lower persistence",
            "validation_selected": selected,
            "test_event": {**concise(test_event), "event_f1": round(event_f1(test_event), 4)},
            "validation_window_at_selected_threshold": report_metrics(split_targets["val"], probabilities_by_split["val"], threshold),
            "test_window_at_selected_threshold": report_metrics(split_targets["test"], probabilities_by_split["test"], threshold),
            "search_candidate_count": len(candidates),
        }

    legacy_historical = args.legacy_dir / "event_report.json"
    payload = {
        "comparison_version": "phase10_checkpoint_compatibility_v1",
        "comparison_scope": "checkpoint compatibility on identical observed-only Phase 10 raw windows",
        "windows_dir": str(args.windows_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "threshold_search": "0.05..0.95 step 0.01 plus original report threshold",
        "persistence_search": [1, 2, 3],
        "test_policy": "one evaluation after validation operating-point selection",
        "results": results,
        "historical_preprocessing_reference": str(legacy_historical.resolve()) if legacy_historical.is_file() else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"validation_selected": value["validation_selected"], "test_event": value["test_event"]} for name, value in results.items()}, ensure_ascii=False, indent=2))
    print(f"comparison: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
