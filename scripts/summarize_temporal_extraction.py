#!/usr/bin/env python3
"""Summarize observed-only extraction quality without claiming model accuracy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


COUNTERS = (
    "decoded_frames",
    "pose_probes",
    "rows",
    "no_primary",
    "duplicate_skip",
    "non_monotonic_skip",
    "gap_reset",
    "track_reset",
)


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return round(float(numerator) / float(denominator), 6) if denominator else None


def summarize(features_dir: Path, manifest: dict) -> dict:
    index = json.loads((features_dir / "features_index.json").read_text(encoding="utf-8"))
    manifest_by_id = {item["video_id"]: item for item in manifest.get("items", [])}
    totals = {name: 0 for name in COUNTERS}
    by_camera: dict[str, dict[str, int]] = defaultdict(lambda: {name: 0 for name in COUNTERS})
    visible_joint_sum = 0.0
    visible_row_count = 0
    csv_count = 0
    empty_csv_count = 0

    for result in index.get("results", []):
        if result.get("status") != "ok":
            continue
        video_id = str(result["video_id"])
        camera = str(manifest_by_id.get(video_id, {}).get("camera_id") or "unknown")
        for name in COUNTERS:
            value = int(result.get(name, 0))
            totals[name] += value
            by_camera[camera][name] += value
        csv_path = Path(result["out"])
        if not csv_path.is_file():
            continue
        try:
            frame = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            empty_csv_count += 1
            csv_count += 1
            continue
        if frame.empty:
            empty_csv_count += 1
            csv_count += 1
            continue
        visible_columns = [f"kpt_{index}_visible" for index in range(17)]
        if not frame.empty and all(column in frame for column in visible_columns):
            visible_joint_sum += float(frame[visible_columns].sum(axis=1).sum())
            visible_row_count += len(frame)
        csv_count += 1

    duration_total_sec = sum(float(item.get("duration_sec") or 0.0) for item in manifest_by_id.values())

    def decorate(values: dict[str, int]) -> dict:
        return {
            **values,
            "pose_observation_coverage": _rate(values["rows"], values["pose_probes"]),
            "no_primary_rate": _rate(values["no_primary"], values["pose_probes"]),
            "gap_resets_per_1000_probes": round(1000.0 * values["gap_reset"] / values["pose_probes"], 3) if values["pose_probes"] else None,
        }

    elapsed_sec = float(index.get("elapsed_sec") or 0.0)
    return {
        "report": "observed_pose_extraction_quality_v1",
        "accuracy_claim": False,
        "feature_schema_version": index.get("feature_schema_version"),
        "sequence_contract_version": index.get("sequence_contract_version"),
        "sample_hz": index.get("sample_hz"),
        "video_count": len(index.get("results", [])),
        "successful_videos": sum(result.get("status") == "ok" for result in index.get("results", [])),
        "csv_count": csv_count,
        "empty_csv_count": empty_csv_count,
        "source_duration_sec": round(duration_total_sec, 3),
        "elapsed_sec": elapsed_sec,
        "offline_realtime_factor": round(duration_total_sec / elapsed_sec, 3) if elapsed_sec else None,
        "mean_visible_joints_per_observation": round(visible_joint_sum / visible_row_count, 4) if visible_row_count else None,
        "totals": decorate(totals),
        "by_camera": {camera: decorate(values) for camera, values in sorted(by_camera.items())},
        "limitations": list(manifest.get("warnings", [])) + [
            "metrics describe Pose observation quality and compute throughput, not fall accuracy",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = summarize(args.features_dir.resolve(), manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
