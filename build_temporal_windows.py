#!/usr/bin/env python3
"""Build causal TCN windows without crossing pose gaps or track changes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_features import FEATURE_SCHEMA_VERSION, temporal_feature_names

WINDOW_SCHEMA_VERSION = "pose_tcn_window_v2_observed_only"
MIN_INTERVAL_SEC = 0.070
MAX_INTERVAL_SEC = 0.150


def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = temporal_feature_names()
    base_names = names[:75]
    missing = [name for name in base_names if name not in df.columns]
    if "timestamp_sec" not in df.columns:
        missing.append("timestamp_sec")
    if missing:
        raise ValueError(f"missing feature columns: {missing[:5]}")

    norm_cols = names[:34]
    visible_cols = names[51:68]
    xy = df[norm_cols].fillna(0.0).to_numpy(dtype=np.float32)
    base = df[base_names].fillna(0.0).to_numpy(dtype=np.float32)
    visible = df[visible_cols].fillna(0.0).to_numpy(dtype=np.float32)
    timestamps = df["timestamp_sec"].to_numpy(dtype=np.float64)

    velocity = np.zeros_like(xy)
    if len(df) > 1:
        dt = np.diff(timestamps)
        valid_dt = dt > 1e-6
        raw_velocity = np.zeros((len(df) - 1, xy.shape[1]), dtype=np.float32)
        raw_velocity[valid_dt] = np.diff(xy, axis=0)[valid_dt] / dt[valid_dt, None]
        coordinate_visible = np.repeat(visible[1:] * visible[:-1], 2, axis=1)
        velocity[1:] = raw_velocity * coordinate_visible

    matrix = np.concatenate([base, velocity], axis=1).astype(np.float32)
    if matrix.shape[1] != len(names):
        raise AssertionError(f"unexpected feature shape: {matrix.shape}")
    return matrix, names


def _contract_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    observed = df[df["person_detected"].astype(bool)].copy()
    if observed.empty:
        return []
    observed = observed.sort_values("timestamp_sec").reset_index(drop=True)
    explicit_sequence = observed["sequence_id"] if "sequence_id" in observed.columns else pd.Series(1, index=observed.index)
    track = observed["track_id"] if "track_id" in observed.columns else pd.Series(1, index=observed.index)
    timestamps = observed["timestamp_sec"].to_numpy(dtype=np.float64)
    boundary = np.ones(len(observed), dtype=bool)
    if len(observed) > 1:
        dt = np.diff(timestamps)
        boundary[1:] = (
            (explicit_sequence.iloc[1:].to_numpy() != explicit_sequence.iloc[:-1].to_numpy())
            | (track.iloc[1:].to_numpy() != track.iloc[:-1].to_numpy())
            | (dt < MIN_INTERVAL_SEC - 1e-9)
            | (dt > MAX_INTERVAL_SEC + 1e-9)
            | (dt <= 0.0)
        )
    observed["_contract_segment"] = np.cumsum(boundary)
    return [group.reset_index(drop=True) for _, group in observed.groupby("_contract_segment", sort=False)]


def windows_from_video(df: pd.DataFrame, window_rows: int, stride_rows: int) -> tuple[list[np.ndarray], list[int], list[dict], list[str]]:
    names = temporal_feature_names()
    windows: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict] = []
    for segment in _contract_segments(df):
        if len(segment) < window_rows:
            continue
        features, segment_names = feature_matrix(segment)
        if segment_names != names:
            raise AssertionError("canonical feature order mismatch")
        raw_targets = segment["target"].astype(str).to_numpy()
        labels = (raw_targets == "fall").astype(np.int64)
        sequence_ready_sec = float(segment.iloc[window_rows - 1]["timestamp_sec"])
        sequence_observation_start_sec = float(segment.iloc[0]["timestamp_sec"])
        sequence_observation_end_sec = float(segment.iloc[-1]["timestamp_sec"])
        for end in range(window_rows - 1, len(segment), stride_rows):
            start = end - window_rows + 1
            # An ignored endpoint must never become an implicit non-fall
            # target. Ignore rows may remain in causal context.
            if raw_targets[end] == "ignore":
                continue
            windows.append(features[start : end + 1])
            targets.append(int(labels[end]))
            end_row = segment.iloc[end]
            start_row = segment.iloc[start]
            metadata.append({
                "video_id": str(end_row["video_id"]),
                "subject_id": None if pd.isna(end_row.get("subject_id")) else str(end_row.get("subject_id")),
                "split": str(end_row.get("split")),
                "track_id": int(end_row.get("track_id", 1)),
                "sequence_id": int(end_row.get("sequence_id", 1)),
                "sequence_ready_sec": sequence_ready_sec,
                "sequence_observation_start_sec": sequence_observation_start_sec,
                "sequence_observation_end_sec": sequence_observation_end_sec,
                "start_sec": float(start_row["timestamp_sec"]),
                "end_sec": float(end_row["timestamp_sec"]),
                "fall_fraction": round(float(labels[start : end + 1].mean()), 4),
                "ignored_fraction": round(float((raw_targets[start : end + 1] == "ignore").mean()), 4),
                "person_fraction": 1.0,
            })
    return windows, targets, metadata, names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--features-dir", type=Path, default=project_root / "external_datasets/features/tcn_109_v2_no_missing/gmdcsa24")
    parser.add_argument("--out-dir", type=Path, default=project_root / "external_datasets/windows/tcn_109_v2_no_missing/gmdcsa24_3s")
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--stride-sec", type=float, default=0.5)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    args = parser.parse_args()

    window_rows = int(round(args.window_sec * args.sample_hz))
    stride_rows = int(round(args.stride_sec * args.sample_hz))
    if window_rows <= 0 or stride_rows <= 0:
        raise ValueError("window and stride must be positive")

    all_by_split: dict[str, dict[str, list]] = {split: {"x": [], "y": [], "meta": []} for split in ("train", "val", "test")}
    feature_names = temporal_feature_names()
    csv_files = sorted(args.features_dir.glob("*/*.csv"))
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        windows, targets, metadata, names = windows_from_video(df, window_rows, stride_rows)
        if feature_names != names:
            raise ValueError(f"feature order mismatch: {csv_path}")
        split = str(df["split"].iloc[0])
        if split not in all_by_split:
            raise ValueError(f"unsupported split {split}: {csv_path}")
        all_by_split[split]["x"].extend(windows)
        all_by_split[split]["y"].extend(targets)
        all_by_split[split]["meta"].extend(metadata)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "window_schema_version": WINDOW_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sequence_contract_version": "observed_only_10hz_v2",
        "source_dir": str(args.features_dir.resolve()),
        "csv_count": len(csv_files),
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "sample_hz": args.sample_hz,
        "window_rows": window_rows,
        "stride_rows": stride_rows,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "splits": {},
    }
    for split, values in all_by_split.items():
        x = np.asarray(values["x"], dtype=np.float32)
        y = np.asarray(values["y"], dtype=np.int64)
        if len(x) == 0:
            x = np.empty((0, window_rows, len(feature_names)), dtype=np.float32)
        np.savez_compressed(args.out_dir / f"{split}.npz", x=x, y=y)
        (args.out_dir / f"{split}_metadata.json").write_text(json.dumps(values["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
        counts = Counter(y.tolist())
        summary["splits"][split] = {"windows": len(y), "non_fall": counts.get(0, 0), "fall": counts.get(1, 0), "videos": len({row["video_id"] for row in values["meta"]})}

    (args.out_dir / "window_index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
    index_path = (args.out_dir / "window_index.json").resolve()
    print(f"window_index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
