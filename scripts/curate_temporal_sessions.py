#!/usr/bin/env python3
"""Curate reviewed automatic sessions into cadence-safe 30x109 windows."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def contiguous_slices(
    timestamps: np.ndarray,
    track_ids: np.ndarray,
    *,
    min_dt: float = 0.070,
    max_dt: float = 0.250,
) -> list[slice]:
    if len(timestamps) == 0:
        return []
    boundaries = [0]
    for index in range(1, len(timestamps)):
        dt = float(timestamps[index] - timestamps[index - 1])
        if (
            int(track_ids[index]) != int(track_ids[index - 1])
            or dt < min_dt
            or dt > max_dt
        ):
            boundaries.append(index)
    boundaries.append(len(timestamps))
    return [
        slice(start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]


def curate_session(
    session_dir: Path,
    *,
    window_size: int = 30,
    stride: int = 5,
    min_dt: float = 0.070,
    max_dt: float = 0.250,
) -> tuple[list[dict], dict]:
    manifest = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("review_status") != "reviewed":
        return [], {"session_id": manifest.get("session_id"), "excluded": "not_reviewed"}
    binary_label = manifest.get("binary_fall_label")
    if binary_label not in (0, 1):
        return [], {"session_id": manifest.get("session_id"), "excluded": "missing_binary_label"}
    if int(binary_label) == 1 and any(
        manifest.get(key) is None
        for key in ("fall_onset_sec", "impact_sec", "fall_end_sec")
    ):
        return [], {"session_id": manifest.get("session_id"), "excluded": "missing_fall_boundaries"}

    with np.load(session_dir / "features.npz") as arrays:
        features = np.asarray(arrays["features"], dtype=np.float32)
        timestamps = np.asarray(arrays["relative_timestamps_sec"], dtype=np.float64)
        track_ids = np.asarray(arrays["track_ids"], dtype=np.int64)
        quality = np.asarray(arrays["pose_quality"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 109:
        return [], {"session_id": manifest.get("session_id"), "excluded": "invalid_feature_shape"}
    if not np.all(np.isfinite(features)):
        return [], {"session_id": manifest.get("session_id"), "excluded": "non_finite_features"}
    if len({len(features), len(timestamps), len(track_ids), len(quality)}) != 1:
        return [], {"session_id": manifest.get("session_id"), "excluded": "array_length_mismatch"}

    slices = contiguous_slices(
        timestamps, track_ids, min_dt=min_dt, max_dt=max_dt
    )
    windows: list[dict] = []
    positive = int(binary_label) == 1
    onset = float(manifest.get("fall_onset_sec", 0.0)) if positive else None
    end = float(manifest.get("fall_end_sec", 0.0)) if positive else None
    for segment_index, segment in enumerate(slices):
        segment_length = segment.stop - segment.start
        if segment_length < window_size:
            continue
        for local_start in range(0, segment_length - window_size + 1, stride):
            start = segment.start + local_start
            stop = start + window_size
            window_end = float(timestamps[stop - 1])
            if positive:
                if window_end > end:
                    continue
                label = int(window_end >= onset)
            else:
                label = 0
            windows.append({
                "x": features[start:stop].copy(),
                "y": label,
                "session_id": str(manifest["session_id"]),
                "track_id": int(track_ids[start]),
                "segment_index": segment_index,
                "start_sec": float(timestamps[start]),
                "end_sec": window_end,
                "mean_quality": float(np.mean(quality[start:stop])),
            })
    return windows, {
        "session_id": manifest.get("session_id"),
        "label": manifest.get("label"),
        "binary_fall_label": int(binary_label),
        "sample_count": int(len(features)),
        "segment_lengths": [item.stop - item.start for item in slices],
        "window_count": len(windows),
        "quality_mean": float(np.mean(quality)) if len(quality) else 0.0,
        "excluded": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("runtime_data/temporal_sessions"))
    parser.add_argument("--output-dir", type=Path, default=Path("runtime_data/curated_temporal_sessions/v1"))
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min-dt", type=float, default=0.070)
    parser.add_argument("--max-dt", type=float, default=0.250)
    args = parser.parse_args()

    all_windows: list[dict] = []
    sessions: list[dict] = []
    for manifest_path in sorted(args.input_root.glob("*/manifest.json")):
        windows, summary = curate_session(
            manifest_path.parent,
            window_size=args.window_size,
            stride=args.stride,
            min_dt=args.min_dt,
            max_dt=args.max_dt,
        )
        all_windows.extend(windows)
        sessions.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if all_windows:
        x = np.stack([item["x"] for item in all_windows]).astype(np.float32)
    else:
        x = np.empty((0, args.window_size, 109), dtype=np.float32)
    y = np.asarray([item["y"] for item in all_windows], dtype=np.int64)
    np.savez_compressed(
        args.output_dir / "reviewed_windows.npz",
        x=x,
        y=y,
        session_ids=np.asarray([item["session_id"] for item in all_windows]),
        track_ids=np.asarray([item["track_id"] for item in all_windows], dtype=np.int64),
        segment_indices=np.asarray([item["segment_index"] for item in all_windows], dtype=np.int64),
        start_sec=np.asarray([item["start_sec"] for item in all_windows], dtype=np.float64),
        end_sec=np.asarray([item["end_sec"] for item in all_windows], dtype=np.float64),
        mean_quality=np.asarray([item["mean_quality"] for item in all_windows], dtype=np.float32),
    )
    class_counts = Counter(int(value) for value in y)
    report = {
        "schema_version": "curated_temporal_sessions_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_root": str(args.input_root.resolve()),
        "output_npz": str((args.output_dir / "reviewed_windows.npz").resolve()),
        "feature_shape": list(x.shape),
        "window_size": args.window_size,
        "stride": args.stride,
        "cadence_sec": {"min": args.min_dt, "max": args.max_dt},
        "window_count": int(len(x)),
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "group_key": "session_id",
        "split_assignment_ready": False,
        "training_ready": False,
        "training_blockers": [
            "insufficient_class_coverage" if len(class_counts) < 2 else None,
            "group_split_assignment_pending",
        ],
        "sessions": sessions,
    }
    report["training_blockers"] = [item for item in report["training_blockers"] if item]
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
