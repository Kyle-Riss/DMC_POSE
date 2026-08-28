#!/usr/bin/env python3
"""Build non-promotable GRU windows from reviewed staged recordings.

The staged corpus has synchronized multiview recordings but no authoritative
subject identity. Camera views of a recording stay locked to one split, and
the resulting corpus is permanently marked shadow-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_temporal_windows import windows_from_video
from temporal_features import FEATURE_SCHEMA_VERSION, temporal_feature_names
from temporal_sequence import cadence_interval_bounds, observed_sequence_contract

POSITIVE_STATUSES = {"complete", "needs_adjudication"}
HARD_NEGATIVE_MARKER = "hard negative"
SPLITS = ("train", "val", "test")
SPLIT_SEED = 20260824
SOURCE_SAMPLE_HZ = 20.0


def resample_observed_frame(frame: pd.DataFrame, target_hz: float) -> pd.DataFrame:
    """Deterministically decimate the observed 20 Hz source without interpolation."""
    ratio = SOURCE_SAMPLE_HZ / float(target_hz)
    stride = int(round(ratio))
    if target_hz <= 0 or abs(ratio - stride) > 1e-6:
        raise ValueError("target sample rate must evenly divide the 20 Hz source")
    if stride == 1:
        return frame.copy()
    group_columns = [name for name in ("sequence_id", "track_id") if name in frame.columns]
    if not group_columns:
        return frame.iloc[::stride].reset_index(drop=True)
    pieces = [
        group.iloc[::stride]
        for _, group in frame.groupby(group_columns, sort=False, dropna=False)
    ]
    return pd.concat(pieces).sort_values("timestamp_sec").reset_index(drop=True)


def recording_splits(recording_labels: dict[str, int]) -> dict[str, str]:
    """Deterministically stratify recordings; camera views remain locked."""
    by_label: dict[int, list[str]] = defaultdict(list)
    for recording, label in recording_labels.items():
        by_label[int(label)].append(recording)
    if any(len(by_label[label]) < 3 for label in (0, 1)):
        raise ValueError("at least three positive and three hard-negative recordings are required")
    result: dict[str, str] = {}
    for label in (0, 1):
        recordings = sorted(by_label[label])
        random.Random(SPLIT_SEED + label).shuffle(recordings)
        val_count = max(1, round(len(recordings) * 0.2))
        test_count = max(1, round(len(recordings) * 0.2))
        train_count = len(recordings) - val_count - test_count
        if train_count < 1:
            raise ValueError(f"not enough label={label} recordings for train/val/test")
        for index, recording in enumerate(recordings):
            result[recording] = (
                "train" if index < train_count
                else "val" if index < train_count + val_count
                else "test"
            )
    return result


def reviewed_recordings(rows: list[dict[str, str]]) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["recording_id"]].append(row)
    labels: dict[str, int] = {}
    for recording, views in grouped.items():
        statuses = {row["annotation_status"] for row in views}
        if len(statuses) != 1:
            raise ValueError(f"multiview status disagreement: {recording}: {sorted(statuses)}")
        status = next(iter(statuses))
        if status in POSITIVE_STATUSES:
            for row in views:
                onset = str(row.get("fall_onset_frame") or "").strip()
                end = str(row.get("fall_end_frame") or "").strip()
                if not onset or not end or int(onset) > int(end):
                    raise ValueError(f"reviewed positive lacks onset/end: {row['video_id']}")
                if status == "complete" and not str(row.get("impact_frame") or "").strip():
                    raise ValueError(f"complete positive lacks impact: {row['video_id']}")
            labels[recording] = 1
        elif status == "excluded" and all(
            HARD_NEGATIVE_MARKER in str(row.get("notes") or "").lower() for row in views
        ):
            labels[recording] = 0
    return {recording: grouped[recording] for recording in labels}, labels


def apply_reviewed_target(frame: pd.DataFrame, annotation: dict[str, str], label: int, split: str) -> pd.DataFrame:
    output = frame.copy()
    output["split"] = split
    output["subject_id"] = "unknown_staged_subject"
    if label == 0:
        output["target"] = "non_fall"
        output["active_labels"] = ""
        return output
    onset = int(annotation["fall_onset_frame"])
    end = int(annotation["fall_end_frame"])
    active = output["frame_idx"].astype(int).between(onset, end)
    output["target"] = np.where(active, "fall", "non_fall")
    source = (
        "fall_reviewed_no_discrete_impact"
        if annotation["annotation_status"] == "needs_adjudication"
        else "fall_reviewed"
    )
    output["active_labels"] = np.where(active, source, "")
    return output


def build(features_dir: Path, annotations_path: Path, out_dir: Path, *, sample_hz: float = 20.0) -> dict:
    feature_index = json.loads((features_dir / "features_index.json").read_text(encoding="utf-8"))
    source_contract = observed_sequence_contract(SOURCE_SAMPLE_HZ)
    if feature_index.get("sequence_contract_version") != source_contract:
        raise ValueError("source features are not observed_only_20hz_v1")
    contract = observed_sequence_contract(sample_hz)
    window_rows = int(round(4.0 * sample_hz))
    stride_rows = max(1, int(round(0.25 * sample_hz)))
    with annotations_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected, labels = reviewed_recordings(rows)
    splits = recording_splits(labels)
    feature_paths = {
        row["video_id"]: Path(row["out"])
        for row in feature_index.get("results", []) if row.get("status") in {"ok", "skipped"}
    }
    by_split = {split: {"x": [], "y": [], "meta": []} for split in SPLITS}
    min_interval, max_interval = cadence_interval_bounds(sample_hz)
    excluded = []
    for recording in sorted(selected):
        for annotation in sorted(selected[recording], key=lambda row: row["video_id"]):
            video_id = annotation["video_id"]
            path = feature_paths.get(video_id)
            if path is None or not path.is_file() or path.stat().st_size == 0:
                excluded.append({"video_id": video_id, "reason": "missing_or_empty_feature_csv"})
                continue
            split = splits[recording]
            frame = apply_reviewed_target(pd.read_csv(path), annotation, labels[recording], split)
            frame = resample_observed_frame(frame, sample_hz)
            windows, targets, metadata, names = windows_from_video(
                frame, window_rows, stride_rows, min_interval_sec=min_interval, max_interval_sec=max_interval,
            )
            if names != temporal_feature_names():
                raise ValueError("feature order mismatch")
            for row in metadata:
                row.update({
                    "recording_id": recording,
                    "camera_id": annotation["camera_id"],
                    "label_source": "manual_multiview_review",
                    "annotation_status": annotation["annotation_status"],
                    "impact_observed": bool(str(annotation.get("impact_frame") or "").strip()),
                    "subject_identity_known": False,
                    "promotion_eligible": False,
                })
            by_split[split]["x"].extend(windows)
            by_split[split]["y"].extend(targets)
            by_split[split]["meta"].extend(metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "window_schema_version": "pose_gru_reviewed_staged_shadow_v2_resampled",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sequence_contract_version": contract,
        "data_provenance": "manual_multiview_reviewed_hospital_staged",
        "synthetic_smoke_fixture": False,
        "promotion_eligible": False,
        "accuracy_claim": False,
        "subject_identity_known": False,
        "evaluation_eligible": False,
        "usage_contract": "central_shadow_only",
        "split_policy": "recording-disjoint stratified diagnostic split; subject leakage may exist",
        "split_seed": SPLIT_SEED,
        "impact_policy": "complete uses observed impact; needs_adjudication uses onset-to-end binary target and is excluded from impact metrics",
        "negative_policy": "excluded recordings enter only when every view is explicitly marked hard negative",
        "window_sec": 4.0,
        "stride_sec": 0.25,
        "sample_hz": sample_hz,
        "window_rows": window_rows,
        "stride_rows": stride_rows,
        "feature_count": 109,
        "feature_names": temporal_feature_names(),
        "recording_splits": dict(sorted(splits.items())),
        "recording_label_counts": {str(k): v for k, v in sorted(Counter(labels.values()).items())},
        "excluded": excluded,
        "splits": {},
        "warnings": [
            "subject identity is unknown and a person may occur across splits",
            "multiview windows are correlated and must not be treated as independent clinical events",
            "validation/test metrics are engineering diagnostics only",
            "checkpoint must remain shadow-only and cannot authorize alerts",
        ],
    }
    for split, values in by_split.items():
        x = np.asarray(values["x"], dtype=np.float32)
        y = np.asarray(values["y"], dtype=np.int64)
        if not len(x):
            x = np.empty((0, window_rows, 109), dtype=np.float32)
        np.savez_compressed(out_dir / f"{split}.npz", x=x, y=y)
        (out_dir / f"{split}_metadata.json").write_text(
            json.dumps(values["meta"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        counts = Counter(y.tolist())
        report["splits"][split] = {
            "windows": len(y), "non_fall": counts.get(0, 0), "fall": counts.get(1, 0),
            "videos": len({row["video_id"] for row in values["meta"]}),
            "recordings": len({row["recording_id"] for row in values["meta"]}),
        }
    (out_dir / "window_index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=PROJECT_ROOT / "external_datasets/features/pose_109_observed_only_20hz/usb_sim_falldown_full")
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "external_datasets/windows/pose_gru_109_observed_only_20hz/usb_reviewed_staged_shadow_v1_4s")
    parser.add_argument("--sample-hz", type=float, default=20.0)
    args = parser.parse_args()
    report = build(args.features_dir.resolve(), args.annotations.resolve(), args.out_dir.resolve(), sample_hz=args.sample_hz)
    print(json.dumps({key: report[key] for key in ("promotion_eligible", "recording_label_counts", "splits", "excluded")}, ensure_ascii=False, indent=2))
    print(f"window_index: {(args.out_dir / 'window_index.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
