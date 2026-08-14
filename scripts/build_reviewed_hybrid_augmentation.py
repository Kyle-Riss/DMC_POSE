#!/usr/bin/env python3
"""Build train-only reviewed augmentation and an archive-disjoint diagnostic set.

Only completed manual FallVision fall intervals and explicitly reviewed local
non-fall sessions are accepted.  FallVision participant identity is unresolved,
so every output is marked diagnostic/non-promotable even though archive groups
are kept disjoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


DEFAULT_HOLDOUT_FALL_GROUPS = {
    "fallvision_archive:fall:bed:4",
    "fallvision_archive:fall:chair:1",
    "fallvision_archive:fall:stand:3",
}
SPLITS = ("train", "val", "test")


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def empty_x(window_rows: int, feature_count: int) -> np.ndarray:
    return np.empty((0, window_rows, feature_count), dtype=np.float32)


def validate_contract(index: dict) -> tuple[int, int]:
    if index.get("sequence_contract_version") != "observed_only_10hz_v2":
        raise ValueError("FallVision windows are not observed_only_10hz_v2")
    window_rows = int(index.get("window_rows", 0))
    feature_count = int(index.get("feature_count", 0))
    if (window_rows, feature_count) != (30, 109):
        raise ValueError(f"expected 30x109 windows, got {window_rows}x{feature_count}")
    return window_rows, feature_count


def load_fallvision(
    windows_dir: Path, manifest_path: Path, holdout_groups: set[str]
) -> tuple[list[dict], list[dict], dict]:
    index = read_json(windows_dir / "window_index.json")
    validate_contract(index)
    with np.load(windows_dir / "test.npz") as arrays:
        x = np.asarray(arrays["x"], dtype=np.float32)
        y = np.asarray(arrays["y"], dtype=np.int64)
    metadata = read_json(windows_dir / "test_metadata.json")
    manifest = read_json(manifest_path)
    if len(x) != len(y) or len(y) != len(metadata):
        raise ValueError("FallVision window/label/metadata length mismatch")
    by_id = {item["video_id"]: item for item in manifest["items"]}

    train: list[dict] = []
    diagnostic: list[dict] = []
    for features, label, row in zip(x, y, metadata):
        item = by_id.get(row["video_id"])
        if item is None:
            raise ValueError(f"missing manifest item: {row['video_id']}")
        group = item.get("split_group")
        if not group:
            raise ValueError(f"missing split_group: {row['video_id']}")
        record = {
            "x": features,
            "y": int(label),
            "metadata": {
                **row,
                "label": int(label),
                "dataset": "fallvision",
                "source_kind": "manual_reviewed_fall" if int(label) == 1 else "official_non_fall",
                "split_group": group,
                "promotion_metric_eligible": False,
            },
        }
        if int(label) == 1:
            if item.get("annotation_scope") != "manual_interval_diagnostic_only":
                raise ValueError(f"positive is not manually reviewed: {row['video_id']}")
            if item.get("annotation_source") not in {"manual_pilot", "manual"}:
                raise ValueError(f"unexpected positive annotation source: {row['video_id']}")
            (diagnostic if group in holdout_groups else train).append(record)
        else:
            diagnostic.append(record)

    train_groups = {row["metadata"]["split_group"] for row in train}
    diagnostic_groups = {row["metadata"]["split_group"] for row in diagnostic}
    overlap = train_groups & diagnostic_groups
    if overlap:
        raise ValueError(f"archive group leakage: {sorted(overlap)}")
    missing_holdout = holdout_groups - diagnostic_groups
    if missing_holdout:
        raise ValueError(f"holdout groups have no usable windows: {sorted(missing_holdout)}")
    return train, diagnostic, manifest


def load_local_negatives(curated_npz: Path, curated_report: Path) -> list[dict]:
    report = read_json(curated_report)
    if report.get("schema_version") != "curated_temporal_sessions_v1":
        raise ValueError("unexpected local curation schema")
    with np.load(curated_npz) as arrays:
        x = np.asarray(arrays["x"], dtype=np.float32)
        y = np.asarray(arrays["y"], dtype=np.int64)
        session_ids = np.asarray(arrays["session_ids"]).astype(str)
        track_ids = np.asarray(arrays["track_ids"], dtype=np.int64)
        segment_indices = np.asarray(arrays["segment_indices"], dtype=np.int64)
        start_sec = np.asarray(arrays["start_sec"], dtype=np.float64)
        end_sec = np.asarray(arrays["end_sec"], dtype=np.float64)
        quality = np.asarray(arrays["mean_quality"], dtype=np.float32)
    lengths = {len(value) for value in (x, y, session_ids, track_ids, segment_indices, start_sec, end_sec, quality)}
    if len(lengths) != 1:
        raise ValueError("local curated array length mismatch")
    if x.ndim != 3 or x.shape[1:] != (30, 109):
        raise ValueError(f"local windows must be Nx30x109, got {x.shape}")
    if np.any(y != 0):
        raise ValueError("local augmentation currently accepts reviewed non-fall only")
    records = []
    for index in range(len(y)):
        session = str(session_ids[index])
        records.append({
            "x": x[index],
            "y": 0,
            "metadata": {
                "video_id": f"local_session:{session}",
                "subject_id": None,
                "split": "train",
                "track_id": int(track_ids[index]),
                "sequence_id": int(segment_indices[index]),
                "start_sec": float(start_sec[index]),
                "end_sec": float(end_sec[index]),
                "fall_fraction": 0.0,
                "ignored_fraction": 0.0,
                "person_fraction": 1.0,
                "label": 0,
                "dataset": "dmc_local",
                "source_kind": "operator_reviewed_hard_negative",
                "split_group": f"local_session:{session}",
                "mean_quality": float(quality[index]),
                "promotion_metric_eligible": False,
            },
        })
    return records


def write_window_set(root: Path, records_by_split: dict[str, list[dict]], template: dict, extra: dict) -> dict:
    window_rows, feature_count = validate_contract(template)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        **template,
        "source_dir": str(root.resolve()),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split_policy": extra.pop("split_policy"),
        "evaluation_eligible": False,
        "promotion_metric_eligible": False,
        **extra,
        "splits": {},
    }
    for split in SPLITS:
        records = records_by_split.get(split, [])
        x = np.stack([row["x"] for row in records]).astype(np.float32) if records else empty_x(window_rows, feature_count)
        y = np.asarray([row["y"] for row in records], dtype=np.int64)
        metadata = [{**row["metadata"], "split": split} for row in records]
        np.savez_compressed(root / f"{split}.npz", x=x, y=y)
        (root / f"{split}_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        counts = Counter(int(value) for value in y)
        summary["splits"][split] = {
            "windows": int(len(y)),
            "non_fall": counts.get(0, 0),
            "fall": counts.get(1, 0),
            "videos": len({row["video_id"] for row in metadata}),
            "groups": len({row["split_group"] for row in metadata}),
        }
    (root / "window_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build(args: argparse.Namespace) -> dict:
    fall_train, diagnostic, source_manifest = load_fallvision(
        args.fallvision_windows, args.fallvision_manifest, set(args.holdout_fall_group)
    )
    local = load_local_negatives(args.local_npz, args.local_report)
    template = read_json(args.fallvision_windows / "window_index.json")
    augmentation = write_window_set(
        args.out / "augmentation_train_only",
        {"train": fall_train + local, "val": [], "test": []},
        template,
        {
            "schema_version": "reviewed_hybrid_augmentation_v1",
            "split_policy": "train only; FallVision manual positives excluding archive holdout + reviewed local negatives",
            "training_eligible": True,
            "known_limitations": [
                "FallVision participant identity unresolved",
                "local negatives are single-site operator protocols",
                "not usable as promotion evidence",
            ],
        },
    )
    diagnostic_summary = write_window_set(
        args.out / "fallvision_archive_holdout_diagnostic",
        {"train": [], "val": [], "test": diagnostic},
        template,
        {
            "schema_version": "fallvision_archive_holdout_diagnostic_v1",
            "split_policy": "archive-disjoint from augmentation; subject identity unresolved; diagnostic only",
            "training_eligible": False,
            "holdout_fall_groups": sorted(args.holdout_fall_group),
            "known_limitations": [
                "archive-disjoint is not proven subject-disjoint",
                "manual pilot was selected for proposer calibration",
                "not usable as promotion evidence",
            ],
        },
    )
    diagnostic_ids = {row["metadata"]["video_id"] for row in diagnostic}
    diagnostic_manifest = {
        **{key: value for key, value in source_manifest.items() if key != "items"},
        "schema_version": "temporal_manifest_v2_archive_holdout_diagnostic",
        "split_policy": "archive-disjoint diagnostic only; subject identity unresolved",
        "evaluation_eligible": False,
        "promotion_metric_eligible": False,
        "video_count": len(diagnostic_ids),
        "items": [
            {**item, "split": "test", "evaluation_eligible": False, "promotion_metric_eligible": False}
            for item in source_manifest["items"] if item["video_id"] in diagnostic_ids
        ],
    }
    diagnostic_root = args.out / "fallvision_archive_holdout_diagnostic"
    (diagnostic_root / "manifest.json").write_text(
        json.dumps(diagnostic_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "reviewed_hybrid_package_v1",
        "augmentation": augmentation["splits"],
        "diagnostic": diagnostic_summary["splits"],
        "fallvision_train_groups": sorted({row["metadata"]["split_group"] for row in fall_train}),
        "fallvision_holdout_groups": sorted({row["metadata"]["split_group"] for row in diagnostic}),
        "archive_group_overlap": [],
        "promotion_eligible": False,
        "next_step": "merge augmentation_train_only into frozen GMDCSA train; keep GMDCSA val/test unchanged",
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallvision-windows", type=Path, default=project / "external_datasets/windows/tcn_109_v2_no_missing/fallvision_pilot_balanced_diagnostic_v1_3s")
    parser.add_argument("--fallvision-manifest", type=Path, default=project / "external_datasets/manifests/fallvision_pilot_balanced_diagnostic_v1.json")
    parser.add_argument("--local-npz", type=Path, default=project / "runtime_data/curated_temporal_sessions/v1/reviewed_windows.npz")
    parser.add_argument("--local-report", type=Path, default=project / "runtime_data/curated_temporal_sessions/v1/report.json")
    parser.add_argument("--out", type=Path, default=project / "external_datasets/windows/tcn_109_v2_no_missing/reviewed_hybrid_v1")
    parser.add_argument("--holdout-fall-group", action="append", default=sorted(DEFAULT_HOLDOUT_FALL_GROUPS))
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
