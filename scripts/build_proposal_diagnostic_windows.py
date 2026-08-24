#!/usr/bin/env python3
"""Build non-promotable real-feature windows from uncalibrated review proposals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_temporal_windows import windows_from_video
from temporal_features import FEATURE_SCHEMA_VERSION, temporal_feature_names
from temporal_sequence import cadence_interval_bounds, observed_sequence_contract


def recording_splits(recordings: list[str]) -> dict[str, str]:
    ordered = sorted(set(recordings))
    if len(ordered) < 3:
        raise ValueError("at least three recording groups are required")
    train_end = max(1, int(round(len(ordered) * 0.6)))
    val_end = min(len(ordered) - 1, train_end + max(1, int(round(len(ordered) * 0.2))))
    return {
        recording: "train" if index < train_end else "val" if index < val_end else "test"
        for index, recording in enumerate(ordered)
    }


def apply_proposal_targets(frame: pd.DataFrame, onset_frame: int, end_frame: int, split: str) -> pd.DataFrame:
    if onset_frame < 0 or end_frame < onset_frame:
        raise ValueError("invalid proposal boundary order")
    output = frame.copy()
    is_fall = (output["frame_idx"].astype(int) >= onset_frame) & (output["frame_idx"].astype(int) <= end_frame)
    output["target"] = np.where(is_fall, "fall", "non_fall")
    output["active_labels"] = np.where(is_fall, "fall_proposal", "")
    output["split"] = split
    return output


def build(features_dir: Path, annotations_path: Path, proposals_path: Path, out_dir: Path) -> dict:
    index = json.loads((features_dir / "features_index.json").read_text(encoding="utf-8"))
    if index.get("sequence_contract_version") != observed_sequence_contract(20.0):
        raise ValueError("source features are not observed_only_20hz_v1")
    with annotations_path.open(newline="", encoding="utf-8-sig") as handle:
        annotations = {row["video_id"]: row for row in csv.DictReader(handle)}
    with proposals_path.open(newline="", encoding="utf-8-sig") as handle:
        proposals = {row["video_id"]: row for row in csv.DictReader(handle)}
    if set(proposals) != set(annotations):
        raise ValueError("proposal and annotation video sets do not match")
    split_by_recording = recording_splits([row["recording_id"] for row in annotations.values()])
    result_paths = {row["video_id"]: Path(row["out"]) for row in index.get("results", []) if row.get("status") == "ok"}
    all_by_split = {split: {"x": [], "y": [], "meta": []} for split in ("train", "val", "test")}
    min_interval, max_interval = cadence_interval_bounds(20.0)
    excluded = []
    for video_id in sorted(annotations):
        annotation = annotations[video_id]
        proposal = proposals[video_id]
        path = result_paths.get(video_id)
        if path is None or not path.is_file() or path.stat().st_size == 0:
            excluded.append({"video_id": video_id, "reason": "missing_or_empty_feature_csv"})
            continue
        recording = annotation["recording_id"]
        split = split_by_recording[recording]
        frame = apply_proposal_targets(
            pd.read_csv(path),
            int(proposal["proposed_fall_onset_frame"]),
            int(proposal["proposed_fall_end_frame"]),
            split,
        )
        windows, targets, metadata, names = windows_from_video(
            frame, 80, 5, min_interval_sec=min_interval, max_interval_sec=max_interval,
        )
        if names != temporal_feature_names():
            raise ValueError("feature order mismatch")
        for row in metadata:
            row.update({
                "recording_id": recording,
                "camera_id": annotation["camera_id"],
                "label_source": "uncalibrated_motion_review_proposal",
                "promotion_eligible": False,
            })
        all_by_split[split]["x"].extend(windows)
        all_by_split[split]["y"].extend(targets)
        all_by_split[split]["meta"].extend(metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "window_schema_version": "pose_gru_proposal_diagnostic_v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sequence_contract_version": observed_sequence_contract(20.0),
        "data_provenance": "real_usb_pose_with_uncalibrated_motion_review_proposals",
        "synthetic_smoke_fixture": False,
        "promotion_eligible": False,
        "accuracy_claim": False,
        "subject_identity_known": False,
        "split_policy": "recording-disjoint diagnostic only; subject leakage remains possible",
        "window_sec": 4.0,
        "stride_sec": 0.25,
        "sample_hz": 20.0,
        "window_rows": 80,
        "stride_rows": 5,
        "feature_count": 109,
        "feature_names": temporal_feature_names(),
        "recording_splits": split_by_recording,
        "excluded": excluded,
        "splits": {},
        "warnings": [
            "proposal boundaries are uncalibrated navigation aids, not ground truth",
            "subject identity is unknown and recording-disjoint splits may leak people",
            "do not report model accuracy or promote checkpoints from this corpus",
        ],
    }
    for split, values in all_by_split.items():
        x = np.asarray(values["x"], dtype=np.float32)
        y = np.asarray(values["y"], dtype=np.int64)
        if len(x) == 0:
            x = np.empty((0, 80, 109), dtype=np.float32)
        np.savez_compressed(out_dir / f"{split}.npz", x=x, y=y)
        (out_dir / f"{split}_metadata.json").write_text(json.dumps(values["meta"], ensure_ascii=False, indent=2), encoding="utf-8")
        counts = Counter(y.tolist())
        report["splits"][split] = {
            "windows": len(y), "non_fall": counts.get(0, 0), "fall": counts.get(1, 0),
            "videos": len({row["video_id"] for row in values["meta"]}),
            "recordings": len({row["recording_id"] for row in values["meta"]}),
        }
    (out_dir / "window_index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--features-dir", type=Path, default=project / "external_datasets/features/pose_109_observed_only_20hz/usb_sim_falldown_full")
    parser.add_argument("--annotations", type=Path, default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv")
    parser.add_argument("--proposals", type=Path, default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1_proposals.csv")
    parser.add_argument("--out-dir", type=Path, default=project / "external_datasets/windows/diagnostic/usb_proposal_gru_80x109_20hz_v1")
    args = parser.parse_args()
    report = build(args.features_dir.resolve(), args.annotations.resolve(), args.proposals.resolve(), args.out_dir.resolve())
    print(json.dumps({"promotion_eligible": report["promotion_eligible"], "splits": report["splits"], "excluded": report["excluded"]}, ensure_ascii=False, indent=2))
    print(f"window_index: {(args.out_dir / 'window_index.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
