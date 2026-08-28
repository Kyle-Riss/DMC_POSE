#!/usr/bin/env python3
"""Build a recording-disjoint RGB clip manifest for the Swin3D verifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_reviewed_staged_shadow_windows import (
    POSITIVE_STATUSES,
    recording_splits,
    reviewed_recordings,
)


def video_path(video_root: Path, camera_id: str, recording_id: str) -> Path:
    camera = camera_id.split("_", 1)[0]
    recording = recording_id.removeprefix("sim")
    return video_root / f"{camera}_{recording}.mp4"


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    finally:
        capture.release()


def bounded_window(center: int, frame_count: int, span: int = 80) -> tuple[int, int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    span = min(int(span), frame_count)
    start = max(0, min(int(center) - span // 2, frame_count - span))
    return start, start + span - 1


def clip_rows(annotation: dict[str, str], label: int, path: Path, split: str) -> list[dict]:
    frame_count = video_frame_count(path)
    if frame_count < 16:
        return []
    base = {
        "recording_id": annotation["recording_id"],
        "video_id": annotation["video_id"],
        "camera_id": annotation["camera_id"],
        "split": split,
        "video_path": str(path.resolve()),
        "frame_count": frame_count,
        "source_fps": 20.0,
        "sample_frames": 16,
        "promotion_eligible": False,
    }
    rows = []
    if label == 1:
        onset = int(annotation["fall_onset_frame"])
        impact_raw = str(annotation.get("impact_frame") or "").strip()
        impact = int(float(impact_raw)) if impact_raw and impact_raw.lower() != "nan" else onset + 40
        positive_centers = (round((onset + impact) / 2), impact + 20)
        for index, center in enumerate(positive_centers):
            start, end = bounded_window(center, frame_count)
            rows.append({**base, "clip_id": f"{annotation['video_id']}_fall_{index}", "label": 1,
                         "start_frame": start, "end_frame": end, "label_source": "reviewed_fall_transition"})
        if onset >= 24:
            start, end = bounded_window(max(0, onset - 40), frame_count)
            if end >= onset:
                end = onset - 1
                start = max(0, end - 79)
            if end - start + 1 >= 16:
                rows.append({**base, "clip_id": f"{annotation['video_id']}_prefall", "label": 0,
                             "start_frame": start, "end_frame": end, "label_source": "same_recording_prefall"})
    else:
        centers = (frame_count // 4, frame_count // 2, (3 * frame_count) // 4)
        for index, center in enumerate(centers):
            start, end = bounded_window(center, frame_count)
            rows.append({**base, "clip_id": f"{annotation['video_id']}_hardneg_{index}", "label": 0,
                         "start_frame": start, "end_frame": end, "label_source": "reviewed_hard_negative"})
    return rows


def build(annotations_path: Path, video_root: Path) -> dict:
    with annotations_path.open(newline="", encoding="utf-8-sig") as handle:
        annotation_rows = list(csv.DictReader(handle))
    selected, labels = reviewed_recordings(annotation_rows)
    splits = recording_splits(labels)
    clips = []
    missing = []
    for recording_id in sorted(selected):
        for annotation in sorted(selected[recording_id], key=lambda row: row["video_id"]):
            path = video_path(video_root, annotation["camera_id"], recording_id)
            if not path.is_file():
                missing.append({"video_id": annotation["video_id"], "path": str(path)})
                continue
            clips.extend(clip_rows(annotation, labels[recording_id], path, splits[recording_id]))
    summary = {}
    for split in ("train", "val", "test"):
        rows = [row for row in clips if row["split"] == split]
        summary[split] = {
            "clips": len(rows),
            "non_fall": sum(row["label"] == 0 for row in rows),
            "fall": sum(row["label"] == 1 for row in rows),
            "recordings": len({row["recording_id"] for row in rows}),
            "videos": len({row["video_id"] for row in rows}),
        }
    return {
        "schema_version": "dmc_swin3d_verifier_manifest_v1",
        "backbone": "torchvision_swin3d_b_kinetics400_imagenet22k_v1",
        "usage_contract": "candidate_only_shadow",
        "promotion_eligible": False,
        "split_policy": "same recording-disjoint split as temporal GRU",
        "recording_splits": dict(sorted(splits.items())),
        "clips": clips,
        "missing": missing,
        "summary": summary,
        "warnings": [
            "subject identity is unknown",
            "multiview clips are correlated",
            "metrics are engineering diagnostics only",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv")
    parser.add_argument("--video-root", type=Path, default=Path("/home/dmc/바탕화면/study_data_nor/color100"))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "external_datasets/manifests/swin3d_verifier_staged_v1.json")
    args = parser.parse_args()
    report = build(args.annotations.resolve(), args.video_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "missing": report["missing"]}, ensure_ascii=False, indent=2))
    print(f"manifest: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
