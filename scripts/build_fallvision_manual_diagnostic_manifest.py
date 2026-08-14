#!/usr/bin/env python3
"""Convert completed manual FallVision pilot labels to diagnostic-only GT."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_item(row: dict[str, str]) -> dict:
    if row["annotation_status"] != "complete":
        raise ValueError(f"annotation is not complete: {row['video_id']}")
    fps = float(row["fps"])
    onset = int(row["fall_onset_frame"])
    impact = int(row["impact_frame"])
    stable = int(row["post_fall_stable_frame"])
    end = int(row["fall_end_frame"])
    if not 0 <= onset <= impact <= stable <= end < int(float(row["frame_count"])):
        raise ValueError(f"invalid manual boundary order: {row['video_id']}")
    sec = lambda frame: round(frame / fps, 6)
    return {
        "video_id": row["video_id"], "dataset": "fallvision", "subject_id": None,
        "split": "test", "source_group": "fall",
        "source_path": str(Path(row["local_video_path"]).resolve()),
        "video_path": str(Path(row["local_video_path"]).resolve()),
        "scene_id": row["scene_id"], "camera_id": None,
        "activity_label": "fall", "binary_fall_label": 1,
        "fall_type": f"fall_from_{row['scene_id']}", "bed_related": row["scene_id"] == "bed",
        "staged_or_real": "staged", "fall_start_sec": sec(onset),
        "impact_sec": sec(impact), "post_fall_stable_sec": sec(stable), "fall_end_sec": sec(end),
        "annotation_source": row["annotator"] or "manual_pilot",
        "annotation_scope": "manual_interval_diagnostic_only",
        "annotation_confidence": row["annotation_confidence"],
        "evaluation_eligible": False, "promotion_metric_eligible": False,
        "leakage_risk": "subject_identity_unresolved; proposer_calibration_pilot",
        "split_group": f"fallvision_archive:fall:{row['scene_id']}:{row['chunk_id']}",
        "has_video": True, "training_eligible": False,
        "fps": fps, "frame_count": int(float(row["frame_count"])),
        "duration_sec": float(row["duration_sec"]),
        "width": int(float(row["width"])), "height": int(float(row["height"])),
        "media_sha256": row["media_sha256"],
        "intervals": [{
            "source_label": "manual_fall_onset_to_end", "label": "fall",
            "start_sec": sec(onset), "end_sec": sec(end),
        }],
    }


def build_non_fall_item(row: dict[str, str]) -> dict:
    return {
        "video_id": row["video_id"], "dataset": "fallvision", "subject_id": None,
        "split": "test", "source_group": "non_fall",
        "source_path": str(Path(row["local_video_path"]).resolve()),
        "video_path": str(Path(row["local_video_path"]).resolve()),
        "scene_id": row["scene_id"], "camera_id": None,
        "activity_label": "non_fall", "binary_fall_label": 0, "fall_type": None,
        "bed_related": row["scene_id"] == "bed", "staged_or_real": "staged",
        "fall_start_sec": None, "impact_sec": None, "fall_end_sec": None,
        "annotation_source": "fallvision_official_video_label",
        "annotation_scope": "video_non_fall_diagnostic_only",
        "evaluation_eligible": False, "promotion_metric_eligible": False,
        "leakage_risk": "subject_identity_unresolved",
        "split_group": f"fallvision_archive:non_fall:{row['scene_id']}:{row['chunk_id']}",
        "has_video": True, "training_eligible": False,
        "fps": float(row["fps"]), "frame_count": int(float(row["frame_count"])),
        "duration_sec": float(row["duration_sec"]),
        "width": int(float(row["width"])), "height": int(float(row["height"])),
        "media_sha256": row["media_sha256"], "intervals": [],
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=project / "external_datasets/annotations/fallvision_pilot_v1_complete.csv")
    parser.add_argument("--non-fall-annotations", type=Path)
    parser.add_argument("--out", type=Path, default=project / "external_datasets/manifests/fallvision_pilot_manual_diagnostic_v1.json")
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        annotation_rows = list(csv.DictReader(handle))
    completed_rows = [row for row in annotation_rows if row["annotation_status"] == "complete"]
    excluded_rows = [row for row in annotation_rows if row["annotation_status"] == "excluded"]
    unresolved_rows = [
        row for row in annotation_rows
        if row["annotation_status"] not in {"complete", "excluded"}
    ]
    if unresolved_rows:
        raise ValueError(f"annotations still unresolved: {len(unresolved_rows)}")
    items = [build_item(row) for row in completed_rows]
    if args.non_fall_annotations:
        with args.non_fall_annotations.open(newline="", encoding="utf-8") as handle:
            items.extend(build_non_fall_item(row) for row in csv.DictReader(handle))
    manifest = {
        "schema_version": "temporal_manifest_v2_manual_diagnostic",
        "dataset": "fallvision", "sample_hz_target": 10.0,
        "split_policy": "diagnostic test only; not promotion eligible",
        "evaluation_eligible": False, "promotion_metric_eligible": False,
        "video_count": len(items), "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "videos": len(items),
        "completed": len(completed_rows),
        "excluded": len(excluded_rows),
        "unresolved": len(unresolved_rows),
        "output": str(args.out.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
