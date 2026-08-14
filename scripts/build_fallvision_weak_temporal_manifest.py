#!/usr/bin/env python3
"""Build a train-only FallVision manifest from auto proposals.

The proposed transition is explicitly ignored. These weak labels are never
eligible for validation, test, or model-promotion metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["video_id"]: row for row in csv.DictReader(handle)}


def frame_sec(frame: int, fps: float) -> float:
    return round(frame / fps, 6)


def build_item(source: dict[str, str], proposal: dict[str, str]) -> dict:
    fps = float(source["fps"])
    frame_count = int(float(source["frame_count"]))
    last_frame = max(0, frame_count - 1)
    onset_earliest = max(0, min(last_frame, int(proposal["proposed_onset_earliest_frame"])))
    impact = max(onset_earliest, min(last_frame, int(proposal["proposed_impact_frame"])))
    fall_end = max(impact, min(last_frame, int(proposal["proposed_fall_end_frame"])))
    intervals = []
    if impact > onset_earliest:
        intervals.append({
            "source_label": "auto_boundary_uncertain_transition",
            "label": "ignore",
            "start_sec": frame_sec(onset_earliest, fps),
            "end_sec": frame_sec(impact - 1, fps),
        })
    intervals.append({
        "source_label": "auto_proposed_impact_to_end",
        "label": "fall",
        "start_sec": frame_sec(impact, fps),
        "end_sec": frame_sec(fall_end, fps),
    })
    return {
        "video_id": source["video_id"],
        "dataset": "fallvision",
        "subject_id": None,
        "split": "train",
        "source_group": "fall",
        "source_path": str(Path(source["local_video_path"]).resolve()),
        "video_path": str(Path(source["local_video_path"]).resolve()),
        "scene_id": source["scene_id"],
        "camera_id": None,
        "activity_label": "fall",
        "binary_fall_label": 1,
        "fall_type": f"fall_from_{source['scene_id']}",
        "bed_related": source["scene_id"] == "bed",
        "staged_or_real": "staged",
        "fall_start_sec": frame_sec(int(proposal["proposed_fall_onset_frame"]), fps),
        "impact_sec": frame_sec(impact, fps),
        "fall_end_sec": frame_sec(fall_end, fps),
        "annotation_source": "frame_motion_pose_v2_auto_proposal",
        "annotation_scope": "weak_interval_train_only",
        "annotation_quality": "weak_auto_review_required",
        "evaluation_eligible": False,
        "promotion_metric_eligible": False,
        "split_group": f"fallvision_archive:fall:{source['scene_id']}:{source['chunk_id']}",
        "has_video": True,
        "training_eligible": True,
        "excluded_reasons": [],
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": float(source["duration_sec"]),
        "width": int(float(source["width"])),
        "height": int(float(source["height"])),
        "media_sha256": source["media_sha256"],
        "proposal_status": proposal["proposal_status"],
        "intervals": intervals,
    }


def build_non_fall_item(source: dict[str, str]) -> dict:
    return {
        "video_id": source["video_id"], "dataset": "fallvision", "subject_id": None,
        "split": "train", "source_group": "non_fall",
        "source_path": str(Path(source["local_video_path"]).resolve()),
        "video_path": str(Path(source["local_video_path"]).resolve()),
        "scene_id": source["scene_id"], "camera_id": None,
        "activity_label": "non_fall", "binary_fall_label": 0, "fall_type": None,
        "bed_related": source["scene_id"] == "bed", "staged_or_real": "staged",
        "fall_start_sec": None, "impact_sec": None, "fall_end_sec": None,
        "annotation_source": "fallvision_official_video_label",
        "annotation_scope": "video_non_fall_train_only",
        "annotation_quality": "official_video_level",
        "evaluation_eligible": False, "promotion_metric_eligible": False,
        "split_group": f"fallvision_archive:non_fall:{source['scene_id']}:{source['chunk_id']}",
        "has_video": True, "training_eligible": True, "excluded_reasons": [],
        "fps": float(source["fps"]), "frame_count": int(float(source["frame_count"])),
        "duration_sec": float(source["duration_sec"]),
        "width": int(float(source["width"])), "height": int(float(source["height"])),
        "media_sha256": source["media_sha256"], "proposal_status": None,
        "intervals": [],
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=project / "external_datasets/annotations/fallvision_weak_train_v1.csv")
    parser.add_argument("--proposals", type=Path, default=project / "external_datasets/annotations/fallvision_weak_train_v1_proposals.csv")
    parser.add_argument("--out", type=Path, default=project / "external_datasets/manifests/fallvision_weak_train_v1.json")
    args = parser.parse_args()
    sources = read_by_id(args.sources)
    proposals = read_by_id(args.proposals)
    fall_ids = {video_id for video_id in sources if ":fall:" in video_id}
    missing = sorted(fall_ids - set(proposals))
    if missing:
        raise ValueError(f"missing proposals for {len(missing)} videos: {missing[:3]}")
    items = [
        build_item(sources[video_id], proposals[video_id])
        if video_id in fall_ids else build_non_fall_item(sources[video_id])
        for video_id in sorted(sources)
    ]
    manifest = {
        "schema_version": "temporal_manifest_v2_weak_ignore",
        "dataset": "fallvision",
        "sample_hz_target": 10.0,
        "split_policy": "train_only_provenance_chunk_1; subject identity unresolved",
        "label_policy": "fall clips: non_fall before onset_earliest, ignore through impact-1, fall from impact through proposed end; non-fall clips: all non_fall",
        "evaluation_eligible": False,
        "promotion_metric_eligible": False,
        "video_count": len(items),
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(items), "output": str(args.out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
