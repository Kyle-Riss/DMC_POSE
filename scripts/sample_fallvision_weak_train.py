#!/usr/bin/env python3
"""Select a provenance-isolated FallVision weak-label training batch."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sample_fallvision_annotation_pilot import ANNOTATION_FIELDS, SCENES, evenly_spaced


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=project / "external_datasets/manifests/fallvision_canonical_inventory_v1.json",
    )
    parser.add_argument(
        "--manual-annotations",
        type=Path,
        default=project / "external_datasets/annotations/fallvision_pilot_v1_complete.csv",
    )
    parser.add_argument("--per-scene", type=int, default=24)
    parser.add_argument("--chunks", default="1")
    parser.add_argument("--activity", choices=("fall", "non_fall"), default="fall")
    parser.add_argument(
        "--out",
        type=Path,
        default=project / "external_datasets/annotations/fallvision_weak_train_v1.csv",
    )
    args = parser.parse_args()
    allowed_chunks = {int(value) for value in args.chunks.split(",") if value.strip()}
    with args.manual_annotations.open(newline="", encoding="utf-8") as handle:
        excluded = {row["video_id"] for row in csv.DictReader(handle)}
    items = json.loads(args.inventory.read_text(encoding="utf-8"))["items"]
    selected = []
    for scene in SCENES:
        candidates = [
            row for row in items
            if row["activity_label"] == args.activity
            and row["scene_id"] == scene
            and int(row["chunk_id"]) in allowed_chunks
            and row["pair_status"] == "complete"
            and row["raw_video_exists"]
            and row["canonical_id"] not in excluded
        ]
        candidates.sort(key=lambda row: row["recording_id"])
        for row in evenly_spaced(candidates, args.per_scene):
            selected.append({
                "video_id": row["canonical_id"],
                "scene_id": scene,
                "chunk_id": row["chunk_id"],
                "recording_id": row["recording_id"],
                "raw_archive": row["raw_archive"],
                "raw_member": row["raw_member"],
                "pair_status": row["pair_status"],
                "local_video_path": "", "media_sha256": "", "decode_ok": "",
                "fps": "", "frame_count": "", "duration_sec": "", "width": "", "height": "",
                "fall_onset_frame": "", "impact_frame": "", "post_fall_stable_frame": "",
                "fall_end_frame": "", "onset_earliest_frame": "", "onset_latest_frame": "",
                "annotation_status": "unreviewed", "annotation_confidence": "",
                "annotator": "", "notes": (
                    "weak_train_only; never evaluation ground truth"
                    if args.activity == "fall"
                    else "official_video_level_non_fall; train_only"
                ),
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    print(json.dumps({
        "rows": len(selected),
        "activity": args.activity,
        "chunks": sorted(allowed_chunks),
        "scene_counts": dict(Counter(row["scene_id"] for row in selected)),
        "excluded_manual_count": len(excluded),
        "output": str(args.out.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
