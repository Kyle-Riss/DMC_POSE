#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SCENES = ("bed", "chair", "stand")
ANNOTATION_FIELDS = (
    "video_id",
    "scene_id",
    "chunk_id",
    "recording_id",
    "raw_archive",
    "raw_member",
    "pair_status",
    "local_video_path",
    "media_sha256",
    "decode_ok",
    "fps",
    "frame_count",
    "duration_sec",
    "width",
    "height",
    "fall_onset_frame",
    "impact_frame",
    "post_fall_stable_frame",
    "fall_end_frame",
    "onset_earliest_frame",
    "onset_latest_frame",
    "annotation_status",
    "annotation_confidence",
    "annotator",
    "notes",
)


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def build_pilot(items: list[dict], per_scene: int) -> list[dict]:
    selected: list[dict] = []
    for scene in SCENES:
        candidates = [
            row
            for row in items
            if row["activity_label"] == "fall"
            and row["scene_id"] == scene
            and row["raw_video_exists"]
            and row["pair_status"] == "complete"
        ]
        candidates.sort(key=lambda row: (row["chunk_id"], row["recording_id"]))
        for row in evenly_spaced(candidates, per_scene):
            selected.append(
                {
                    "video_id": row["canonical_id"],
                    "scene_id": scene,
                    "chunk_id": row["chunk_id"],
                    "recording_id": row["recording_id"],
                    "raw_archive": row["raw_archive"],
                    "raw_member": row["raw_member"],
                    "pair_status": row["pair_status"],
                    "local_video_path": "",
                    "media_sha256": "",
                    "decode_ok": "",
                    "fps": "",
                    "frame_count": "",
                    "duration_sec": "",
                    "width": "",
                    "height": "",
                    "fall_onset_frame": "",
                    "impact_frame": "",
                    "post_fall_stable_frame": "",
                    "fall_end_frame": "",
                    "onset_earliest_frame": "",
                    "onset_latest_frame": "",
                    "annotation_status": "unreviewed",
                    "annotation_confidence": "",
                    "annotator": "",
                    "notes": "",
                }
            )
    return selected


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory",
        type=Path,
        default=project / "external_datasets/manifests/fallvision_canonical_inventory_v1.json",
    )
    parser.add_argument("--per-scene", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=project / "external_datasets/annotations/fallvision_pilot_v1.csv",
    )
    args = parser.parse_args()
    payload = json.loads(args.inventory.read_text())
    rows = build_pilot(payload["items"], args.per_scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "scene_counts": dict(Counter(row["scene_id"] for row in rows)), "output": str(args.out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
