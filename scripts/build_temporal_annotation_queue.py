#!/usr/bin/env python3
"""Create a human-only temporal annotation queue from a diagnostic manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


FIELDS = (
    "video_id",
    "dataset",
    "scene_id",
    "camera_id",
    "recording_id",
    "split_group",
    "subject_id",
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
    "training_eligible",
    "training_blockers",
)


def recording_id(item: dict) -> str:
    stem = Path(str(item["video_path"])).stem
    match = re.match(r"^c\d+_(.+)$", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def build_queue(manifest: dict) -> list[dict[str, str]]:
    dataset = str(manifest["dataset"])
    rows = []
    for item in manifest.get("items", []):
        if not item.get("readable"):
            continue
        group = recording_id(item)
        camera = str(item.get("camera_id") or "unknown_camera")
        subject = item.get("subject_id")
        blockers = ["temporal_annotation_incomplete"]
        if not subject:
            blockers.append("subject_identity_unknown")
        rows.append({
            "video_id": str(item["video_id"]),
            "dataset": dataset,
            "scene_id": f"staged_fall_{camera}",
            "camera_id": camera,
            "recording_id": group,
            "split_group": f"{dataset}:{group}",
            "subject_id": "" if subject is None else str(subject),
            "local_video_path": str(Path(item["video_path"]).resolve()),
            "media_sha256": str(item.get("media_sha256") or ""),
            "decode_ok": "true",
            "fps": str(item["fps"]),
            "frame_count": str(item["frame_count"]),
            "duration_sec": str(item["duration_sec"]),
            "width": str(item["width"]),
            "height": str(item["height"]),
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
            "training_eligible": "false",
            "training_blockers": ";".join(blockers),
        })
    return rows


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project / "external_datasets/manifests/usb_sim_falldown_diagnostic.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing queue")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite annotations: {args.out}")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = build_queue(payload)
    if not rows:
        raise ValueError("manifest has no readable videos")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    groups = Counter(row["recording_id"] for row in rows)
    summary = {
        "rows": len(rows),
        "recording_groups": len(groups),
        "views_per_group": dict(sorted(Counter(groups.values()).items())),
        "training_eligible": 0,
        "output": str(args.out.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
