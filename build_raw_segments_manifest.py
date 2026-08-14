#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rail_hint_from_name(name: str) -> str:
    return "rail" if "rail" in name.lower() else "no_rail"


def build_segments(
    video_item: dict, segment_sec: int, include_tail: bool
) -> list[dict]:
    duration = float(video_item.get("duration_sec") or 0.0)
    if duration <= 0:
        return []

    count = math.floor(duration / segment_sec)
    if include_tail and (duration - count * segment_sec) > 0:
        count += 1

    segments: list[dict] = []
    for idx in range(count):
        start_sec = idx * segment_sec
        end_sec = min((idx + 1) * segment_sec, duration)
        if end_sec <= start_sec:
            continue
        segments.append(
            {
                "segment_id": f"{Path(video_item['path']).stem}__seg_{idx:05d}",
                "video_file": video_item["file"],
                "video_path": video_item["path"],
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "rail_hint": video_item["rail_hint"],
                "rotation_profile": video_item.get("rotation_profile"),
                "rotation_decision_status": video_item.get(
                    "rotation_decision_status", "pending_definition"
                ),
                "label_status": "unlabeled",
                "notes": "",
            }
        )
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build time-window segments manifest from Raw_data metadata."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/dmc/Dataset/Raw_data/raw_rotation_manifest.json"),
        help="Raw rotation manifest path",
    )
    parser.add_argument(
        "--segment-sec",
        type=int,
        default=30,
        help="Segment length in seconds",
    )
    parser.add_argument(
        "--include-tail",
        action="store_true",
        help="Include last partial segment",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: alongside manifest)",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = data.get("items", [])

    all_segments: list[dict] = []
    for item in items:
        all_segments.extend(
            build_segments(
                video_item=item,
                segment_sec=args.segment_sec,
                include_tail=args.include_tail,
            )
        )

    out_path = args.out
    if out_path is None:
        out_path = manifest_path.parent / (
            f"raw_segments_manifest_{args.segment_sec}s.json"
        )
    out_path = out_path.resolve()

    payload = {
        "source_manifest": str(manifest_path),
        "segment_sec": args.segment_sec,
        "include_tail": args.include_tail,
        "video_count": len(items),
        "segment_count": len(all_segments),
        "segments": all_segments,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out_path)
    print(f"segments={len(all_segments)} videos={len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
