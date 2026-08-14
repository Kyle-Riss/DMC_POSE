#!/usr/bin/env python3
"""Add one ground-truth segment event (for rule validation)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LABELS = [
    "in_bed_normal",
    "edge_observe",
    "out_bed_floor",
    "out_bed_stand",
    "exit_normal",
    "unsafe_exit",
    "occluded",
    "unknown",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Append a GT segment event to segment_events.json")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("config/segment_events.json"),
        help="Output JSON (created from template if missing)",
    )
    ap.add_argument("--video", required=True, help="video file name e.g. 'Raw0 (3).mp4'")
    ap.add_argument("--start", type=float, required=True, help="start_sec")
    ap.add_argument("--end", type=float, required=True, help="end_sec")
    ap.add_argument(
        "--label",
        required=True,
        choices=DEFAULT_LABELS,
        help="event_label (GT)",
    )
    ap.add_argument("--notes", default="", help="free text")
    ap.add_argument("--by", default="", help="labeled_by")
    args = ap.parse_args()

    out = args.out.resolve()
    if out.is_file():
        data = json.loads(out.read_text(encoding="utf-8"))
    else:
        tpl = Path(__file__).resolve().parents[1] / "config" / "segment_events_template.json"
        data = json.loads(tpl.read_text(encoding="utf-8"))
        data["segments"] = []

    data.setdefault("segments", []).append(
        {
            "video_file": args.video,
            "start_sec": round(args.start, 3),
            "end_sec": round(args.end, 3),
            "event_label": args.label,
            "notes": args.notes,
            "labeled_by": args.by,
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} (+1 segment, total={len(data['segments'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
