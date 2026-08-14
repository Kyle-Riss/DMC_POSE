#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply labeled CSV fields back into raw segments manifest."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/dmc/Dataset/Raw_data/raw_segments_30s_labeling_template.csv"),
        help="Labeled CSV path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/dmc/Dataset/Raw_data/raw_segments_manifest_30s.json"),
        help="Target manifest path",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create manifest backup before writing",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    manifest_path = args.manifest.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])
    by_id = {s.get("segment_id"): s for s in segments}

    touched = 0
    updated = 0
    skipped_missing = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        sid = row.get("segment_id", "")
        if not sid:
            continue
        seg = by_id.get(sid)
        if seg is None:
            skipped_missing += 1
            continue
        touched += 1

        changed = False
        for key in (
            "rotation_profile",
            "rotation_decision_status",
            "label_status",
            "notes",
        ):
            val = row.get(key, "")
            if val is None:
                continue
            val = val.strip()
            # empty string means keep existing value
            if val == "":
                continue
            if seg.get(key) != val:
                seg[key] = val
                changed = True
        if changed:
            updated += 1

    if args.backup:
        backup_path = manifest_path.with_suffix(".json.bak")
        backup_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"csv_rows={len(rows)} touched={touched} updated={updated} missing_segment_id={skipped_missing}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
