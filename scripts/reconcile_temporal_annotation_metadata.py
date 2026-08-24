#!/usr/bin/env python3
"""Reconcile annotation media metadata with a sequentially decoded manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BOUNDARY_FIELDS = (
    "fall_onset_frame",
    "impact_frame",
    "post_fall_stable_frame",
    "fall_end_frame",
    "onset_earliest_frame",
    "onset_latest_frame",
)


def reconcile(rows: list[dict[str, str]], manifest: dict) -> tuple[list[dict[str, str]], dict]:
    audited = {str(item["video_id"]): item for item in manifest.get("items", [])}
    if not audited:
        raise ValueError("audited manifest has no items")
    output = []
    changed = []
    for source in rows:
        row = dict(source)
        video_id = str(row.get("video_id") or "")
        if video_id not in audited:
            raise ValueError(f"video missing from audited manifest: {video_id}")
        item = audited[video_id]
        decoded_count = int(item["frame_count"])
        for field in BOUNDARY_FIELDS:
            value = str(row.get(field) or "").strip()
            if value and not 0 <= int(value) < decoded_count:
                raise ValueError(
                    f"{video_id}: {field}={value} outside decoded frame range "
                    f"0..{decoded_count - 1}"
                )
        previous = int(float(row["frame_count"]))
        if previous != decoded_count:
            changed.append({
                "video_id": video_id,
                "previous_frame_count": previous,
                "decoded_frame_count": decoded_count,
            })
        row["decode_ok"] = "true" if item.get("readable") else "false"
        row["fps"] = str(item["fps"])
        row["frame_count"] = str(decoded_count)
        row["duration_sec"] = str(item["duration_sec"])
        row["width"] = str(item["width"])
        row["height"] = str(item["height"])
        output.append(row)
    return output, {
        "rows": len(output),
        "changed_rows": len(changed),
        "completed_rows_preserved": sum(
            row.get("annotation_status") == "complete" for row in output
        ),
        "changes": changed,
    }


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project / "external_datasets/manifests/usb_sim_falldown_diagnostic.json",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reconciled, report = reconcile(rows, manifest)
    destination = args.out or args.annotations
    write_rows(destination, fields, reconciled)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"annotations: {destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
