#!/usr/bin/env python3
"""Convert fully reviewed multiview staged clips into a leakage-safe manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = {"train", "val", "test"}
RESOLVED_STATUSES = {"complete", "excluded", "needs_adjudication"}


def _number(row: dict[str, str], field: str) -> int:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{row.get('video_id')}: missing {field}")
    return int(value)


def build_manifest(rows: list[dict[str, str]], identity: dict) -> dict:
    unresolved = [
        row["video_id"] for row in rows
        if row.get("annotation_status") not in RESOLVED_STATUSES
    ]
    if unresolved:
        raise ValueError(f"annotations still unresolved: {len(unresolved)}")
    recording_statuses: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        recording_statuses[row["recording_id"]].add(row["annotation_status"])
    mixed = {
        recording: sorted(statuses)
        for recording, statuses in recording_statuses.items()
        if len(statuses) != 1
    }
    if mixed:
        raise ValueError(f"multiview recording status disagreement: {mixed}")
    eligible_rows = [row for row in rows if row["annotation_status"] == "complete"]
    if not eligible_rows:
        raise ValueError("no complete annotations available")
    mappings = identity.get("recordings") or {}
    items = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in eligible_rows:
        recording = row["recording_id"]
        mapping = mappings.get(recording)
        if not isinstance(mapping, dict):
            raise ValueError(f"missing identity mapping for recording: {recording}")
        subject = str(mapping.get("subject_id") or "").strip()
        session = str(mapping.get("session_id") or "").strip()
        split = str(mapping.get("split") or "").strip()
        if not subject or not session:
            raise ValueError(f"incomplete subject/session mapping: {recording}")
        if split not in SPLITS:
            raise ValueError(f"invalid split for {recording}: {split}")
        group_splits[recording].add(split)

        fps = float(row["fps"])
        frame_count = int(float(row["frame_count"]))
        onset = _number(row, "fall_onset_frame")
        impact = _number(row, "impact_frame")
        stable = _number(row, "post_fall_stable_frame")
        end = _number(row, "fall_end_frame")
        if not 0 <= onset <= impact <= stable <= end < frame_count:
            raise ValueError(f"{row['video_id']}: invalid boundary order")
        sec = lambda frame: round(frame / fps, 6)
        item = {
            "video_id": row["video_id"],
            "dataset": row.get("dataset") or "hospital_staged",
            "subject_id": subject,
            "session_id": session,
            "recording_id": recording,
            "split": split,
            "split_group": f"subject:{subject}",
            "multiview_group": f"session:{session}:recording:{recording}",
            "camera_id": row["camera_id"],
            "scene_id": row["scene_id"],
            "source_path": str(Path(row["local_video_path"]).resolve()),
            "video_path": str(Path(row["local_video_path"]).resolve()),
            "activity_label": "fall",
            "binary_fall_label": 1,
            "staged_or_real": "staged",
            "fall_start_sec": sec(onset),
            "impact_sec": sec(impact),
            "post_fall_stable_sec": sec(stable),
            "fall_end_sec": sec(end),
            "annotation_source": "manual_review",
            "annotation_scope": "manual_temporal_interval",
            "annotation_confidence": row.get("annotation_confidence") or "",
            "annotator": row.get("annotator") or "",
            "training_eligible": True,
            "temporal_tcn_eligible": True,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": float(row["duration_sec"]),
            "width": int(float(row["width"])),
            "height": int(float(row["height"])),
            "media_sha256": row.get("media_sha256") or "",
            "intervals": [{
                "source_label": "manual_fall_onset_to_end",
                "label": "fall",
                "start_sec": sec(onset),
                "end_sec": sec(end),
            }],
        }
        items.append(item)

    leaking = {group: sorted(values) for group, values in group_splits.items() if len(values) != 1}
    if leaking:
        raise ValueError(f"multiview recording split leakage: {leaking}")
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for item in items:
        subject_splits[item["subject_id"]].add(item["split"])
    subject_leaks = {subject: sorted(values) for subject, values in subject_splits.items() if len(values) != 1}
    if subject_leaks:
        raise ValueError(f"subject split leakage: {subject_leaks}")

    split_counts = Counter(item["split"] for item in items)
    view_status_counts = Counter(row["annotation_status"] for row in rows)
    recording_status_counts = Counter(
        next(iter(statuses)) for statuses in recording_statuses.values()
    )
    return {
        "schema_version": "temporal_manifest_v2_reviewed_multiview",
        "dataset": "hospital_staged_multiview",
        "sample_hz_target": 20.0,
        "split_policy": "subject-disjoint with multiview recording lock",
        "task": "binary temporal event detection",
        "video_count": len(items),
        "recording_count": len(group_splits),
        "subject_count": len(subject_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "review_summary": {
            "view_status_counts": dict(sorted(view_status_counts.items())),
            "recording_status_counts": dict(sorted(recording_status_counts.items())),
            "policy": "complete included; excluded and needs_adjudication held out",
        },
        "items": items,
    }


def identity_template(rows: list[dict[str, str]]) -> dict:
    recordings = {}
    for row in rows:
        if row.get("annotation_status") != "complete":
            continue
        recordings.setdefault(row["recording_id"], {
            "subject_id": None,
            "session_id": None,
            "split": None,
            "camera_views": [],
        })["camera_views"].append(row["camera_id"])
    for value in recordings.values():
        value["camera_views"] = sorted(set(value["camera_views"]))
    return {
        "schema_version": "dmc_staged_identity_split_v1",
        "warning": "Human-enter subject/session identity for complete training candidates only. Do not assign split until subject groups are known.",
        "recordings": dict(sorted(recordings.items())),
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path,
        default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv",
    )
    parser.add_argument(
        "--identity", type=Path,
        default=project / "external_datasets/annotations/usb_sim_falldown_identity_v1.json",
    )
    parser.add_argument(
        "--out", type=Path,
        default=project / "external_datasets/manifests/usb_sim_falldown_reviewed_v1.json",
    )
    parser.add_argument("--write-identity-template", action="store_true")
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.write_identity_template:
        if args.identity.exists():
            raise FileExistsError(f"refusing to overwrite identity file: {args.identity}")
        args.identity.parent.mkdir(parents=True, exist_ok=True)
        args.identity.write_text(
            json.dumps(identity_template(rows), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"identity template: {args.identity.resolve()}")
        return 0
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    manifest = build_manifest(rows, identity)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "video_count", "recording_count", "subject_count", "split_counts"
    )}, ensure_ascii=False, indent=2))
    print(f"manifest: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
