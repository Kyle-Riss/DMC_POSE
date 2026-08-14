#!/usr/bin/env python3
"""Build a subject-disjoint temporal manifest for the local GMDCSA-24 copy."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import cv2

SPLIT_BY_SUBJECT = {
    "Subject 1": "train",
    "Subject 2": "train",
    "Subject 3": "val",
    "Subject 4": "test",
}

_INTERVAL_RE = re.compile(
    r"(?P<label>[^;\[]+?)\s*\[\s*(?P<start>\d+(?:\.\d+)?)\s*"
    r"to\s*(?P<end>\d+(?:\.\d+)?)\s*\]",
    flags=re.IGNORECASE,
)


def canonical_label(raw_label: str) -> str:
    label = " ".join(raw_label.strip(" :").lower().split())
    if label.startswith("fall"):
        return "fall"
    aliases = {
        "sleeping": "lying",
        "sitting": "sitting",
        "standing": "standing",
        "walking": "walking",
        "reading": "reading",
        "exercise": "exercise",
    }
    return aliases.get(label, label.replace(" ", "_"))


def parse_intervals(text: str, duration_sec: float) -> list[dict]:
    intervals: list[dict] = []
    for match in _INTERVAL_RE.finditer(text or ""):
        start = max(0.0, float(match.group("start")))
        end = min(float(duration_sec), float(match.group("end")))
        if end < start:
            start, end = end, start
        raw = " ".join(match.group("label").strip(" :").split())
        intervals.append(
            {
                "source_label": raw,
                "label": canonical_label(raw),
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
            }
        )
    return intervals


def video_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"readable": False, "fps": None, "frame_count": None, "duration_sec": None}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else None
    return {
        "readable": fps > 0 and frame_count > 0,
        "fps": round(fps, 6) if fps > 0 else None,
        "frame_count": frame_count,
        "duration_sec": round(duration, 6) if duration is not None else None,
        "width": width,
        "height": height,
    }


def build_manifest(root: Path) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    errors: list[str] = []

    for subject, split in SPLIT_BY_SUBJECT.items():
        subject_dir = root / subject
        for source_group in ("Fall", "ADL"):
            csv_path = subject_dir / f"{source_group}.csv"
            if not csv_path.is_file():
                errors.append(f"missing csv: {csv_path}")
                continue

            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

            for row in rows:
                filename = (row.get("File Name") or "").strip()
                video_path = subject_dir / source_group / filename
                if not video_path.is_file():
                    errors.append(f"missing video: {video_path}")
                    continue

                declared_duration = float(row.get("Length (seconds)") or 0.0)
                meta = video_metadata(video_path)
                actual_duration = meta.get("duration_sec")
                label_duration = actual_duration or declared_duration
                intervals = parse_intervals(row.get(" Classes") or "", label_duration)
                if not intervals:
                    errors.append(f"no intervals parsed: {csv_path}:{filename}")

                items.append(
                    {
                        "video_id": f"gmdcsa24_{subject.lower().replace(' ', '')}_{source_group.lower()}_{Path(filename).stem}",
                        "dataset": "gmdcsa24",
                        "subject_id": subject.lower().replace(" ", "_"),
                        "split": split,
                        "source_group": source_group.lower(),
                        "source_path": str(video_path.resolve()),
                        "video_path": str(video_path.resolve()),
                        "source_csv": str(csv_path.resolve()),
                        "scene_id": None,
                        "camera_id": None,
                        "activity_label": "fall" if source_group == "Fall" else "adl",
                        "binary_fall_label": int(any(interval["label"] == "fall" for interval in intervals)),
                        "fall_type": None,
                        "bed_related": None,
                        "staged_or_real": "staged",
                        "fall_start_sec": next((interval["start_sec"] for interval in intervals if interval["label"] == "fall"), None),
                        "impact_sec": None,
                        "fall_end_sec": next((interval["end_sec"] for interval in intervals if interval["label"] == "fall"), None),
                        "annotation_source": str(csv_path.resolve()),
                        "annotation_scope": "interval",
                        "split_group": "gmdcsa24_subject:" + subject.lower().replace(" ", "_"),
                        "has_video": True,
                        "training_eligible": bool(meta.get("readable") and intervals),
                        "excluded_reasons": [] if meta.get("readable") and intervals else ["unreadable_or_missing_intervals"],
                        "declared_duration_sec": declared_duration,
                        "recording_condition": (row.get("Time of Recording") or "").strip(),
                        "description": (row.get("Description") or "").strip(),
                        "intervals": intervals,
                        **meta,
                    }
                )

    return items, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "external_datasets" / "gmdcsa24",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "external_datasets" / "manifests" / "gmdcsa24.json",
    )
    parser.add_argument("--strict", action="store_true", help="return non-zero on any manifest error")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    items, errors = build_manifest(root)
    split_counts = Counter(item["split"] for item in items)
    group_counts = Counter(item["source_group"] for item in items)
    interval_counts = Counter(
        interval["label"] for item in items for interval in item["intervals"]
    )
    payload = {
        "schema_version": "temporal_manifest_v2",
        "dataset": "gmdcsa24",
        "root": str(root),
        "sample_hz_target": 10.0,
        "split_policy": "subject_disjoint: subjects 1-2 train, 3 val, 4 test",
        "task": "binary temporal event detection (fall vs non_fall)",
        "video_count": len(items),
        "split_counts": dict(sorted(split_counts.items())),
        "source_group_counts": dict(sorted(group_counts.items())),
        "interval_label_counts": dict(sorted(interval_counts.items())),
        "errors": errors,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("video_count", "split_counts", "source_group_counts", "interval_label_counts", "errors")}, ensure_ascii=False, indent=2))
    print(f"manifest: {args.out.resolve()}")
    return 2 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
