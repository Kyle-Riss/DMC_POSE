#!/usr/bin/env python3
"""Build a conservative temporal_manifest_v2 from extracted FallVision videos."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_RECORDING_RE = re.compile(r"(?:^|_)(?P<number>\d+)$")


def parse_recording_name(path: Path) -> dict:
    stem = path.stem
    lowered = stem.lower()
    anonymized = lowered.endswith("_anonymized")
    if anonymized:
        stem = stem[: -len("_anonymized")]
        lowered = stem.lower()
    resized = lowered.endswith("_resized")
    if resized:
        stem = stem[: -len("_resized")]
    match = _RECORDING_RE.search(stem)
    return {
        "recording_id": stem,
        "recording_number": int(match.group("number")) if match else None,
        "variant": (
            "resized_anonymized" if resized and anonymized
            else "anonymized" if anonymized
            else "resized" if resized
            else "base"
        ),
        "anonymized": anonymized,
        "resized": resized,
    }


def classify_path(path: Path, root: Path) -> tuple[str, str] | None:
    parts = [part.lower().replace("_", " ") for part in path.relative_to(root).parts]
    joined = "/".join(parts)
    if "nonfall bed" in joined or "no fall/bed" in joined:
        return "non_fall", "bed"
    if "nonfall chair" in joined or "no fall/chair" in joined:
        return "non_fall", "chair"
    if "nonfall stand" in joined or "no fall/stand" in joined:
        return "non_fall", "stand"
    if "fall bed" in joined or "fall/bed" in joined:
        return "fall", "bed"
    if "fall chair" in joined or "fall/chair" in joined:
        return "fall", "chair"
    if "fall stand" in joined or "fall/stand" in joined:
        return "fall", "stand"
    return None


def eligibility_fields(activity: str) -> dict:
    """Describe label availability separately from temporal training readiness."""
    return {
        "video_classification_eligible": True,
        "temporal_tcn_eligible": False,
        "subject_disjoint_split_ready": False,
        # Backward-compatible aggregate gate: temporal training remains locked.
        "training_eligible": False,
        "excluded_reasons": [
            "subject_mapping_unverified",
            *(["fall_event_intervals_missing"] if activity == "fall" else []),
        ],
    }


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    first_ok, _ = capture.read()
    last_ok = False
    if frame_count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
        last_ok, _ = capture.read()
    capture.release()
    decode_ok = bool(opened and first_ok and last_ok and fps > 0 and frame_count > 0)
    return {
        "decode_ok": decode_ok,
        "fps": round(fps, 6) if fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "duration_sec": round(frame_count / fps, 6) if fps > 0 and frame_count > 0 else None,
        "width": width if width > 0 else None,
        "height": height if height > 0 else None,
        "file_size_bytes": path.stat().st_size,
    }


def build_manifest(root: Path) -> tuple[list[dict], list[str]]:
    root = root.resolve()
    items: list[dict] = []
    errors: list[str] = []
    groups: dict[tuple[str, str, int | str], list[dict]] = defaultdict(list)

    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES):
        classification = classify_path(path, root)
        if classification is None:
            errors.append(f"unclassified_path:{path}")
            continue
        activity, scene = classification
        parsed = parse_recording_name(path)
        metadata = inspect_video(path)
        group_value: int | str = (
            parsed["recording_number"]
            if parsed["recording_number"] is not None
            else parsed["recording_id"]
        )
        item = {
            "dataset": "fallvision",
            "video_id": "fallvision_" + "_".join(
                [activity, scene, parsed["recording_id"].lower(), parsed["variant"]]
            ),
            "source_path": str(path.resolve()),
            "video_path": str(path.resolve()),
            "source_archive_group": path.relative_to(root).parts[1] if len(path.relative_to(root).parts) > 2 else None,
            "recording_id": parsed["recording_id"],
            "recording_number": parsed["recording_number"],
            "variant": parsed["variant"],
            "subject_id": None,
            "scene_id": scene,
            "camera_id": None,
            "activity_label": activity,
            "binary_fall_label": int(activity == "fall"),
            "fall_type": None,
            "bed_related": scene == "bed",
            "staged_or_real": "staged",
            "fall_start_sec": None,
            "impact_sec": None,
            "fall_end_sec": None,
            "intervals": [],
            "annotation_source": "fallvision_archive_directory",
            "annotation_scope": "video_level",
            "split": None,
            "split_group": f"fallvision_recording:{group_value}",
            "has_video": True,
            **eligibility_fields(activity),
            **metadata,
        }
        groups[(activity, scene, group_value)].append(item)
        items.append(item)

    for variants in groups.values():
        preferred = max(
            variants,
            key=lambda row: (
                row["decode_ok"],
                row["variant"] != "resized_anonymized",
                row["variant"] != "anonymized",
                row["file_size_bytes"],
            ),
        )
        for row in variants:
            row["preferred_variant"] = row is preferred
            row["duplicate_variant_count"] = len(variants)

    return items, errors


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "external_datasets" / "fallvision" / "smoke_extract",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "external_datasets" / "manifests" / "fallvision_smoke_v2.json",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    items, errors = build_manifest(args.root)
    payload = {
        "schema_version": "temporal_manifest_v2",
        "dataset": "fallvision",
        "root": str(args.root.resolve()),
        "task": "binary temporal event detection (fall transition vs non-fall)",
        "split_policy": "locked for temporal TCN: subject mapping and fall event intervals unresolved",
        "annotation_capabilities": {
            "video_level_fall_no_fall": True,
            "video_level_scene_bed_chair_stand": True,
            "frame_level_17_keypoint_csv_in_distribution": True,
            "fall_onset_impact_end": False,
            "explicit_subject_mapping": False,
        },
        "video_count": len(items),
        "decode_ok_count": sum(bool(item["decode_ok"]) for item in items),
        "preferred_variant_count": sum(bool(item["preferred_variant"]) for item in items),
        "video_classification_eligible_count": sum(bool(item["video_classification_eligible"]) for item in items),
        "temporal_tcn_eligible_count": sum(bool(item["temporal_tcn_eligible"]) for item in items),
        "subject_disjoint_split_ready_count": sum(bool(item["subject_disjoint_split_ready"]) for item in items),
        "training_eligible_count": sum(bool(item["training_eligible"]) for item in items),
        "activity_counts": dict(sorted(Counter(item["activity_label"] for item in items).items())),
        "scene_counts": dict(sorted(Counter(item["scene_id"] for item in items).items())),
        "fps_counts": dict(sorted(Counter(str(item["fps"]) for item in items).items())),
        "resolution_counts": dict(sorted(Counter(f"{item['width']}x{item['height']}" for item in items).items())),
        "errors": errors,
        "items": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "items"}, ensure_ascii=False, indent=2))
    print(f"manifest: {args.out.resolve()}")
    return 2 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
