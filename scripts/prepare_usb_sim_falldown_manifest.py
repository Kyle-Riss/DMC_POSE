#!/usr/bin/env python3
"""Inventory the local USB simulated-fall videos without inventing intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"readable": False}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    container_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    decoded_frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded_frames += 1
    capture.release()
    return {
        "readable": fps > 0.0 and decoded_frames > 0,
        "fps": round(fps, 6),
        "frame_count": decoded_frames,
        "container_frame_count": container_frames,
        "frame_count_matches_container": decoded_frames == container_frames,
        "duration_sec": round(decoded_frames / fps, 6) if fps > 0.0 else None,
        "width": width,
        "height": height,
    }


def build_manifest(root: Path, *, include_sha256: bool = False) -> dict:
    items = []
    errors = []
    warnings = []
    for path in sorted(root.glob("*/*.mp4")):
        camera = path.parent.name
        meta = video_metadata(path)
        if not meta.get("readable"):
            errors.append(f"unreadable video: {path}")
        elif not meta.get("frame_count_matches_container", True):
            warnings.append(
                f"frame count mismatch: {path}: "
                f"container={meta.get('container_frame_count')} decoded={meta.get('frame_count')}"
            )
        item = {
            "video_id": f"usb_sim_falldown_{camera}_{path.stem}",
            "dataset": "usb_sim_falldown",
            "subject_id": None,
            "split": "diagnostic",
            "camera_id": camera,
            "video_path": str(path.resolve()),
            "source_path": str(path.resolve()),
            "activity_label": "simulated_fall_video_level",
            "binary_fall_label": 1,
            "staged_or_real": "staged",
            "annotation_scope": "video_only",
            "intervals": [],
            "training_eligible": False,
            "temporal_tcn_eligible": False,
            "diagnostic_eligible": bool(meta.get("readable")),
            "excluded_reasons": ["fall_interval_not_annotated", "subject_identity_unknown"],
            **meta,
        }
        if include_sha256 and meta.get("readable"):
            item["media_sha256"] = file_sha256(path)
        items.append(item)
    return {
        "schema_version": "temporal_manifest_v2",
        "dataset": "usb_sim_falldown",
        "root": str(root.resolve()),
        "sample_hz_target": 20.0,
        "split_policy": "diagnostic_only_no_training_split",
        "annotation_warning": "Video-level fall label does not define fall onset/end; no row/window target may be generated.",
        "video_count": len(items),
        "camera_counts": dict(sorted(Counter(item["camera_id"] for item in items).items())),
        "duration_total_sec": round(sum(float(item.get("duration_sec") or 0.0) for item in items), 6),
        "errors": errors,
        "warnings": warnings,
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=Path("/media/dmc/8GB/sim_falldown"))
    parser.add_argument("--out", type=Path, default=project_root / "external_datasets/manifests/usb_sim_falldown_diagnostic.json")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    payload = build_manifest(args.root, include_sha256=args.sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "video_count", "camera_counts", "duration_total_sec", "errors", "warnings"
    )}, ensure_ascii=False, indent=2))
    print(f"manifest: {args.out.resolve()}")
    return 2 if args.strict and payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
