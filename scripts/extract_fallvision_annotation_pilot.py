#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

import cv2


def safe_name(video_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", video_id).strip("_") + ".mp4"


def inspect_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    first_ok, _ = cap.read()
    cap.release()
    return {
        "decode_ok": bool(opened and first_ok and fps > 0 and frames > 0),
        "fps": round(fps, 6) if fps else "",
        "frame_count": frames or "",
        "duration_sec": round(frames / fps, 6) if fps and frames else "",
        "width": width or "",
        "height": height or "",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=project / "external_datasets/annotations/fallvision_pilot_v1.csv",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=project / "external_datasets/fallvision/annotation_pilot_media",
    )
    parser.add_argument(
        "--unrar",
        type=Path,
        default=Path("/home/dmc/.local/dmc_pose_tools/unrar/usr/bin/unrar-nonfree"),
    )
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    args.media_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for row in rows:
        scene_dir = args.media_root / row["scene_id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        output = scene_dir / safe_name(row["video_id"])
        if not output.exists():
            part = output.with_suffix(".mp4.part")
            with part.open("wb") as handle:
                process = subprocess.run(
                    [str(args.unrar), "p", "-inul", row["raw_archive"], row["raw_member"]],
                    stdout=handle,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            if process.returncode != 0:
                failures.append({"video_id": row["video_id"], "returncode": process.returncode})
                continue
            part.replace(output)
        row["local_video_path"] = str(output.resolve())
        row["media_sha256"] = sha256(output)
        row.update(inspect_video(output))

    temp = args.annotations.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(args.annotations)
    summary = {
        "rows": len(rows),
        "decode_ok": sum(str(row.get("decode_ok")).lower() == "true" for row in rows),
        "failures": failures,
        "media_root": str(args.media_root.resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
