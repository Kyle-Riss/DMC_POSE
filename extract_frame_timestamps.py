#!/usr/bin/env python3
"""
MP4에서 프레임 단위 timestamp(초)를 추출해 frame_timestamps.json 으로 저장.

CFR 가정: timestamp_sec = frame_id / fps
(JPG 저장 없음, 메타만)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


def extract_one(mp4: Path, out_path: Path, force: bool) -> dict:
    if out_path.exists() and not force:
        return {"path": str(mp4), "status": "skipped"}

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        return {"path": str(mp4), "status": "error", "error": "cannot open"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        cap.release()
        return {"path": str(mp4), "status": "error", "error": "invalid fps"}

    frames: list[dict] = []
    frame_id = 0
    while True:
        if not cap.grab():
            break
        t_sec = frame_id / fps
        frames.append(
            {
                "frame_id": frame_id,
                "timestamp_sec": round(t_sec, 6),
                "timestamp": f"{t_sec:.6f}s",
            }
        )
        frame_id += 1

    cap.release()
    actual = len(frames)
    duration_sec = actual / fps if actual else 0.0

    payload = {
        "video_file": mp4.name,
        "source_path": str(mp4.resolve()),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count_reported": reported,
        "frame_count": actual,
        "duration_sec": round(duration_sec, 6),
        "time_base": "seconds",
        "method": "cfr_frame_index_over_fps",
        "frames": frames,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    status = "ok"
    if reported > 0 and reported != actual:
        status = "ok_count_mismatch"

    return {
        "path": str(mp4),
        "status": status,
        "frame_count": actual,
        "duration_sec": duration_sec,
        "out": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MP4 프레임 단위 timestamp 추출")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dmc/Dataset"),
        help="mp4 검색 루트 (기본: /home/dmc/Dataset)",
    )
    parser.add_argument(
        "--out-name",
        default="{stem}_frame_timestamps.json",
        help="출력 파일명 패턴 ({stem}=mp4 파일명 sans ext)",
    )
    parser.add_argument("--force", action="store_true", help="기존 JSON 덮어쓰기")
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="전체 요약 index JSON 경로 (기본: <root>/frame_timestamps_index.json)",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"root not found: {root}", file=sys.stderr)
        return 1

    index_path = args.index or (root / "frame_timestamps_index.json")
    mp4_list = sorted(root.rglob("*.mp4"))
    if not mp4_list:
        print(f"no mp4 under {root}", file=sys.stderr)
        return 1

    t0 = time.time()
    results: list[dict] = []
    total_frames = 0

    for i, mp4 in enumerate(mp4_list, 1):
        out_name = args.out_name.format(stem=mp4.stem)
        out_path = mp4.with_name(out_name)
        rel = mp4.relative_to(root)
        print(f"[{i}/{len(mp4_list)}] {rel}", flush=True)
        row = extract_one(mp4, out_path, args.force)
        results.append(row)
        if row.get("frame_count"):
            total_frames += row["frame_count"]

    summary = {
        "root": str(root),
        "video_count": len(mp4_list),
        "total_frames": total_frames,
        "elapsed_sec": round(time.time() - t0, 2),
        "method": "cfr_frame_index_over_fps",
        "out_name_pattern": args.out_name,
        "videos": results,
    }
    index_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok = sum(1 for r in results if r.get("status", "").startswith("ok"))
    err = sum(1 for r in results if r.get("status") == "error")
    skip = sum(1 for r in results if r.get("status") == "skipped")
    print(
        f"\nDone: ok={ok} skipped={skip} error={err} "
        f"frames={total_frames} index={index_path}",
        flush=True,
    )
    return 0 if err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
