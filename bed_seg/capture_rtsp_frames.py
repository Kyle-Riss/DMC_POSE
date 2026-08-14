#!/usr/bin/env python3
"""RTSP에서 학습/라벨용 프레임 캡처 (실시간 카메라 .161 기준)."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import imutils

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "rtsp_raw"


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture RTSP frames for bed-seg labeling")
    parser.add_argument(
        "--url",
        default="rtsp://192.168.0.161:8554/stream",
        help="RTSP URL (default: bed camera)",
    )
    parser.add_argument("--count", type=int, default=50, help="저장할 프레임 수")
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=2.0,
        help="프레임 간격(초) — 자세/조명 다양화",
    )
    parser.add_argument("--width", type=int, default=640, help="server.py 와 동일 (640→360p)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open RTSP: {args.url}")

    saved: list[str] = []
    t_next = time.time()
    while len(saved) < args.count:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.1)
            continue
        now = time.time()
        if now < t_next:
            continue
        t_next = now + args.interval_sec

        frame = imutils.resize(frame, width=args.width)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"rtsp_{len(saved):04d}_{ts}.jpg"
        path = args.out_dir / name
        cv2.imwrite(str(path), frame)
        saved.append(name)
        print(f"[{len(saved)}/{args.count}] {name}")

    cap.release()
    meta = {
        "url": args.url,
        "count": len(saved),
        "width": args.width,
        "interval_sec": args.interval_sec,
        # captured_at in UTC ISO Z format
        "captured_at": datetime.utcnow().isoformat() + 'Z',
        "files": saved,
    }
    (args.out_dir / "capture_index.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"done: {args.out_dir} ({len(saved)} frames)")
    print("next: python label_bed_polygon.py --images ... --labels ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
