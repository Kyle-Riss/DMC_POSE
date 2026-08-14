#!/usr/bin/env python3
"""침대 ROI 수동 지정 — frame_ref.jpg 에서 드래그 후 bed_roi.json 저장.

조작:
  마우스 드래그: ROI 사각형
  Enter / s: 저장
  r: 초기화
  q: 종료
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "frame_ref.jpg"
OUT = ROOT / "bed_roi.json"

start_pt: tuple[int, int] | None = None
end_pt: tuple[int, int] | None = None
dragging = False


def on_mouse(event, x, y, _flags, _param):
    global start_pt, end_pt, dragging
    if event == cv2.EVENT_LBUTTONDOWN:
        start_pt = (x, y)
        end_pt = (x, y)
        dragging = True
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        end_pt = (x, y)
        dragging = False


def main() -> int:
    if not IMG.is_file():
        print(f"missing {IMG} — capture RTSP frame first")
        return 1
    img = cv2.imread(str(IMG))
    if img is None:
        return 1
    h, w = img.shape[:2]
    cv2.namedWindow("pick bed ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pick bed ROI", on_mouse)

    while True:
        vis = img.copy()
        if start_pt and end_pt:
            cv2.rectangle(vis, start_pt, end_pt, (0, 255, 255), 2)
        cv2.imshow("pick bed ROI", vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("s")) and start_pt and end_pt:
            x1, x2 = sorted([start_pt[0], end_pt[0]])
            y1, y2 = sorted([start_pt[1], end_pt[1]])
            cfg = {
                "ref_width": w,
                "ref_height": h,
                "bbox_px": [x1, y1, x2, y2],
                "bbox_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                "camera": "192.168.0.161",
                "note": "manual ROI",
            }
            OUT.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            preview = img.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.imwrite(str(ROOT / "roi_preview.jpg"), preview)
            print(f"saved {OUT}")
            break
        if key == ord("r"):
            start_pt = end_pt = None
        if key == ord("q"):
            return 0
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
