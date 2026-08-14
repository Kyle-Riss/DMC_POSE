#!/usr/bin/env python3
"""
Capture DOWN/UP reference frames for one rail side (no person),
then create an absdiff map and ROI suggestion.

Restored from pose-sixclass-viewer/capture_rail_pair.py (2026-05-11).

Usage:
  python3 capture_rail_pair.py --side left
  python3 capture_rail_pair.py --side right --rtsp rtsp://...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

RAIL_DIR = Path(__file__).resolve().parent


def grab_one(rtsp_url: str) -> tuple[bool, np.ndarray | None]:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = cap.read()
    cap.release()
    return ok, frame


def suggest_roi(up_bgr: np.ndarray, down_bgr: np.ndarray, side: str) -> tuple[dict, np.ndarray, float]:
    h, w = up_bgr.shape[:2]
    up = cv2.GaussianBlur(cv2.cvtColor(up_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    dn = cv2.GaussianBlur(cv2.cvtColor(down_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    diff = cv2.absdiff(up, dn)

    y_min = int(h * 0.06)
    y_max = int(h * 0.82)
    if side == "left":
        x_min, x_max = 0, int(w * 0.55)
    else:
        x_min, x_max = int(w * 0.45), w

    win_w = int(w * 0.24)
    win_h = int(h * 0.16)
    step_x = max(4, int(w * 0.02))
    step_y = max(4, int(h * 0.02))

    best_score = -1.0
    best_box = (x_min, y_min, min(w, x_min + win_w), min(h, y_min + win_h))

    for y0 in range(y_min, max(y_min + 1, y_max - win_h), step_y):
        for x0 in range(x_min, max(x_min + 1, x_max - win_w), step_x):
            patch = diff[y0 : y0 + win_h, x0 : x0 + win_w]
            score = float(patch.mean())
            if score > best_score:
                best_score = score
                best_box = (x0, y0, x0 + win_w, y0 + win_h)

    crop = diff[y_min:y_max, x_min:x_max]
    p = float(np.percentile(crop, 98))
    hot = (crop >= p).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)

    comp_best_score = -1.0
    comp_best_box = None
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < 20:
            continue
        mask = labels == i
        mean_val = float(crop[mask].mean())
        score = mean_val * (area ** 0.5)
        if score > comp_best_score:
            comp_best_score = score
            comp_best_box = (
                int(x + x_min),
                int(y + y_min),
                int(x + ww + x_min),
                int(y + hh + y_min),
            )

    x0, y0, x1, y1 = best_box
    if comp_best_box is not None:
        cx0, cy0, cx1, cy1 = comp_best_box
        comp_w = max(12, cx1 - cx0)
        comp_h = max(12, cy1 - cy0)
        cx = (cx0 + cx1) // 2
        cy = (cy0 + cy1) // 2
        ww = min(w, max(win_w, int(comp_w * 2.2)))
        hh = min(h, max(win_h, int(comp_h * 2.6)))
        x0 = max(0, min(w - ww, cx - ww // 2))
        y0 = max(0, min(h - hh, cy - hh // 2))
        x1 = x0 + ww
        y1 = y0 + hh

    roi = {
        "x0": round(x0 / w, 4),
        "x1": round(x1 / w, 4),
        "y0": round(y0 / h, 4),
        "y1": round(y1 / h, 4),
    }

    vis = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.putText(
        vis,
        f"{side} mean-win={best_score:.2f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    if comp_best_box is not None:
        cx0, cy0, cx1, cy1 = comp_best_box
        cv2.rectangle(vis, (cx0, cy0), (cx1, cy1), (255, 0, 255), 1)
        cv2.putText(
            vis,
            "magenta=hot component",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )

    return roi, vis, best_score


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture rail UP/DOWN pair and suggest ROI")
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--rtsp", default="rtsp://192.168.0.161:8554/stream")
    ap.add_argument("--out-dir", type=Path, default=RAIL_DIR / "reference")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] {args.side.upper()} rail DOWN 상태로 만들고 Enter")
    input()
    ok, down = grab_one(args.rtsp)
    if not ok or down is None:
        raise SystemExit("failed to capture DOWN frame")
    down_path = out_dir / f"{args.side}_down.jpg"
    cv2.imwrite(str(down_path), down)
    print("saved", down_path)

    print(f"[2/2] {args.side.upper()} rail UP 상태로 만들고 Enter")
    input()
    ok, up = grab_one(args.rtsp)
    if not ok or up is None:
        raise SystemExit("failed to capture UP frame")
    up_path = out_dir / f"{args.side}_up.jpg"
    cv2.imwrite(str(up_path), up)
    print("saved", up_path)

    roi, vis, score = suggest_roi(up, down, args.side)
    map_path = out_dir / f"{args.side}_up_down_absdiff_map.jpg"
    cv2.imwrite(str(map_path), vis)
    print("saved", map_path)
    print("suggested roi:", roi)
    print("max mean diff:", round(score, 3))


if __name__ == "__main__":
    main()
