"""Grey-mattress heuristic → YOLO-seg polygon label."""
from __future__ import annotations

import cv2
import numpy as np

CLASS_ID = 0


def bed_polygon(img: np.ndarray) -> np.ndarray | None:
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    roi = np.zeros((h, w), np.uint8)
    roi[int(h * 0.05) : int(h * 0.95), int(w * 0.08) : int(w * 0.92)] = 255
    grey = cv2.inRange(hsv, (0, 0, 45), (180, 70, 190))
    white = cv2.inRange(hsv, (0, 0, 210), (180, 35, 255))
    mask = cv2.bitwise_and(grey, roi)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(contour) / (h * w)
    if area < 0.08 or area > 0.75:
        return None
    eps = 0.01 * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, eps, True)


def yolo_seg_line(poly: np.ndarray, w: int, h: int) -> str:
    pts = [(float(p[0][0]), float(p[0][1])) for p in poly]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = ((x_min + x_max) / 2) / w
    cy = ((y_min + y_max) / 2) / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    norm = []
    for x, y in pts:
        norm.extend([x / w, y / h])
    coords = " ".join(f"{v:.6f}" for v in norm)
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {coords}"


def draw_preview(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    vis = img.copy()
    cv2.polylines(vis, [poly], True, (0, 255, 0), 2)
    overlay = vis.copy()
    cv2.fillPoly(overlay, [poly], (0, 255, 0))
    return cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
