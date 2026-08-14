#!/usr/bin/env python3
"""침대 seg + zone + pose + 6-class 데모 이미지/영상 프레임 저장."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import imutils
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.config.set_visible_devices([], "GPU")

BASE = Path(__file__).resolve().parent
CLASS_NAMES = [
    "정면_누움",
    "엎드림_등",
    "옆누움_가까움",
    "옆누움_멀음",
    "앉음_중앙",
    "앉음_가장자리",
]
RISK_KEYPOINTS = {"L_wrist": 9, "R_wrist": 10, "L_ankle": 15, "R_ankle": 16}
RISK_THRESHOLDS = {"LOW": 0.05, "MED": 0.15, "HIGH": 0.25}
RISK_COLORS = {
    "SAFE": (0, 255, 0),
    "LOW": (0, 255, 255),
    "MED": (0, 165, 255),
    "HIGH": (0, 0, 255),
}


def mask_to_numpy(mask, h: int, w: int) -> np.ndarray:
    m = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return m


def get_bed_bbox(bed_masks, h: int, w: int):
    if bed_masks is None:
        return None
    combined = None
    for mask in bed_masks:
        m_np = mask_to_numpy(mask, h, w)
        combined = m_np if combined is None else np.maximum(combined, m_np)
    if combined is None:
        return None
    cols = np.any(combined > 0.5, axis=0)
    rows = np.any(combined > 0.5, axis=1)
    if not cols.any():
        return None
    x_min = int(np.argmax(cols))
    x_max = int(len(cols) - np.argmax(cols[::-1]) - 1)
    y_min = int(np.argmax(rows))
    y_max = int(len(rows) - np.argmax(rows[::-1]) - 1)
    return x_min, y_min, x_max, y_max


def draw_bed_mask(frame: np.ndarray, bed_masks) -> np.ndarray:
    if bed_masks is None:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for mask in bed_masks:
        m_np = mask_to_numpy(mask, h, w)
        color_layer = np.zeros_like(frame)
        color_layer[m_np > 0.5] = (60, 200, 120)
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)
    return overlay


def draw_bed_zones(frame: np.ndarray, bed_bbox) -> np.ndarray:
    if bed_bbox is None:
        return frame
    x_min, y_min, x_max, y_max = bed_bbox
    w = x_max - x_min
    x1_3 = x_min + w // 3
    x2_3 = x_min + 2 * w // 3
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (220, 220, 220), 2)
    cv2.line(frame, (x1_3, y_min), (x1_3, y_max), (220, 220, 220), 1)
    cv2.line(frame, (x2_3, y_min), (x2_3, y_max), (220, 220, 220), 1)
    cy = (y_min + y_max) // 2
    for label, x in (("L", x_min + 8), ("C", x1_3 + 8), ("R", x2_3 + 8)):
        cv2.putText(frame, label, (x, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def is_person_in_bed(person_bbox, bed_masks, h: int, w: int) -> bool:
    if bed_masks is None:
        return False
    x1, y1, x2, y2 = person_bbox[:4]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    for mask in bed_masks:
        m_np = mask_to_numpy(mask, h, w)
        if 0 <= cy < m_np.shape[0] and 0 <= cx < m_np.shape[1] and m_np[cy, cx] > 0.5:
            return True
    return False


def calc_fall_risk(kpts_xy, kpts_conf, bed_bbox):
    if bed_bbox is None or kpts_xy.shape[0] < 17:
        return "SAFE", 0.0, []
    x_min, _, x_max, _ = bed_bbox
    bed_width = x_max - x_min
    if bed_width <= 0:
        return "SAFE", 0.0, []
    max_overflow = 0.0
    danger_pts = []
    for name, idx in RISK_KEYPOINTS.items():
        if kpts_conf[idx] < 0.3:
            continue
        x, y = kpts_xy[idx]
        overflow = 0.0
        if x < x_min:
            overflow = (x_min - x) / bed_width
        elif x > x_max:
            overflow = (x - x_max) / bed_width
        if overflow > 0:
            danger_pts.append((name, int(x), int(y), overflow))
            max_overflow = max(max_overflow, overflow)
    if max_overflow >= RISK_THRESHOLDS["HIGH"]:
        level = "HIGH"
    elif max_overflow >= RISK_THRESHOLDS["MED"]:
        level = "MED"
    elif max_overflow >= RISK_THRESHOLDS["LOW"]:
        level = "LOW"
    else:
        level = "SAFE"
    return level, max_overflow, danger_pts


def draw_hud(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 28 + 34 * len(lines)), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    y = 28
    for text, color in lines:
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        y += 34
    return out


def process_frame(frame: np.ndarray, seg_model, pose_model, clf_model, device: str) -> tuple[np.ndarray, dict]:
    h, w = frame.shape[:2]
    seg_res = seg_model.predict(frame, classes=[59], conf=0.05, device=device, verbose=False)
    bed_masks = seg_res[0].masks.data if seg_res[0].masks is not None else None
    bed_bbox = get_bed_bbox(bed_masks, h, w)

    out = draw_bed_mask(frame, bed_masks)
    out = draw_bed_zones(out, bed_bbox)

    pose_res = pose_model.predict(frame, conf=0.5, device=device, verbose=False)
    in_bed = "NO"
    pose_label = "None"
    pose_conf = 0.0
    risk_level = "SAFE"
    max_overflow = 0.0

    if len(pose_res) > 0 and len(pose_res[0].keypoints) > 0:
        out = pose_res[0].plot(img=out)
        out = draw_bed_zones(out, bed_bbox)
        person_bbox = pose_res[0].boxes.xyxy[0].cpu().numpy()
        in_bed = "YES" if is_person_in_bed(person_bbox, bed_masks, h, w) else "NO"

        kpts_xy = pose_res[0].keypoints[0].xy.cpu().numpy()
        kpts_conf = pose_res[0].keypoints[0].conf.cpu().numpy()
        if kpts_xy.ndim == 3:
            kpts_xy = kpts_xy[0]
        if kpts_conf.ndim > 1:
            kpts_conf = kpts_conf.reshape(-1)
        risk_level, max_overflow, danger_pts = calc_fall_risk(kpts_xy, kpts_conf, bed_bbox)
        for name, x, y, overflow in danger_pts:
            color = RISK_COLORS[risk_level]
            cv2.circle(out, (x, y), 10, color, -1)
            cv2.putText(out, f"{name}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        flat = kpts_xy.reshape(-1)
        if len(flat) == 34:
            pred = clf_model.predict(flat.reshape(1, -1).astype("float32"), verbose=0)[0]
            idx = int(np.argmax(pred))
            pose_label = CLASS_NAMES[idx]
            pose_conf = float(pred[idx])

    out = draw_hud(
        out,
        [
            (f"In Bed: {in_bed}", (0, 255, 255)),
            (f"Pose: {pose_label} ({pose_conf:.2f})", (0, 255, 0)),
            (f"Risk: {risk_level} ({max_overflow * 100:.1f}%)", RISK_COLORS[risk_level]),
            ("Bed seg mask + L/C/R zones + pose", (255, 255, 255)),
        ],
    )
    meta = {
        "in_bed": in_bed,
        "pose": pose_label,
        "pose_conf": round(pose_conf, 4),
        "risk": risk_level,
        "max_overflow": round(max_overflow, 4),
        "bed_detected": bed_bbox is not None,
    }
    return out, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=BASE / "runs" / "seg_demo")
    parser.add_argument("--device", default="0")
    parser.add_argument("--frame-sec", type=float, default=30.0, help="영상일 때 추출할 초")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seg_model = YOLO(str(BASE / "yolo11n-seg.pt"))
    pose_model = YOLO(str(BASE / "yolo11m-pose.pt"))
    clf_model = tf.keras.models.load_model(str(BASE / "my_model_six.keras"))

    src = args.input
    if src.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = cv2.imread(str(src))
        if frame is None:
            raise SystemExit(f"cannot read image: {src}")
        frame = imutils.resize(frame, width=640)
        out, meta = process_frame(frame, seg_model, pose_model, clf_model, args.device)
        out_path = args.out_dir / f"{src.stem}_overlay.jpg"
        cv2.imwrite(str(out_path), out)
        meta_path = args.out_dir / f"{src.stem}_overlay.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out_path)
        print(meta_path)
        return 0

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read video frame")
    frame = imutils.resize(frame, width=640)
    out, meta = process_frame(frame, seg_model, pose_model, clf_model, args.device)
    out_path = args.out_dir / f"{src.stem}_{int(args.frame_sec)}s_overlay.jpg"
    cv2.imwrite(str(out_path), out)
    print(out_path)
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
