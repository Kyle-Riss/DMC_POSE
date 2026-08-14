"""Bed YOLO-seg detection (shared by batch enrich and server)."""
from __future__ import annotations

from pathlib import Path

import cv2
import imutils
import numpy as np
from ultralytics import YOLO

MASK_THRESH = 0.35
BBOX_PAD_RATIO = 0.02
MIN_BED_AREA = 0.08
MAX_BED_AREA = 0.55
BED_SEG_CLASS = 0


def _mask_to_frame(mask, h: int, w: int) -> np.ndarray:
    m_np = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    if m_np.ndim == 3:
        m_np = m_np[0]
    if m_np.shape[0] != h or m_np.shape[1] != w:
        m_np = cv2.resize(m_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return m_np


def _refine_binary_mask(m_np: np.ndarray) -> np.ndarray | None:
    binary = (m_np > MASK_THRESH).astype(np.uint8)
    if not binary.any():
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    refined = (labels == largest).astype(np.uint8)
    area_ratio = refined.sum() / refined.size
    if area_ratio < MIN_BED_AREA or area_ratio > MAX_BED_AREA:
        return None
    return refined


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    cols = np.any(mask > 0, axis=0)
    rows = np.any(mask > 0, axis=1)
    if not cols.any() or not rows.any():
        return None
    x_min = int(np.argmax(cols))
    x_max = int(len(cols) - np.argmax(cols[::-1]) - 1)
    y_min = int(np.argmax(rows))
    y_max = int(len(rows) - np.argmax(rows[::-1]) - 1)
    return x_min, y_min, x_max, y_max


def _pad_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    pad_x = int((x_max - x_min) * BBOX_PAD_RATIO)
    pad_y = int((y_max - y_min) * BBOX_PAD_RATIO)
    return (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(w - 1, x_max + pad_x),
        min(h - 1, y_max + pad_y),
    )


def extract_bed_detection(seg_result, h: int, w: int) -> dict:
    if seg_result.boxes is None or len(seg_result.boxes) == 0:
        return {"mask": None, "bbox": None, "source": "none"}

    best_idx = int(seg_result.boxes.conf.argmax())
    refined_mask = None
    bbox = None
    source = "none"

    if seg_result.masks is not None and len(seg_result.masks) > best_idx:
        m_np = _mask_to_frame(seg_result.masks.data[best_idx], h, w)
        refined_mask = _refine_binary_mask(m_np)
        if refined_mask is not None:
            bbox = _bbox_from_mask(refined_mask)
            if bbox is not None:
                source = "mask"

    if bbox is None:
        x1, y1, x2, y2 = seg_result.boxes.xyxy[best_idx].cpu().numpy()
        bbox = _pad_bbox((int(x1), int(y1), int(x2), int(y2)), h, w)
        source = "box"

    if bbox is not None:
        bbox = _pad_bbox(bbox, h, w)

    return {"mask": refined_mask, "bbox": bbox, "source": source}


def detect_bed_from_frame(
    frame_bgr: np.ndarray,
    seg_model: YOLO,
    *,
    device: str = "0",
    seg_conf: float = 0.01,
    resize_width: int = 640,
) -> dict:
    frame = imutils.resize(frame_bgr, width=resize_width)
    h, w = frame.shape[:2]
    res = seg_model.predict(
        frame,
        classes=[BED_SEG_CLASS],
        conf=seg_conf,
        device=device,
        verbose=False,
    )[0]
    bed = extract_bed_detection(res, h, w)
    bed["frame_width"] = w
    bed["frame_height"] = h
    return bed


def detect_bed_from_video(
    video_path: Path,
    seg_model: YOLO,
    *,
    device: str = "0",
    seg_conf: float = 0.01,
    resize_width: int = 640,
    sample_frame_idx: int = 0,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read frame {sample_frame_idx} from {video_path}")
    bed = detect_bed_from_frame(
        frame, seg_model, device=device, seg_conf=seg_conf, resize_width=resize_width
    )
    bed["sample_frame_idx"] = sample_frame_idx
    bed["video_file"] = video_path.name
    return bed


def bed_to_jsonable(bed: dict) -> dict:
    mask = bed.get("mask")
    bbox = bed.get("bbox")
    return {
        "video_file": bed.get("video_file"),
        "sample_frame_idx": bed.get("sample_frame_idx"),
        "frame_width": bed.get("frame_width"),
        "frame_height": bed.get("frame_height"),
        "source": bed.get("source"),
        "bbox": list(bbox) if bbox is not None else None,
        "has_mask": mask is not None,
    }
