"""Approximate bed zone — dilated seg mask + ROI/bbox fallback for unstable seg."""
from __future__ import annotations

import cv2
import numpy as np

from bed_roi.roi_utils import bbox_area, intersect_bbox


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or mask is None or not mask.any():
        return mask
    k = px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    x1 = int(np.argmax(cols))
    x2 = int(len(cols) - np.argmax(cols[::-1]) - 1)
    y1 = int(np.argmax(rows))
    y2 = int(len(rows) - np.argmax(rows[::-1]) - 1)
    return x1, y1, x2, y2


def _bbox_to_mask(bbox: tuple[int, int, int, int], h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    m = np.zeros((h, w), dtype=np.uint8)
    m[max(0, y1) : min(h, y2 + 1), max(0, x1) : min(w, x2 + 1)] = 1
    return m


def build_approx_bed_zone(
    bed: dict,
    roi_bbox: tuple[int, int, int, int] | None,
    h: int,
    w: int,
    preset: dict,
) -> dict:
    """
    Imperfect seg → usable zone for attachment/overflow.

    - dilate YOLO mask (approximate boundary)
    - weak/missing mask → seg bbox or fixed ROI
  - zone_quality: mask_ok | mask_weak | bbox_only | roi_only | none
    """
    cfg = preset.get("bed_zone", {})
    dilate_px = int(cfg.get("mask_dilate_px", 14))
    min_mask_area = float(cfg.get("min_mask_area_ratio", 0.04))
    max_mask_area = float(cfg.get("max_mask_area_ratio", 0.62))
    shrink_to_roi = bool(cfg.get("clip_mask_to_roi", True))

    raw_mask = bed.get("raw_mask", bed.get("mask"))
    if raw_mask is not None and raw_mask.dtype != np.uint8:
        raw_mask = (raw_mask > 0).astype(np.uint8)

    seg_bbox = bed.get("bbox")
    base_source = str(bed.get("source", "none"))
    frame_area = max(h * w, 1)

    zone_mask: np.ndarray | None = None
    zone_bbox: tuple[int, int, int, int] | None = None
    zone_quality = "none"

    if raw_mask is not None and raw_mask.any():
        area_r = float(raw_mask.sum()) / frame_area
        if area_r < min_mask_area or area_r > max_mask_area:
            zone_quality = "mask_weak"
        else:
            zone_quality = "mask_ok"
        zone_mask = _dilate_mask(raw_mask, dilate_px)
        if shrink_to_roi and roi_bbox is not None:
            roi_m = _bbox_to_mask(roi_bbox, h, w)
            zone_mask = zone_mask * roi_m
        zone_bbox = _mask_to_bbox(zone_mask) or seg_bbox

    if zone_mask is None or not zone_mask.any():
        if seg_bbox is not None:
            zone_quality = "bbox_only"
            zone_bbox = seg_bbox
            if roi_bbox is not None:
                clipped = intersect_bbox(seg_bbox, roi_bbox)
                if clipped is not None:
                    zone_bbox = clipped
            zone_mask = _dilate_mask(_bbox_to_mask(zone_bbox, h, w), max(4, dilate_px // 2))
        elif roi_bbox is not None:
            zone_quality = "roi_only"
            zone_bbox = roi_bbox
            zone_mask = _bbox_to_mask(roi_bbox, h, w)

    if zone_bbox is None and zone_mask is not None:
        zone_bbox = _mask_to_bbox(zone_mask)

    return {
        **bed,
        "mask": zone_mask,
        "raw_mask": raw_mask,
        "bbox": zone_bbox,
        "zone_quality": zone_quality,
        "zone_built": True,
        "source": f"{base_source}|{zone_quality}",
    }
