"""Fixed bed ROI loader — clip seg/bbox to camera-specific zone."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_ROI_PATH = Path(__file__).resolve().parent / "bed_roi.json"


def load_bed_roi(path: Path | None, w: int, h: int) -> tuple[int, int, int, int] | None:
    roi_path = path or DEFAULT_ROI_PATH
    if not roi_path.is_file():
        return None
    data = json.loads(roi_path.read_text(encoding="utf-8"))
    if "bbox_norm" in data:
        x1, y1, x2, y2 = data["bbox_norm"]
        return (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )
    if "bbox_px" in data:
        ref_w = data.get("ref_width", w)
        ref_h = data.get("ref_height", h)
        x1, y1, x2, y2 = data["bbox_px"]
        sx, sy = w / ref_w, h / ref_h
        return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
    return None


def intersect_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_area(b: tuple[int, int, int, int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def apply_bed_roi(bed: dict, roi_bbox: tuple[int, int, int, int] | None, h: int, w: int) -> dict:
    """Clip seg to ROI; use ROI bbox when seg is missing or too wide."""
    if roi_bbox is None:
        return bed

    x1, y1, x2, y2 = roi_bbox
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y1 : y2 + 1, x1 : x2 + 1] = 1

    mask = bed.get("mask")
    if mask is not None:
        clipped = (mask > 0).astype(np.uint8) * roi_mask
        if clipped.any():
            bed = {**bed, "mask": clipped, "source": f"{bed['source']}+roi"}
        else:
            bed = {**bed, "mask": None}

    seg_bbox = bed.get("bbox")
    roi_area = bbox_area(roi_bbox)
    use_roi_bbox = seg_bbox is None
    if seg_bbox is not None:
        clipped_bbox = intersect_bbox(seg_bbox, roi_bbox)
        seg_area = bbox_area(seg_bbox)
        if clipped_bbox is None or seg_area > roi_area * 1.08:
            use_roi_bbox = True
        else:
            bed = {**bed, "bbox": clipped_bbox}

    # clipped mask가 있으면 mask 기준 bbox가 ROI 박스보다 타이트할 때 우선
    mask = bed.get("mask")
    if mask is not None and mask.any():
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        if rows.any() and cols.any():
            mx1 = int(np.argmax(cols))
            mx2 = int(len(cols) - np.argmax(cols[::-1]) - 1)
            my1 = int(np.argmax(rows))
            my2 = int(len(rows) - np.argmax(rows[::-1]) - 1)
            mask_bbox = (mx1, my1, mx2, my2)
            if not use_roi_bbox or bbox_area(mask_bbox) < bbox_area(bed.get("bbox") or roi_bbox):
                bed = {**bed, "bbox": mask_bbox}
                use_roi_bbox = False

    if use_roi_bbox:
        bed = {**bed, "bbox": roi_bbox, "source": "roi" if bed.get("source") == "none" else f"{bed['source']}->roi"}

    return bed
