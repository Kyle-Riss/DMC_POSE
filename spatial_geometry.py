"""Spatial geometry shared by automatic bed ROI and person tracking."""

from __future__ import annotations

import cv2
import numpy as np


def orient_image(image: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate an image into the analysis coordinate system."""
    rotation = int(rotation) % 360
    if rotation == 0:
        return image.copy()
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rotation must be one of 0, 90, 180, 270")


def mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def orient_bed_detection(
    bed: dict,
    rotation: int,
    source_h: int,
    source_w: int,
) -> dict:
    """Rotate a raw-camera bed detection into analysis coordinates.

    Segmentation is intentionally performed in the camera/model's native
    orientation. Pose may use a rotated analysis frame, so both mask and bbox
    must be transformed before they are consumed by the ROI manager.
    """
    oriented = dict(bed or {})
    mask = oriented.get("mask")
    if mask is None:
        bbox = oriented.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            mask = np.zeros((int(source_h), int(source_w)), dtype=np.uint8)
            mask[
                max(0, y1):min(int(source_h), y2 + 1),
                max(0, x1):min(int(source_w), x2 + 1),
            ] = 1
    if mask is None:
        oriented["bbox"] = None
        return oriented

    rotated = orient_image((mask > 0).astype(np.uint8), rotation)
    oriented["mask"] = rotated
    oriented["bbox"] = mask_bbox(rotated)
    return oriented


def refined_bed_from_mask(
    mask: np.ndarray | None,
    coarse_bbox: tuple[int, int, int, int] | None,
    confidence: float,
    frame_h: int,
    frame_w: int,
    *,
    min_area_ratio: float = 0.04,
    max_area_ratio: float = 0.65,
    min_extent_ratio: float = 0.40,
    prompt_point: tuple[float, float] | None = None,
) -> dict | None:
    """Validate a prompt-refined bed mask before it can become an ROI.

    The coarse detector supplies only a positive point. A refined result must
    contain that point and occupy a plausible fraction of the frame. Returning
    ``None`` deliberately blocks the spatial path instead of falling back to a
    known-overbroad coarse mask.
    """
    if mask is None or coarse_bbox is None or mask.size == 0:
        return None
    if mask.shape[:2] != (frame_h, frame_w):
        mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8)
    bbox = mask_bbox(binary)
    if bbox is None:
        return None
    area_ratio = float(np.count_nonzero(binary)) / float(max(1, frame_h * frame_w))
    if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
        return None
    x1, y1, x2, y2 = (int(v) for v in coarse_bbox)
    coarse_extent = max(1, max(abs(x2 - x1), abs(y2 - y1)))
    refined_extent = max(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))
    if refined_extent / float(coarse_extent) < float(min_extent_ratio):
        return None
    if prompt_point is None:
        prompt_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    cx = int(np.clip(round(prompt_point[0]), 0, frame_w - 1))
    cy = int(np.clip(round(prompt_point[1]), 0, frame_h - 1))
    if binary[cy, cx] == 0:
        return None
    return {
        "mask": binary,
        "bbox": bbox,
        "confidence": float(confidence),
        "source": "mobile_sam_multipoint_refined",
    }


def select_refined_bed_candidate(
    masks: list[np.ndarray],
    prompt_points: list[tuple[float, float]],
    coarse_bbox: tuple[int, int, int, int] | None,
    confidence: float,
    frame_h: int,
    frame_w: int,
    *,
    min_area_ratio: float = 0.04,
    max_area_ratio: float = 0.65,
    min_extent_ratio: float = 0.40,
) -> dict | None:
    """Choose the largest valid mattress candidate from one SAM batch."""
    valid: list[dict] = []
    for mask, point in zip(masks, prompt_points):
        candidate = refined_bed_from_mask(
            mask, coarse_bbox, confidence, frame_h, frame_w,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            min_extent_ratio=min_extent_ratio,
            prompt_point=point,
        )
        if candidate is not None:
            valid.append(candidate)
    if not valid:
        return None
    return max(valid, key=lambda item: int(np.count_nonzero(item["mask"])))


def skeleton_bed_coverage(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    bed: dict,
    frame_h: int,
    frame_w: int,
    min_kpt_conf: float = 0.3,
) -> float:
    """Return the fraction of observed skeleton joints on the bed mask.

    A keypoint bounding rectangle is not a body mask: for a standing person
    beside a bed it includes empty image area between limbs. Sampling observed
    skeleton joints avoids counting that space. Bbox intersection remains only
    as a fallback for segmentation results that genuinely have no mask.
    """
    if kpts_xy is None or len(kpts_xy) == 0:
        return 0.0
    pts = np.asarray(kpts_xy, dtype=np.float32)
    valid = np.isfinite(pts).all(axis=1)
    valid &= ~((pts[:, 0] < 1.0) & (pts[:, 1] < 1.0))
    if kpts_conf is not None and len(kpts_conf) == len(pts):
        valid &= np.asarray(kpts_conf) >= float(min_kpt_conf)
    pts = pts[valid]
    if len(pts) == 0:
        return 0.0

    bed_mask = bed.get("mask")
    if bed_mask is not None:
        if bed_mask.shape[:2] != (frame_h, frame_w):
            bed_mask = cv2.resize(
                bed_mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
            )
        xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, frame_w - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, frame_h - 1)
        return float(np.count_nonzero(bed_mask[ys, xs] > 0)) / float(len(pts))

    bed_bbox = bed.get("bbox")
    if bed_bbox is None:
        return 0.0
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    bx1, by1, bx2, by2 = (float(v) for v in bed_bbox)
    ix1, iy1 = max(x1, bx1), max(y1, by1)
    ix2, iy2 = min(x2, bx2), min(y2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1)) / float((x2 - x1) * (y2 - y1))
