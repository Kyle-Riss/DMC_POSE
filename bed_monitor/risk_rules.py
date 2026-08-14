from __future__ import annotations

import numpy as np

RISK_KEYPOINTS = {
    "L_wrist": 9,
    "R_wrist": 10,
    "L_ankle": 15,
    "R_ankle": 16,
}

TORSO_KPTS = (5, 6, 11, 12)


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the smaller 2-D angle ABC, or NaN for a degenerate limb."""
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-6:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _is_seated_skeleton(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    *,
    conf_threshold: float,
    max_median_angle_deg: float = 130.0,
) -> bool:
    """Recognize a bent hip/knee chain without assuming image gravity axis."""
    angle_specs = (
        (5, 11, 13), (6, 12, 14),
        (11, 13, 15), (12, 14, 16),
    )
    angles: list[float] = []
    for a, b, c in angle_specs:
        if min(kpts_conf[a], kpts_conf[b], kpts_conf[c]) < conf_threshold:
            continue
        angle = _joint_angle_deg(kpts_xy[a], kpts_xy[b], kpts_xy[c])
        if np.isfinite(angle):
            angles.append(angle)
    return len(angles) >= 2 and float(np.median(angles)) <= max_median_angle_deg


def _point_near_bed(
    center: tuple[float, float],
    bed_mask: np.ndarray | None,
    bed_bbox: tuple[int, int, int, int] | None,
    *,
    max_distance_ratio: float,
) -> bool:
    """Return whether a point is on/just outside the mattress boundary."""
    if bed_bbox is None:
        return False
    bx1, by1, bx2, by2 = (int(v) for v in bed_bbox)
    short_extent = max(1, min(abs(bx2 - bx1), abs(by2 - by1)))
    radius = max(3, int(round(short_extent * max_distance_ratio)))
    cx, cy = (int(round(center[0])), int(round(center[1])))
    if bed_mask is not None:
        h, w = bed_mask.shape[:2]
        x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
        y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
        return x1 < x2 and y1 < y2 and bool(np.any(bed_mask[y1:y2, x1:x2] > 0))
    dx = max(bx1 - cx, 0, cx - bx2)
    dy = max(by1 - cy, 0, cy - by2)
    return dx <= radius and dy <= radius


def skeleton_person_detected(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    *,
    conf_threshold: float = 0.3,
    min_core: int = 2,
    min_total: int = 5,
) -> bool:
    """YOLO box가 아니라 유효 keypoint로 사람 존재 판정."""
    total_ok, core_ok = count_skeleton_keypoints(
        kpts_xy, kpts_conf, conf_threshold=conf_threshold
    )
    return core_ok >= min_core and total_ok >= min_total


def count_skeleton_keypoints(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    *,
    conf_threshold: float = 0.3,
) -> tuple[int, int]:
    """Returns (total_valid, core_valid) for HUD / diagnostics."""
    if kpts_xy.shape[0] < 17:
        return 0, 0
    core_ok = 0
    total_ok = 0
    for i in range(17):
        if kpts_conf[i] < conf_threshold:
            continue
        x, y = kpts_xy[i]
        if np.isnan(x) or np.isnan(y) or (x < 1 and y < 1):
            continue
        total_ok += 1
        if i in TORSO_KPTS:
            core_ok += 1
    return total_ok, core_ok


def _kpt_inside_bed(
    x: float,
    y: float,
    bed_mask: np.ndarray | None,
    bed_bbox: tuple[int, int, int, int] | None,
) -> bool:
    if bed_mask is not None:
        h, w = bed_mask.shape[:2]
        ix, iy = int(x), int(y)
        if 0 <= iy < h and 0 <= ix < w:
            return bool(bed_mask[iy, ix] > 0)
        return False
    if bed_bbox is not None:
        bx1, by1, bx2, by2 = bed_bbox
        return bx1 <= x <= bx2 and by1 <= y <= by2
    return False


def classify_seg_attachment(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    center: tuple[float, float] | None,
    bed_mask: np.ndarray | None,
    bed_bbox: tuple[int, int, int, int] | None,
    *,
    conf_threshold: float = 0.3,
    attach_ratio_min: float = 0.35,
) -> tuple[str, float, bool]:
    """
    skeleton vs bed seg.
    Returns: on_seg | partial | off_seg | unknown, kpt_on_seg_ratio, limbs_outside_seg
    """
    inside = 0
    total = 0
    for i in range(min(17, len(kpts_xy))):
        if kpts_conf[i] < conf_threshold:
            continue
        x, y = kpts_xy[i]
        if np.isnan(x) or np.isnan(y) or (x < 1 and y < 1):
            continue
        total += 1
        if _kpt_inside_bed(x, y, bed_mask, bed_bbox):
            inside += 1
    if total == 0 or center is None:
        return "unknown", 0.0, False

    ratio = inside / total
    hip_on_seg, _ = person_in_bed(center, bed_bbox, bed_mask)

    limbs_out = False
    for idx in RISK_KEYPOINTS.values():
        if kpts_conf[idx] < conf_threshold:
            continue
        x, y = kpts_xy[idx]
        if np.isnan(x) or np.isnan(y):
            continue
        if not _kpt_inside_bed(x, y, bed_mask, bed_bbox):
            limbs_out = True
            break

    on_seg_min = 0.65
    if not hip_on_seg or ratio < attach_ratio_min:
        return "off_seg", ratio, limbs_out
    if limbs_out or ratio < on_seg_min:
        return "partial", ratio, limbs_out
    return "on_seg", ratio, limbs_out


def classify_seg_attachment_with_preset(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    center: tuple[float, float] | None,
    bed: dict,
    preset: dict,
    *,
    conf_threshold: float = 0.3,
) -> tuple[str, float, bool]:
    ev = preset.get("events", {})
    attach_min = float(ev.get("attach_ratio_min", 0.35))
    on_seg_min = float(ev.get("on_seg_ratio_min", 0.65))
    edge_distance_ratio = float(ev.get("edge_contact_max_ratio", 0.10))
    bed_mask = bed.get("mask")
    bed_bbox = bed.get("bbox")

    inside = 0
    total = 0
    for i in range(min(17, len(kpts_xy))):
        if kpts_conf[i] < conf_threshold:
            continue
        x, y = kpts_xy[i]
        if np.isnan(x) or np.isnan(y) or (x < 1 and y < 1):
            continue
        total += 1
        if _kpt_inside_bed(x, y, bed_mask, bed_bbox):
            inside += 1
    if total == 0 or center is None:
        return "unknown", 0.0, False

    ratio = inside / total
    hip_on_seg, _ = person_in_bed(center, bed_bbox, bed_mask)

    limbs_out = False
    for idx in RISK_KEYPOINTS.values():
        if kpts_conf[idx] < conf_threshold:
            continue
        x, y = kpts_xy[idx]
        if np.isnan(x) or np.isnan(y):
            continue
        if not _kpt_inside_bed(x, y, bed_mask, bed_bbox):
            limbs_out = True
            break

    # weak mask / roi fallback: hip in zone bbox counts as on-zone
    zone_q = bed.get("zone_quality", "")
    if zone_q in ("mask_weak", "bbox_only", "roi_only") and bed_bbox is not None:
        cx, cy = center
        bx1, by1, bx2, by2 = bed_bbox
        hip_in_bbox = bx1 <= cx <= bx2 and by1 <= cy <= by2
        if hip_in_bbox and not hip_on_seg:
            hip_on_seg = True

    # A true bed-edge sit can place the detected pelvis a few pixels outside
    # the mattress mask. Proximity alone is insufficient: standing beside the
    # bed must remain off-seg.
    seated_edge_contact = (
        _is_seated_skeleton(
            kpts_xy, kpts_conf, conf_threshold=conf_threshold
        )
        and _point_near_bed(
            center, bed_mask, bed_bbox,
            max_distance_ratio=edge_distance_ratio,
        )
    )
    if seated_edge_contact and (not hip_on_seg or ratio < attach_min):
        return "partial", ratio, True
    if not hip_on_seg or ratio < attach_min:
        return "off_seg", ratio, limbs_out
    if limbs_out or ratio < on_seg_min:
        return "partial", ratio, limbs_out
    return "on_seg", ratio, limbs_out


def calc_limb_overflow(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    bed_bbox: tuple[int, int, int, int] | None,
    *,
    conf_threshold: float = 0.3,
) -> float:
    if bed_bbox is None or kpts_xy.shape[0] < 17:
        return 0.0
    x_min, _, x_max, _ = bed_bbox
    bed_width = x_max - x_min
    if bed_width <= 0:
        return 0.0
    max_overflow = 0.0
    for idx in RISK_KEYPOINTS.values():
        if kpts_conf[idx] < conf_threshold:
            continue
        x, y = kpts_xy[idx]
        if np.isnan(x) or np.isnan(y) or (x < 1 and y < 1):
            continue
        if x < x_min:
            max_overflow = max(max_overflow, (x_min - x) / bed_width)
        elif x > x_max:
            max_overflow = max(max_overflow, (x - x_max) / bed_width)
    return float(max_overflow)


def overflow_to_risk_level(
    overflow: float,
    thresholds: dict[str, float],
) -> str:
    if overflow >= thresholds.get("overflow_high", 0.25):
        return "HIGH"
    if overflow >= thresholds.get("overflow_med", 0.15):
        return "MED"
    if overflow >= thresholds.get("overflow_low", 0.05):
        return "LOW"
    return "SAFE"


def person_in_bed(
    center: tuple[float, float] | None,
    bed_bbox: tuple[int, int, int, int] | None,
    bed_mask: np.ndarray | None = None,
) -> tuple[bool, str]:
    if center is None:
        return False, "none"
    cx, cy = int(center[0]), int(center[1])
    if bed_mask is not None:
        h, w = bed_mask.shape[:2]
        if 0 <= cy < h and 0 <= cx < w and bed_mask[cy, cx] > 0:
            return True, "mask"
    if bed_bbox is not None:
        bx1, by1, bx2, by2 = bed_bbox
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            return True, "bbox"
    return False, "none"
