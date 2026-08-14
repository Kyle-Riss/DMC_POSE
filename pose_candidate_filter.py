"""Pure post-processing gate for Pose person candidates."""

from typing import Iterable


def accept_pose_candidate(
    *,
    box_confidence: float,
    area_ratio: float,
    visible_count: int,
    keypoint_mean: float,
    strong_box_conf: float,
    strong_min_area_ratio: float,
    weak_box_conf: float,
    weak_min_area_ratio: float,
    weak_min_visible: int,
    weak_min_keypoint_mean: float,
) -> bool:
    """Accept high-confidence small people without admitting tiny clutter.

    Strong detections use a separately calibrated area floor. Weak detections
    retain the larger floor and must also have a plausible keypoint structure.
    """
    strong_candidate = (
        box_confidence >= strong_box_conf
        and area_ratio >= strong_min_area_ratio
    )
    structured_weak_candidate = (
        box_confidence >= weak_box_conf
        and visible_count >= weak_min_visible
        and keypoint_mean >= weak_min_keypoint_mean
        and area_ratio >= weak_min_area_ratio
    )
    return strong_candidate or structured_weak_candidate


def select_tracking_bbox(
    detector_bbox: Iterable[float] | None,
    keypoint_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """Prefer the detector box because visible-keypoint boxes jump by pose."""
    if detector_bbox is not None:
        values = tuple(float(value) for value in detector_bbox)
        if len(values) == 4:
            x1, y1, x2, y2 = values
            if x2 > x1 and y2 > y1:
                return values
    return keypoint_bbox
