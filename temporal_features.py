"""Shared, model-independent pose features for offline and live temporal paths."""

from __future__ import annotations

import numpy as np


FEATURE_SCHEMA_VERSION = "pose_temporal_109_v1"
FEATURE_COUNT = 109


def temporal_feature_names() -> list[str]:
    """Return the canonical index order shared by training and live inference."""
    norm = [
        name
        for index in range(17)
        for name in (f"kpt_{index}_x_norm", f"kpt_{index}_y_norm")
    ]
    confidence = [f"kpt_{index}_conf" for index in range(17)]
    visibility = [f"kpt_{index}_visible" for index in range(17)]
    probabilities = [f"pose_prob_{index}" for index in range(6)]
    velocity = [f"velocity_{name}" for name in norm]
    names = (
        norm + confidence + visibility + probabilities
        + ["person_detected"] + velocity
    )
    if len(names) != FEATURE_COUNT:
        raise AssertionError(f"unexpected temporal feature count: {len(names)}")
    return names


def normalize_pose(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    *,
    min_conf: float = 0.25,
) -> dict[str, np.ndarray | float]:
    """Return translation/scale-normalized COCO-17 features.

    Missing joints become zero in normalized space and remain identifiable via
    the returned visibility vector.
    """
    xy = np.asarray(keypoints_xy, dtype=np.float32)
    conf = np.asarray(keypoints_conf, dtype=np.float32)
    if xy.shape != (17, 2):
        raise ValueError(f"keypoints_xy must be (17, 2), got {xy.shape}")
    if conf.shape != (17,):
        raise ValueError(f"keypoints_conf must be (17,), got {conf.shape}")

    finite = np.isfinite(xy).all(axis=1) & np.isfinite(conf)
    visible = finite & (conf >= min_conf) & (xy[:, 0] > 0) & (xy[:, 1] > 0)
    valid_xy = xy[visible]

    if len(valid_xy) < 2:
        return {
            "xy_norm": np.zeros((17, 2), dtype=np.float32),
            "visibility": visible.astype(np.float32),
            "center": np.array([np.nan, np.nan], dtype=np.float32),
            "scale": np.float32(np.nan),
        }

    # Prefer hip centre when both hips exist; otherwise use visible bbox centre.
    if visible[11] and visible[12]:
        center = (xy[11] + xy[12]) / 2.0
    else:
        center = (valid_xy.min(axis=0) + valid_xy.max(axis=0)) / 2.0

    extent = valid_xy.max(axis=0) - valid_xy.min(axis=0)
    scale = float(max(extent[0], extent[1], 1.0))
    xy_norm = np.zeros((17, 2), dtype=np.float32)
    xy_norm[visible] = (xy[visible] - center) / scale
    return {
        "xy_norm": xy_norm,
        "visibility": visible.astype(np.float32),
        "center": center.astype(np.float32),
        "scale": np.float32(scale),
    }


def temporal_feature_vector(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    pose_probs: np.ndarray,
    *,
    previous_xy_norm: np.ndarray | None = None,
    previous_visibility: np.ndarray | None = None,
    dt: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact 109-value feature row used by ``build_temporal_windows``.

    Order: normalized XY (34), confidence (17), visibility (17), six-class
    probabilities (6), person flag (1), visibility-masked XY velocity (34).
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    normalized = normalize_pose(keypoints_xy, keypoints_conf)
    xy_norm = np.asarray(normalized["xy_norm"], dtype=np.float32)
    visibility = np.asarray(normalized["visibility"], dtype=np.float32)
    confidence = np.nan_to_num(
        np.asarray(keypoints_conf, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    probabilities = np.nan_to_num(
        np.asarray(pose_probs, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if confidence.shape != (17,):
        raise ValueError(f"keypoints_conf must be (17,), got {confidence.shape}")
    if probabilities.shape != (6,):
        raise ValueError(f"pose_probs must be (6,), got {probabilities.shape}")

    velocity = np.zeros((17, 2), dtype=np.float32)
    if previous_xy_norm is not None and previous_visibility is not None:
        previous_xy = np.asarray(previous_xy_norm, dtype=np.float32)
        previous_visible = np.asarray(previous_visibility, dtype=np.float32)
        if previous_xy.shape != (17, 2) or previous_visible.shape != (17,):
            raise ValueError("previous normalized pose must have shapes (17, 2) and (17,)")
        joint_visible = visibility * previous_visible
        velocity = ((xy_norm - previous_xy) / float(dt)) * joint_visible[:, None]

    person_detected = np.array([float(visibility.sum() >= 5)], dtype=np.float32)
    vector = np.concatenate(
        [
            xy_norm.reshape(-1),
            confidence,
            visibility,
            probabilities,
            person_detected,
            velocity.reshape(-1),
        ]
    ).astype(np.float32)
    if vector.shape != (FEATURE_COUNT,):
        raise AssertionError(f"unexpected temporal feature shape: {vector.shape}")
    return vector, xy_norm, visibility


def labels_at(timestamp_sec: float, intervals: list[dict]) -> tuple[str, list[str]]:
    """Return temporal target and all active source labels at time t.

    ``ignore`` has priority over every trainable label. It marks uncertain
    weak-label transitions that must not become false-negative targets.
    """
    active = [
        str(item["label"])
        for item in intervals
        if float(item["start_sec"]) <= timestamp_sec <= float(item["end_sec"])
    ]
    if "ignore" in active:
        return "ignore", active
    return ("fall" if "fall" in active else "non_fall"), active
