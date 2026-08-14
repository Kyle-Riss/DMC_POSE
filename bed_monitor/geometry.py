from __future__ import annotations

import numpy as np


def kpts_to_xy_conf(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten CSV kpt_0..33 → (17,2) xy and (17,) conf (conf unknown → 1.0 if valid)."""
    xy = np.full((17, 2), np.nan, dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    for i in range(17):
        x = row.get(f"kpt_{i * 2}")
        y = row.get(f"kpt_{i * 2 + 1}")
        if x is None or y is None or (isinstance(x, float) and np.isnan(x)):
            continue
        xy[i, 0] = float(x)
        xy[i, 1] = float(y)
        conf[i] = 1.0
    return xy, conf


def hip_center_from_kpts(xy: np.ndarray) -> tuple[float, float] | None:
    if xy.shape[0] < 13:
        return None
    pts = []
    for idx in (11, 12):
        if not np.isnan(xy[idx, 0]):
            pts.append(xy[idx])
    if not pts:
        valid = xy[~np.isnan(xy[:, 0])]
        if len(valid) == 0:
            return None
        return float(valid[:, 0].mean()), float(valid[:, 1].mean())
    arr = np.array(pts)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def edge_zone_from_bbox(cx: float, bed_bbox: tuple[int, int, int, int] | None) -> str | None:
    if bed_bbox is None:
        return None
    x_min, _, x_max, _ = bed_bbox
    bed_w = x_max - x_min
    if bed_w <= 0:
        return None
    rel = (cx - x_min) / bed_w
    if rel < 1.0 / 3.0:
        return "L"
    if rel > 2.0 / 3.0:
        return "R"
    return "C"
