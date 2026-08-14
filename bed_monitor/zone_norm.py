"""Joint zone risk from bed bbox (pose-sixclass geometry, no trapezoid)."""
from __future__ import annotations

import numpy as np

JOINT_TYPES = {
    "hip": [11, 12],
    "wrist": [9, 10],
    "ankle": [15, 16],
}


def _classify_band(norm_y: float, labels: tuple[str, ...], count: int) -> str:
    idx = int(np.clip(norm_y * count, 0, count - 1))
    return labels[idx]


def edge_zone_to_rail_band(edge_zone: str | None) -> str | None:
    """Map pose-sixclass L/C/R edge zone to scoring band label for rail bonus."""
    if edge_zone == "L":
        return "lower"
    if edge_zone == "R":
        return "upper"
    return None


def compute_zone_risk_joints(
    kxy: np.ndarray,
    kconf: np.ndarray,
    bed_bbox: tuple[int, int, int, int] | None,
    zones_cfg: dict,
    zone_risk_cfg: dict,
) -> tuple[float | None, dict]:
    if bed_bbox is None or kxy is None:
        return None, {}

    x0, y0, x1, y1 = bed_bbox
    bw = max(float(x1 - x0), 1.0)
    bh = max(float(y1 - y0), 1.0)

    band = zone_risk_cfg.get("band", {})
    out_val = float(zone_risk_cfg.get("out_of_bed_joint", 1.0))
    joint_weights = zone_risk_cfg.get("joint_weights", {})
    conf_threshold = float(zone_risk_cfg.get("kpt_conf", 0.3))
    labels = tuple(zones_cfg.get("labels", ["upper", "center", "lower"]))
    count = int(zones_cfg.get("count", 3))

    kxy = np.asarray(kxy, dtype=np.float32)
    kconf = np.asarray(kconf, dtype=np.float32)

    detail: dict[str, float] = {}
    num = 0.0
    den = 0.0

    for jtype, idxs in JOINT_TYPES.items():
        w = float(joint_weights.get(jtype, 0.0))
        if w <= 0:
            continue
        risks: list[float] = []
        for i in idxs:
            if i >= len(kconf) or kconf[i] < conf_threshold:
                continue
            x, y = float(kxy[i, 0]), float(kxy[i, 1])
            if np.isnan(x) or np.isnan(y):
                continue
            nx = (x - x0) / bw
            ny = (y - y0) / bh
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                zone = _classify_band(ny, labels, count)
                risks.append(float(band.get(zone, out_val)))
            else:
                risks.append(out_val)
        if risks:
            type_risk = sum(risks) / len(risks)
            detail[jtype] = round(type_risk, 3)
            num += w * type_risk
            den += w

    if den == 0:
        return None, detail
    return num / den, detail
