"""Shared bed geometry, risk rules, and motion features (batch + live)."""

from bed_monitor.geometry import edge_zone_from_bbox, hip_center_from_kpts, kpts_to_xy_conf
from bed_monitor.risk_rules import calc_limb_overflow, overflow_to_risk_level, person_in_bed
from bed_monitor.features import update_motion
from bed_monitor.live import enrich_from_keypoints
from bed_monitor.temporal import LiveEventTracker

__all__ = [
    "edge_zone_from_bbox",
    "hip_center_from_kpts",
    "kpts_to_xy_conf",
    "calc_limb_overflow",
    "overflow_to_risk_level",
    "person_in_bed",
    "update_motion",
    "enrich_from_keypoints",
    "LiveEventTracker",
]
