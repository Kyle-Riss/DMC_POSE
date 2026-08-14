"""Per-frame enrich from pose keypoints + bed (live RTSP)."""
from __future__ import annotations

import numpy as np

from bed_monitor.bed_zone import build_approx_bed_zone
from bed_monitor.features import MotionState, update_motion
from bed_monitor.geometry import edge_zone_from_bbox, hip_center_from_kpts
from bed_monitor.risk_rules import (
    calc_limb_overflow,
    classify_seg_attachment_with_preset,
    count_skeleton_keypoints,
    overflow_to_risk_level,
    skeleton_person_detected,
)
from bed_monitor.scoring import FallScorer
from bed_monitor.zone_norm import compute_zone_risk_joints, edge_zone_to_rail_band


def enrich_from_keypoints(
    kxy: np.ndarray,
    kconf: np.ndarray,
    bed: dict,
    motion_state: MotionState,
    t_sec: float,
    preset: dict,
    *,
    frame_hw: tuple[int, int] | None = None,
    roi_bbox: tuple[int, int, int, int] | None = None,
    pose_ko: str | None = None,
    rail_left_up: bool | None = None,
    rail_right_up: bool | None = None,
    scorer: FallScorer | None = None,
) -> dict:
    """Skeleton + approx bed zone → frame features (live + batch)."""
    thresholds = preset.get("risk_thresholds", {})
    motion_cfg = preset.get("motion", {})
    ev_cfg = preset.get("events", {})
    inf = preset.get("inference", {})
    kpt_conf = float(inf.get("kpt_conf", 0.3))

    if preset.get("bed_zone") and not bed.get("zone_built") and frame_hw is not None:
        h, w = frame_hw
        bed = build_approx_bed_zone(bed, roi_bbox, h, w, preset)

    min_core = int(inf.get("skeleton_min_core_kpts", ev_cfg.get("min_torso_kpts", 1)))
    min_total = int(inf.get("skeleton_min_total_kpts", ev_cfg.get("min_valid_kpts", 5)))

    skel_total, skel_core = count_skeleton_keypoints(kxy, kconf, conf_threshold=kpt_conf)
    person = skeleton_person_detected(
        kxy,
        kconf,
        conf_threshold=kpt_conf,
        min_core=min_core,
        min_total=min_total,
    )

    bed_bbox = bed.get("bbox")
    center = hip_center_from_kpts(kxy) if person else None

    if person:
        seg_attachment, kpt_ratio, limbs_outside = classify_seg_attachment_with_preset(
            kxy,
            kconf,
            center,
            bed,
            preset,
            conf_threshold=kpt_conf,
        )
    else:
        seg_attachment, kpt_ratio, limbs_outside = "none", 0.0, False

    in_bed_flag = seg_attachment in ("on_seg", "partial")
    in_bed_method = "mask" if seg_attachment == "on_seg" else (
        "partial" if seg_attachment == "partial" else "none"
    )

    overflow = (
        calc_limb_overflow(kxy, kconf, bed_bbox, conf_threshold=kpt_conf) if person else 0.0
    )
    risk = overflow_to_risk_level(overflow, thresholds) if person else "SAFE"
    edge = edge_zone_from_bbox(center[0], bed_bbox) if center else None

    motion = update_motion(
        motion_state,
        center,
        t_sec,
        ema_alpha=float(motion_cfg.get("ema_alpha", 0.3)),
        hold_sec=float(motion_cfg.get("hold_sec", 0.5)),
    )

    out = {
        "person_detected": person,
        "skeleton_kpts": skel_total,
        "skeleton_core_kpts": skel_core,
        "skeleton_kpts_need": min_total,
        "skeleton_core_need": min_core,
        "bed_source": bed.get("source", "none"),
        "zone_quality": bed.get("zone_quality", "none"),
        "seg_attachment": seg_attachment,
        "kpt_on_seg_ratio": kpt_ratio,
        "limbs_outside_seg": limbs_outside,
        "attached_to_seg": seg_attachment in ("on_seg", "partial"),
        "in_bed": in_bed_flag,
        "in_bed_method": in_bed_method,
        "limb_overflow_max": overflow,
        "risk_level": risk,
        "edge_zone": edge,
        "center_x": motion.center_x,
        "center_y": motion.center_y,
        "center_speed": motion.center_speed,
        "fall_score": 0.0,
        "fall_level": "SAFE",
        "fall_status": "NO_PERSON",
        "rail_risk": 0.0,
        "zone_risk": 0.0,
        "pose_risk": 0.0,
    }

    scoring_cfg = preset.get("scoring") or {}
    if scoring_cfg.get("enabled") and scorer is not None:
        zone_risk_val = None
        zone_label = edge_zone_to_rail_band(edge)
        joint_detail: dict = {}
        if person and bed_bbox is not None:
            zone_risk_val, joint_detail = compute_zone_risk_joints(
                kxy,
                kconf,
                bed_bbox,
                scoring_cfg.get("zones", {}),
                scoring_cfg.get("zone_risk", {}),
            )
        fall = scorer.score(
            person_detected=person,
            seg_attachment=seg_attachment,
            in_bed=in_bed_flag,
            edge_zone=edge,
            pose_ko=pose_ko,
            rail_left_up=rail_left_up,
            rail_right_up=rail_right_up,
            zone_risk=zone_risk_val,
            zone_label=zone_label,
        )
        detail = dict(fall.detail)
        if joint_detail:
            detail["zone_risk_joints"] = joint_detail
        out.update(
            {
                "fall_score": round(float(fall.score), 2),
                "fall_level": fall.level,
                "fall_status": fall.status,
                "rail_risk": round(float(fall.rail_risk), 4),
                "zone_risk": round(float(fall.zone_risk), 4),
                "pose_risk": round(float(fall.pose_risk), 4),
            }
        )

    return out


def apply_fall_scoring(
    feat: dict,
    kxy: np.ndarray,
    kconf: np.ndarray,
    bed_bbox: tuple[int, int, int, int] | None,
    preset: dict,
    scorer: FallScorer,
    *,
    pose_ko: str | None = None,
    rail_left_up: bool | None = None,
    rail_right_up: bool | None = None,
) -> dict:
    """Add fall_score fields to an existing feat dict (after pose/rail on live server)."""
    scoring_cfg = preset.get("scoring") or {}
    if not scoring_cfg.get("enabled"):
        return feat

    person = bool(feat.get("person_detected"))
    seg_attachment = str(feat.get("seg_attachment", "none"))
    in_bed_flag = bool(feat.get("in_bed"))
    edge = feat.get("edge_zone")

    zone_risk_val = None
    zone_label = edge_zone_to_rail_band(edge)
    if person and bed_bbox is not None:
        zone_risk_val, _joint_detail = compute_zone_risk_joints(
            kxy,
            kconf,
            bed_bbox,
            scoring_cfg.get("zones", {}),
            scoring_cfg.get("zone_risk", {}),
        )

    fall = scorer.score(
        person_detected=person,
        seg_attachment=seg_attachment,
        in_bed=in_bed_flag,
        edge_zone=edge,
        pose_ko=pose_ko,
        rail_left_up=rail_left_up,
        rail_right_up=rail_right_up,
        zone_risk=zone_risk_val,
        zone_label=zone_label,
    )
    feat.update(
        {
            "fall_score": round(float(fall.score), 2),
            "fall_level": fall.level,
            "fall_status": fall.status,
            "rail_risk": round(float(fall.rail_risk), 4),
            "zone_risk": round(float(fall.zone_risk), 4),
            "pose_risk": round(float(fall.pose_risk), 4),
        }
    )
    return feat
