"""Rail up/down detection — restored from pose-sixclass-viewer/server.py (2026-05-11)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

RAIL_DIR = Path(__file__).resolve().parent
RAIL_CONFIG_PATH = Path(
    __import__("os").environ.get("POSE_RAIL_CONFIG", str(RAIL_DIR / "rail_config.json"))
)


def _default_config() -> dict[str, Any]:
    return {
        "right_rail_roi": {"x0": 0.10, "x1": 0.90, "y0": 0.70, "y1": 0.90},
        "left_rail_roi": {"x0": 0.05, "x1": 0.40, "y0": 0.06, "y1": 0.26},
        "method": "edge",
        "method_left": "edge",
        "method_right": "edge",
        "reference_image": None,
        "diff_threshold": 10.0,
        "reference_down": None,
        "reference_up": None,
        "decision_margin": 2.0,
        "left_reference_down": None,
        "left_reference_up": None,
        "left_decision_margin": 2.0,
        "right_reference_down": None,
        "right_reference_up": None,
        "right_decision_margin": 2.0,
        "edge_density_threshold": 0.02,
        "canny": {"t1": 50, "t2": 150},
        "exclude_person_from_rail_diff": True,
        "rail_person_bbox_expand": 0.15,
        "rail_min_valid_pixels_ratio": 0.12,
        "rail_sitting_pose_labels": ["앉음_중앙", "앉음_가장자리"],
        "rail_sitting_up_margin_mult": 2.2,
        "rail_sitting_down_margin_mult": 1.0,
        "rail_sitting_person_expand_add": 0.12,
    }


def load_rail_config(path: Path | None = None) -> dict[str, Any]:
    cfg = _default_config()
    cfg_path = path or RAIL_CONFIG_PATH
    if not cfg_path.is_file():
        return cfg
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            cfg.update(loaded)
    except Exception:
        logging.warning("rail config load failed; using defaults", exc_info=True)
    return cfg


def resolve_rail_image(path: str | None, base_dir: Path | None = None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    base = base_dir or RAIL_CONFIG_PATH.parent
    candidate = base / path
    return candidate if candidate.is_file() else None


def load_rail_reference_gray(path: str | None, base_dir: Path | None = None) -> np.ndarray | None:
    img_path = resolve_rail_image(path, base_dir)
    if img_path is None:
        return None
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _roi_bounds_px(frame_bgr: np.ndarray, roi: dict[str, float]) -> tuple[int, int, int, int] | None:
    h, w = frame_bgr.shape[:2]
    x0 = int(max(0, min(w - 1, roi["x0"] * w)))
    x1 = int(max(0, min(w, roi["x1"] * w)))
    y0 = int(max(0, min(h - 1, roi["y0"] * h)))
    y1 = int(max(0, min(h, roi["y1"] * h)))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return None
    return x0, x1, y0, y1


def _roi_crop_gray(frame_bgr: np.ndarray, roi: dict[str, float]) -> np.ndarray | None:
    b = _roi_bounds_px(frame_bgr, roi)
    if b is None:
        return None
    x0, x1, y0, y1 = b
    patch = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def rail_edge_density(frame_bgr: np.ndarray, roi: dict[str, float], canny: dict[str, int]) -> float:
    b = _roi_bounds_px(frame_bgr, roi)
    if b is None:
        return 0.0
    x0, x1, y0, y1 = b
    patch = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, int(canny["t1"]), int(canny["t2"]))
    return float((edges > 0).mean())


def rail_roi_valid_mask(
    frame_bgr: np.ndarray,
    roi: dict[str, float],
    person_xyxy,
    expand_ratio: float,
) -> np.ndarray | None:
    b = _roi_bounds_px(frame_bgr, roi)
    if b is None:
        return None
    x0, x1, y0, y1 = b
    ph, pw = y1 - y0, x1 - x0
    mask = np.ones((ph, pw), dtype=np.float32)
    if person_xyxy is None or len(person_xyxy) < 4:
        return mask

    h, w = frame_bgr.shape[:2]
    px1, py1, px2, py2 = [float(v) for v in person_xyxy[:4]]
    bw = max(1.0, px2 - px1)
    bh = max(1.0, py2 - py1)
    pad_x = bw * expand_ratio * 0.5
    pad_y = bh * expand_ratio * 0.5
    qx1 = max(0.0, px1 - pad_x)
    qy1 = max(0.0, py1 - pad_y)
    qx2 = min(float(w - 1), px2 + pad_x)
    qy2 = min(float(h - 1), py2 + pad_y)
    if qx2 <= qx1 + 1 or qy2 <= qy1 + 1:
        return mask

    rx0 = int(max(0, min(pw, int(np.floor(qx1)) - x0)))
    ry0 = int(max(0, min(ph, int(np.floor(qy1)) - y0)))
    rx1 = int(max(0, min(pw, int(np.ceil(qx2)) - x0)))
    ry1 = int(max(0, min(ph, int(np.ceil(qy2)) - y0)))
    if rx1 <= rx0 + 1 or ry1 <= ry0 + 1:
        return mask
    mask[ry0:ry1, rx0:rx1] = 0.0
    return mask


def rail_diff_score(
    frame_bgr: np.ndarray,
    roi: dict[str, float],
    ref_gray: np.ndarray | None,
    person_xyxy=None,
    *,
    exclude_person: bool = True,
    person_expand_ratio: float = 0.15,
    min_valid_pixels_ratio: float = 0.12,
) -> float | None:
    cur = _roi_crop_gray(frame_bgr, roi)
    if cur is None or ref_gray is None:
        return None
    if cur.shape != ref_gray.shape:
        ref = cv2.resize(ref_gray, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        ref = ref_gray
    diff = cv2.absdiff(cur, ref).astype(np.float32)

    if not (exclude_person and person_xyxy is not None):
        return float(diff.mean())

    vm = rail_roi_valid_mask(frame_bgr, roi, person_xyxy, person_expand_ratio)
    if vm is None:
        return None
    wsum = float(vm.sum())
    need = float(min_valid_pixels_ratio) * float(cur.size)
    if wsum < max(32.0, need):
        return None
    return float((diff * vm).sum() / wsum)


def detect_rail_state(
    frame: np.ndarray,
    method: str,
    roi: dict[str, float],
    edge_canny: dict[str, int],
    edge_threshold: float,
    diff_ref_gray: np.ndarray | None = None,
    diff_threshold: float = 10.0,
    diff2_ref_down: np.ndarray | None = None,
    diff2_ref_up: np.ndarray | None = None,
    decision_margin: float = 2.0,
    decision_margin_up: float | None = None,
    decision_margin_down: float | None = None,
    prev_state: bool | None = None,
    person_xyxy=None,
    exclude_person: bool = True,
    person_expand_ratio: float = 0.15,
    min_valid_pixels_ratio: float = 0.12,
) -> tuple[bool, float, str]:
    """Returns (rail_up, score, method_tag)."""
    method = (method or "edge").lower()

    if method == "diff2":
        if diff2_ref_down is None or diff2_ref_up is None:
            up = bool(prev_state) if prev_state is not None else False
            return up, 0.0, "diff2-missing"

        d_down = rail_diff_score(
            frame, roi, diff2_ref_down, person_xyxy,
            exclude_person=exclude_person,
            person_expand_ratio=person_expand_ratio,
            min_valid_pixels_ratio=min_valid_pixels_ratio,
        )
        d_up = rail_diff_score(
            frame, roi, diff2_ref_up, person_xyxy,
            exclude_person=exclude_person,
            person_expand_ratio=person_expand_ratio,
            min_valid_pixels_ratio=min_valid_pixels_ratio,
        )
        tag = "diff2-mask" if exclude_person and person_xyxy is not None else "diff2"
        if d_down is None or d_up is None:
            up = bool(prev_state) if prev_state is not None else False
            return up, 0.0, f"{tag}-occluded"

        m_up = float(decision_margin_up if decision_margin_up is not None else decision_margin)
        m_down = float(decision_margin_down if decision_margin_down is not None else decision_margin)
        if (d_down - d_up) > m_up:
            up = True
        elif (d_up - d_down) > m_down:
            up = False
        else:
            up = bool(prev_state) if prev_state is not None else False
        return up, float(d_down - d_up), tag

    if method == "diff":
        score = rail_diff_score(
            frame, roi, diff_ref_gray, person_xyxy,
            exclude_person=exclude_person,
            person_expand_ratio=person_expand_ratio,
            min_valid_pixels_ratio=min_valid_pixels_ratio,
        )
        if score is None:
            up = bool(prev_state) if prev_state is not None else False
            return up, 0.0, "diff-occluded"
        return score >= diff_threshold, float(score), "diff"

    score = rail_edge_density(frame, roi, edge_canny)
    return score >= edge_threshold, float(score), "edge"


def _sitting_rail_params(cfg: dict[str, Any], pose_label: str | None) -> tuple[bool, float, float, float, float, float, float]:
    sitting_labels = cfg.get("rail_sitting_pose_labels") or []
    is_sitting = bool(pose_label and pose_label in sitting_labels)
    base_expand = float(cfg.get("rail_person_bbox_expand", 0.15))
    eff_expand = base_expand + (
        float(cfg.get("rail_sitting_person_expand_add", 0.12)) if is_sitting else 0.0
    )
    up_mult = float(cfg.get("rail_sitting_up_margin_mult", 2.2)) if is_sitting else 1.0
    down_mult = float(cfg.get("rail_sitting_down_margin_mult", 1.0)) if is_sitting else 1.0
    left_margin = float(cfg.get("left_decision_margin", cfg.get("decision_margin", 2.0)))
    right_margin = float(cfg.get("right_decision_margin", cfg.get("decision_margin", 2.0)))
    return (
        is_sitting,
        eff_expand,
        left_margin * up_mult,
        left_margin * down_mult,
        right_margin * up_mult,
        right_margin * down_mult,
    )


def _tag_with_sitting(tag: str, is_sitting: bool) -> str:
    if is_sitting and tag.startswith("diff2"):
        return f"{tag}+sit"
    return tag


def draw_rail_rois(
    frame: np.ndarray,
    cfg: dict[str, Any],
    left_up: bool | None = None,
    right_up: bool | None = None,
) -> np.ndarray:
    out = frame
    for roi_key, up, label in (
        ("left_rail_roi", left_up, "L"),
        ("right_rail_roi", right_up, "R"),
    ):
        roi = cfg.get(roi_key)
        if not roi:
            continue
        b = _roi_bounds_px(out, roi)
        if b is None:
            continue
        x0, x1, y0, y1 = b
        color = (0, 255, 255) if up else (180, 180, 180)
        if up is None:
            color = (255, 180, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            out, label, (x0 + 2, max(12, y0 + 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA,
        )
    return out


def detect_both_rails(
    frame: np.ndarray,
    cfg: dict[str, Any] | None = None,
    person_xyxy=None,
    prev_left: bool | None = None,
    prev_right: bool | None = None,
    pose_label: str | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_rail_config()
    base = RAIL_CONFIG_PATH.parent

    left_roi = cfg["left_rail_roi"]
    right_roi = cfg["right_rail_roi"]
    canny = cfg["canny"]
    (
        is_sitting,
        eff_expand,
        m_up_l,
        m_dn_l,
        m_up_r,
        m_dn_r,
    ) = _sitting_rail_params(cfg, pose_label)

    left_up, left_score, left_tag = detect_rail_state(
        frame,
        cfg.get("method_left") or cfg.get("method", "edge"),
        left_roi,
        canny,
        cfg["edge_density_threshold"],
        diff_ref_gray=load_rail_reference_gray(cfg.get("reference_image"), base),
        diff_threshold=cfg["diff_threshold"],
        diff2_ref_down=load_rail_reference_gray(cfg.get("left_reference_down"), base),
        diff2_ref_up=load_rail_reference_gray(cfg.get("left_reference_up"), base),
        decision_margin=cfg["decision_margin"],
        decision_margin_up=m_up_l,
        decision_margin_down=m_dn_l,
        prev_state=prev_left,
        person_xyxy=person_xyxy,
        exclude_person=cfg.get("exclude_person_from_rail_diff", True),
        person_expand_ratio=eff_expand,
        min_valid_pixels_ratio=cfg.get("rail_min_valid_pixels_ratio", 0.12),
    )
    right_up, right_score, right_tag = detect_rail_state(
        frame,
        cfg.get("method_right") or cfg.get("method", "edge"),
        right_roi,
        canny,
        cfg["edge_density_threshold"],
        diff_ref_gray=load_rail_reference_gray(cfg.get("reference_image"), base),
        diff_threshold=cfg["diff_threshold"],
        diff2_ref_down=load_rail_reference_gray(cfg.get("right_reference_down"), base),
        diff2_ref_up=load_rail_reference_gray(cfg.get("right_reference_up"), base),
        decision_margin=cfg["decision_margin"],
        decision_margin_up=m_up_r,
        decision_margin_down=m_dn_r,
        prev_state=prev_right,
        person_xyxy=person_xyxy,
        exclude_person=cfg.get("exclude_person_from_rail_diff", True),
        person_expand_ratio=eff_expand,
        min_valid_pixels_ratio=cfg.get("rail_min_valid_pixels_ratio", 0.12),
    )
    return {
        "rail_left_up": left_up,
        "rail_left_score": left_score,
        "rail_left_method": _tag_with_sitting(left_tag, is_sitting),
        "rail_right_up": right_up,
        "rail_right_score": right_score,
        "rail_right_method": _tag_with_sitting(right_tag, is_sitting),
    }
