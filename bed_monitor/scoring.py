"""Weighted fall score 0–100 (adapted from fall_monitor, pose-sixclass inputs)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from bed_monitor.zone_norm import edge_zone_to_rail_band


@dataclass
class FallScore:
    score: float
    raw_score: float
    level: str
    status: str
    rail_risk: float
    zone_risk: float
    pose_risk: float
    rails_down: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def _rail_down_names(rail_states: dict | None) -> list[str]:
    if not rail_states:
        return []
    return [name for name, st in rail_states.items() if st.get("state") == "DOWN"]


def rail_states_from_flags(
    rail_left_up: bool | None,
    rail_right_up: bool | None,
    scoring_cfg: dict,
) -> dict:
    rr = scoring_cfg.get("rail_risk", {})
    left = rr.get("left_label", "Lt")
    right = rr.get("right_label", "Rt")
    states: dict = {}
    if rail_left_up is not None:
        states[left] = {"state": "UP" if rail_left_up else "DOWN"}
    if rail_right_up is not None:
        states[right] = {"state": "UP" if rail_right_up else "DOWN"}
    return states


def compute_rail_risk(
    rail_states: dict | None,
    edge_zone: str | None,
    rail_cfg: dict,
) -> tuple[float, list[str]]:
    down = _rail_down_names(rail_states)
    n_down = len(down)

    if not rail_states:
        return 0.0, down
    if n_down == 0:
        return float(rail_cfg["both_up"]), down
    if n_down >= 2:
        return float(rail_cfg["both_down"]), down

    risk = float(rail_cfg["one_down"])
    band = edge_zone_to_rail_band(edge_zone)
    if band is not None and band in down:
        risk = min(1.0, risk + float(rail_cfg.get("near_down_rail_bonus", 0.3)))
    return risk, down


def compute_zone_risk_band(zone_label: str | None, zone_cfg: dict) -> float:
    band = zone_cfg.get("band", zone_cfg)
    if zone_label is None:
        return float(band.get("center", 0.1))
    return float(band.get(zone_label, 0.1))


def compute_pose_risk(pose_ko: str | None, pose_cfg: dict) -> float:
    if not pose_ko or pose_ko == "None":
        return float(pose_cfg.get("None", 0.0))
    return float(pose_cfg.get(pose_ko, 0.0))


def score_to_level(score: float, levels_cfg: dict) -> str:
    if score >= levels_cfg["HIGH"]:
        return "HIGH"
    if score >= levels_cfg["MED"]:
        return "MED"
    if score >= levels_cfg["LOW"]:
        return "LOW"
    return "SAFE"


class FallScorer:
    def __init__(self, scoring_cfg: dict):
        self.cfg = scoring_cfg
        w = scoring_cfg["weights"]
        total = w["rail"] + w["zone"] + w["pose"]
        if total <= 0:
            total = 1.0
        self.w_rail = w["rail"] / total
        self.w_zone = w["zone"] / total
        self.w_pose = w["pose"] / total

        t = scoring_cfg.get("temporal", {})
        self.temporal_mode = t.get("mode", "ema")
        self.ema_alpha = float(t.get("ema_alpha", 0.4))
        self._buf: deque[float] = deque(maxlen=int(t.get("window", 10)))
        self._ema: float | None = None

    def _smooth(self, raw: float) -> float:
        if self.temporal_mode == "mean":
            self._buf.append(raw)
            return sum(self._buf) / len(self._buf)
        if self._ema is None:
            self._ema = raw
        else:
            self._ema = self.ema_alpha * raw + (1 - self.ema_alpha) * self._ema
        return self._ema

    def _out_of_bed_triggered(
        self,
        *,
        seg_attachment: str,
        in_bed: bool,
        oob_cfg: dict,
    ) -> bool:
        trigger = oob_cfg.get("trigger", "seg_off")
        if trigger == "seg_off":
            return seg_attachment == "off_seg"
        return not in_bed

    def score(
        self,
        *,
        person_detected: bool,
        seg_attachment: str,
        in_bed: bool,
        edge_zone: str | None,
        pose_ko: str | None,
        rail_left_up: bool | None = None,
        rail_right_up: bool | None = None,
        zone_risk: float | None = None,
        zone_label: str | None = None,
    ) -> FallScore:
        levels = self.cfg["levels"]
        rail_states = rail_states_from_flags(rail_left_up, rail_right_up, self.cfg)

        if not person_detected:
            raw = 0.0
            return FallScore(
                score=self._smooth(raw),
                raw_score=raw,
                level="SAFE",
                status="NO_PERSON",
                rail_risk=0.0,
                zone_risk=0.0,
                pose_risk=0.0,
                detail={"reason": "person_not_detected"},
            )

        rail_risk, rails_down = compute_rail_risk(rail_states, edge_zone, self.cfg["rail_risk"])

        oob = self.cfg.get("out_of_bed", {})
        if oob.get("enabled", False) and self._out_of_bed_triggered(
            seg_attachment=seg_attachment,
            in_bed=in_bed,
            oob_cfg=oob,
        ):
            raw = float(oob.get("score", 85.0))
            smoothed = self._smooth(raw)
            return FallScore(
                score=smoothed,
                raw_score=raw,
                level=score_to_level(smoothed, levels),
                status="OUT_OF_BED",
                rail_risk=rail_risk,
                zone_risk=0.0,
                pose_risk=0.0,
                rails_down=rails_down,
                detail={"reason": "seg_off" if oob.get("trigger") == "seg_off" else "in_bed_false"},
            )

        if zone_risk is None:
            zone_risk = compute_zone_risk_band(zone_label, self.cfg["zone_risk"])
        pose_risk = compute_pose_risk(pose_ko, self.cfg["pose_risk"])

        raw = 100.0 * (
            self.w_rail * rail_risk + self.w_zone * zone_risk + self.w_pose * pose_risk
        )
        raw = float(max(0.0, min(100.0, raw)))
        smoothed = self._smooth(raw)

        return FallScore(
            score=smoothed,
            raw_score=raw,
            level=score_to_level(smoothed, levels),
            status="IN_BED",
            rail_risk=rail_risk,
            zone_risk=zone_risk,
            pose_risk=pose_risk,
            rails_down=rails_down,
            detail={
                "w_rail": self.w_rail,
                "w_zone": self.w_zone,
                "w_pose": self.w_pose,
                "rail_contrib": self.w_rail * rail_risk * 100,
                "zone_contrib": self.w_zone * zone_risk * 100,
                "pose_contrib": self.w_pose * pose_risk * 100,
            },
        )
