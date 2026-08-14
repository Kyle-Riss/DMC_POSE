"""Live segment event detection (same rules as detect_timeseries_events.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LiveEventTracker:
    preset: dict
    hold_sec: float = 0.3
    overflow_med: float = 0.15
    edge_speed_min: float = 30.0
    _stable_in_since: float | None = field(default=None, init=False)
    _stable_out_since: float | None = field(default=None, init=False)
    _was_in_stable: bool = field(default=False, init=False)
    _overflow_high_since: float | None = field(default=None, init=False)
    _edge_fast_since: float | None = field(default=None, init=False)
    active_events: set[str] = field(default_factory=set, init=False)
    last_event: str | None = field(default=None, init=False)
    last_event_at: datetime | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        ev = self.preset.get("events", {})
        th = self.preset.get("risk_thresholds", {})
        self.hold_sec = float(ev.get("hold_sec", 0.3))
        self.overflow_med = float(th.get("overflow_med", 0.15))
        self.edge_speed_min = float(ev.get("edge_speed_min_px_s", 30.0))

    def update(self, t_sec: float, feat: dict) -> dict[str, bool]:
        overflow = float(feat.get("limb_overflow_max") or 0.0)
        edge = feat.get("edge_zone")
        speed = float(feat.get("center_speed") or 0.0)
        person = bool(feat.get("person_detected"))

        fired: dict[str, bool] = {
            "left_bed": False,
            "high_overflow": False,
            "edge_fast": False,
        }
        self.active_events.clear()

        if not person:
            self._stable_in_since = None
            self._stable_out_since = None
            self._was_in_stable = False
            self._overflow_high_since = None
            self._edge_fast_since = None
            return fired

        # --- seg attachment (on_seg / partial → off_seg) ---
        attachment = feat.get("seg_attachment", "none")
        on_bed = attachment in ("on_seg", "partial")

        if on_bed:
            if self._stable_in_since is None:
                self._stable_in_since = t_sec
            if (t_sec - self._stable_in_since) >= self.hold_sec:
                self._was_in_stable = True
            self._stable_out_since = None
        elif attachment == "off_seg":
            if self._stable_out_since is None:
                self._stable_out_since = t_sec
            if (
                self._was_in_stable
                and self._stable_out_since is not None
                and (t_sec - self._stable_out_since) >= self.hold_sec
            ):
                fired["left_bed"] = True
                self.active_events.add("left_bed")
            self._stable_in_since = None
        else:
            self._stable_in_since = None
            self._stable_out_since = None

        # --- overflow (손발 bbox 밖) ---
        if overflow >= self.overflow_med:
            if self._overflow_high_since is None:
                self._overflow_high_since = t_sec
            if (t_sec - self._overflow_high_since) >= self.hold_sec:
                fired["high_overflow"] = True
                self.active_events.add("high_overflow")
        else:
            self._overflow_high_since = None

        # --- edge + speed ---
        if edge in ("L", "R") and speed >= self.edge_speed_min:
            if self._edge_fast_since is None:
                self._edge_fast_since = t_sec
            if (t_sec - self._edge_fast_since) >= self.hold_sec:
                fired["edge_fast"] = True
                self.active_events.add("edge_fast")
        else:
            self._edge_fast_since = None

        for name, is_fired in fired.items():
            if is_fired:
                self.last_event = name
                self.last_event_at = datetime.now()

        return fired

    @property
    def bed_event(self) -> str | None:
        if not self.active_events:
            return None
        for key in ("left_bed", "high_overflow", "edge_fast"):
            if key in self.active_events:
                return key
        return next(iter(self.active_events))
