from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PRESET = Path(__file__).resolve().parents[1] / "config" / "presets" / "default.json"


def load_preset(path: Path | None = None) -> dict:
    import os

    if path is None:
        env = os.environ.get("POSE_PRESET", "").strip()
        if env:
            p = Path(env)
            if not p.is_file():
                p = DEFAULT_PRESET.parent / f"{env}.json"
            path = p
    p = path or DEFAULT_PRESET
    if not p.is_file():
        return {
            "risk_thresholds": {
                "overflow_low": 0.05,
                "overflow_med": 0.15,
                "overflow_high": 0.25,
            },
            "motion": {"ema_alpha": 0.3, "hold_sec": 0.5},
            "inference": {"kpt_conf": 0.3, "bed_seg_conf": 0.01},
            "events": {
                "hold_sec": 0.3,
                "left_bed_overflow_min": 0.05,
            },
        }
    return json.loads(p.read_text(encoding="utf-8"))
