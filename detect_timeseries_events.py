#!/usr/bin/env python3
"""
Stage B+ — enriched CSV에서 침대 이탈·위험 구간 이벤트 자동 탐지 (배치 트리거).

입력: timeseries_enriched/{stem}.csv
출력: runs/timeseries_events/{stem}_events.json
      runs/timeseries_events/all_events.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bed_monitor.config import load_preset

EVENT_LABELS = {
    "left_bed": "out_bed_floor",
    "high_overflow": "unsafe_exit",
    "edge_fast": "edge_observe",
    "returned_bed": "in_bed_normal",
}


def estimate_sample_hz(df: pd.DataFrame) -> float:
    if "sample_hz" in df.columns and df["sample_hz"].notna().any():
        return float(df["sample_hz"].dropna().iloc[0])
    if len(df) < 2:
        return 1.0
    dts = df["timestamp_sec"].diff().dropna()
    dts = dts[dts > 1e-6]
    if len(dts) == 0:
        return 1.0
    return float(1.0 / dts.median())


def hold_rows(hold_sec: float, sample_hz: float) -> int:
    return max(1, int(round(hold_sec * sample_hz)))


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def detect_events(df: pd.DataFrame, preset: dict, video_file: str) -> list[dict]:
    ev_cfg = preset.get("events", {})
    hold_sec = float(ev_cfg.get("hold_sec", 0.3))
    overflow_min = float(ev_cfg.get("left_bed_overflow_min", 0.05))
    overflow_med = float(preset.get("risk_thresholds", {}).get("overflow_med", 0.15))
    edge_speed = float(ev_cfg.get("edge_speed_min_px_s", 30.0))

    hz = estimate_sample_hz(df)
    hold = hold_rows(hold_sec, hz)

    in_bed = df["in_bed"].fillna(False).astype(bool).to_numpy()
    overflow = df["limb_overflow_max"].fillna(0.0).to_numpy()
    edge_zone = df.get("edge_zone", pd.Series([None] * len(df))).tolist()
    speed = df.get("center_speed", pd.Series([0.0] * len(df))).fillna(0.0).to_numpy()
    ts = df["timestamp_sec"].to_numpy()

    events: list[dict] = []

    # left_bed: was in bed → out of bed
    prev_in = np.roll(in_bed, hold)
    prev_in[:hold] = in_bed[0]
    left_mask = prev_in & (~in_bed)
    for i0, i1 in contiguous_segments(left_mask):
        i0 = max(0, i0 - hold + 1)
        peak_ov = float(overflow[i0 : i1 + 1].max()) if i1 >= i0 else 0.0
        if peak_ov < overflow_min and not left_mask[i0 : i1 + 1].any():
            continue
        events.append(
            {
                "event_type": "left_bed",
                "event_label": EVENT_LABELS["left_bed"],
                "start_sec": float(ts[i0]),
                "end_sec": float(ts[i1]),
                "peak_overflow": round(peak_ov, 4),
                "trigger": "in_bed_false",
            }
        )

    # high_overflow sustained (침대 밖 limb)
    high_mask = overflow >= overflow_med
    for i0, i1 in contiguous_segments(high_mask):
        if (i1 - i0 + 1) < hold:
            continue
        events.append(
            {
                "event_type": "high_overflow",
                "event_label": EVENT_LABELS["high_overflow"],
                "start_sec": float(ts[i0]),
                "end_sec": float(ts[i1]),
                "peak_overflow": round(float(overflow[i0 : i1 + 1].max()), 4),
                "trigger": f"overflow>={overflow_med}",
            }
        )

    # edge + speed
    edge_mask = np.array(
        [(z in ("L", "R")) and (speed[i] >= edge_speed) for i, z in enumerate(edge_zone)],
        dtype=bool,
    )
    for i0, i1 in contiguous_segments(edge_mask):
        if (i1 - i0 + 1) < hold:
            continue
        events.append(
            {
                "event_type": "edge_fast",
                "event_label": EVENT_LABELS["edge_fast"],
                "start_sec": float(ts[i0]),
                "end_sec": float(ts[i1]),
                "peak_speed": round(float(speed[i0 : i1 + 1].max()), 2),
                "trigger": f"edge_zone L/R & speed>={edge_speed}",
            }
        )

    # returned_bed
    prev_out = np.roll(~in_bed, hold)
    prev_out[:hold] = ~in_bed[0]
    ret_mask = prev_out & in_bed
    for i0, i1 in contiguous_segments(ret_mask):
        events.append(
            {
                "event_type": "returned_bed",
                "event_label": EVENT_LABELS["returned_bed"],
                "start_sec": float(ts[i0]),
                "end_sec": float(ts[i1]),
                "trigger": "in_bed_true",
            }
        )

    for e in events:
        e["video_file"] = video_file
        e["hold_sec"] = hold_sec
        e["sample_hz"] = hz
    events.sort(key=lambda x: x["start_sec"])
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect bed-exit events from enriched CSV")
    ap.add_argument("--in-dir", type=Path, required=True, help="timeseries_*_enriched/")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/dmc/AI/DMC_POSE/runs/timeseries_events"),
    )
    ap.add_argument("--preset", type=Path, default=None)
    ap.add_argument("--video", type=str, default=None)
    args = ap.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = load_preset(args.preset)

    csv_files = sorted(in_dir.glob("*.csv"))
    if args.video:
        stem = Path(args.video).stem
        csv_files = [in_dir / f"{stem}.csv"]

    all_events: list[dict] = []
    per_video = []
    t0 = time.time()

    for csv_path in csv_files:
        if csv_path.name in ("enrich_index.json",):
            continue
        df = pd.read_csv(csv_path)
        if "in_bed" not in df.columns:
            print(f"skip {csv_path.name}: not enriched (missing in_bed)")
            continue
        video_name = str(df["video_file"].iloc[0]) if "video_file" in df.columns and len(df) else f"{csv_path.stem}.mp4"
        events = detect_events(df, preset, video_name)
        out_path = out_dir / f"{csv_path.stem}_events.json"
        payload = {
            "video_file": video_name,
            "source_csv": str(csv_path),
            "event_count": len(events),
            "events": events,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        per_video.append({"video": csv_path.stem, "events": len(events), "path": str(out_path)})
        all_events.extend(events)
        print(f"{csv_path.stem}: {len(events)} events")

    summary = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.time() - t0, 2),
        "total_events": len(all_events),
        "videos": per_video,
        "events": all_events,
    }
    (out_dir / "all_events.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"→ {out_dir / 'all_events.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
