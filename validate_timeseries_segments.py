#!/usr/bin/env python3
"""
Compare human GT segment_events.json vs auto-detected timeseries events.

GT:   config/segment_events.json (or --gt)
Pred: runs/timeseries_events/{stem}_events.json from detect_timeseries_events.py

Overlap: IoU on time axis >= --iou-min counts as match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# GT label → predicted event_type(s)
LABEL_TO_PRED: dict[str, list[str]] = {
    "out_bed_floor": ["left_bed"],
    "out_bed_stand": ["left_bed"],
    "unsafe_exit": ["left_bed", "high_overflow", "edge_fast"],
    "exit_normal": ["left_bed"],
    "edge_observe": ["edge_fast", "high_overflow"],
    "in_bed_normal": [],
}


def time_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def stem_from_video(name: str) -> str:
    return Path(name).stem


def load_pred_events(pred_dir: Path, video_file: str) -> list[dict]:
    stem = stem_from_video(video_file)
    path = pred_dir / f"{stem}_events.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("events", data if isinstance(data, list) else [])


def match_segments(
    gt: dict,
    preds: list[dict],
    iou_min: float,
) -> tuple[bool, dict | None]:
    g0, g1 = float(gt["start_sec"]), float(gt["end_sec"])
    label = gt.get("event_label", "unknown")
    want = LABEL_TO_PRED.get(label, [])

    best: dict | None = None
    best_iou = 0.0
    for p in preds:
        et = p.get("event_type", "")
        if want and et not in want:
            continue
        p0, p1 = float(p["start_sec"]), float(p["end_sec"])
        iou = time_iou(g0, g1, p0, p1)
        if iou > best_iou:
            best_iou = iou
            best = p

    if best is None and not want:
        return True, None  # normal segment expects no event
    return best_iou >= iou_min, best


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate rule events vs GT segments")
    ap.add_argument(
        "--gt",
        type=Path,
        default=Path("config/segment_events.json"),
        help="Ground truth segment_events.json",
    )
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("runs/timeseries_events"),
        help="Folder with {stem}_events.json",
    )
    ap.add_argument("--iou-min", type=float, default=0.2, help="min temporal IoU for match")
    ap.add_argument("--out", type=Path, default=Path("runs/timeseries_events/validation_report.json"))
    args = ap.parse_args()

    if not args.gt.is_file():
        print(f"GT file missing: {args.gt}")
        print("  Add labels: python label_segment_event.py --video 'Raw0 (3).mp4' --start 120 --end 145 --label out_bed_floor")
        return 1

    gt_data = json.loads(args.gt.read_text(encoding="utf-8"))
    segments = gt_data.get("segments", [])

    results: list[dict] = []
    tp = fp = fn = 0

    for seg in segments:
        video = seg.get("video_file", "")
        preds = load_pred_events(args.pred_dir, video)
        ok, matched = match_segments(seg, preds, args.iou_min)
        label = seg.get("event_label", "unknown")
        want = LABEL_TO_PRED.get(label, [])

        if want:
            if ok:
                tp += 1
                status = "TP"
            else:
                fn += 1
                status = "FN"
        else:
            status = "OK" if ok else "FP?"
            if not ok and matched is not None:
                fp += 1

        results.append(
            {
                "status": status,
                "video_file": video,
                "gt_label": label,
                "gt_start": seg.get("start_sec"),
                "gt_end": seg.get("end_sec"),
                "matched_event": matched,
            }
        )

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    report = {
        "gt_file": str(args.gt),
        "pred_dir": str(args.pred_dir),
        "iou_min": args.iou_min,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "gt_segments": len(segments)},
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tp": tp, "fp": fp, "fn": fn, "precision": report["precision"], "recall": report["recall"]}, indent=2))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
