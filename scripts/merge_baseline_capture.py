#!/usr/bin/env python3
"""
Phase M0 — pose-sixclass vs fall_monitor baseline capture.

Same frame(s) through both pipelines; writes JSONL + summary for MERGE_BASELINE.md.

  # Static RTSP frames (offline)
  python scripts/merge_baseline_capture.py --images bed_seg/rtsp_raw/*.jpg

  # Live RTSP (both pipelines on same frames)
  python scripts/merge_baseline_capture.py --rtsp rtsp://192.168.0.161:8554/stream --duration 120

  # Poll running server /status only (pose-sixclass live path)
  python scripts/merge_baseline_capture.py --status-url http://127.0.0.1:8000/status --duration 120
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imutils
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HOME))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from bed_monitor.bed_zone import build_approx_bed_zone
from bed_monitor.config import load_preset
from bed_monitor.features import MotionState
from bed_monitor.live import apply_fall_scoring, enrich_from_keypoints
from bed_monitor.scoring import FallScorer
from bed_monitor.temporal import LiveEventTracker
from bed_roi.roi_utils import apply_bed_roi, load_bed_roi
from rail.rail_detect import detect_both_rails, load_rail_config
from server import (
    BED_SEG_CONF,
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    FRAME_WIDTH,
    POSE_KERAS_MODEL,
    SEG_EVERY_N,
    YOLO_DEVICE,
    YOLO_POSE_WEIGHT,
    YOLO_SEG_CLASS,
    YOLO_SEG_WEIGHT,
    extract_bed_detection,
)
from ultralytics import YOLO

import keras


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pose_six_models():
    yolo_seg = YOLO(YOLO_SEG_WEIGHT)
    yolo_pose = YOLO(YOLO_POSE_WEIGHT)
    keras_clf = keras.models.load_model(POSE_KERAS_MODEL)
    return yolo_seg, yolo_pose, keras_clf


def load_fall_monitor():
    from fall_monitor.config import load_config
    from fall_monitor.pipeline import FallMonitor

    return FallMonitor(load_config())


def process_pose_six(
    frame: np.ndarray,
    *,
    yolo_seg,
    yolo_pose,
    keras_clf,
    preset: dict,
    motion_state: MotionState,
    event_tracker: LiveEventTracker,
    rail_cfg: dict | None,
    roi_bbox,
    t_sec: float,
    cached_bed: dict | None,
    frame_idx: int,
    prev_rail: tuple[bool, bool],
    fall_scorer: FallScorer | None = None,
) -> tuple[dict, dict | None]:
    fh, fw = frame.shape[:2]
    bed = cached_bed
    if frame_idx % SEG_EVERY_N == 1 or bed is None:
        seg_res = yolo_seg.predict(
            frame,
            classes=[YOLO_SEG_CLASS],
            conf=BED_SEG_CONF,
            device=YOLO_DEVICE,
            verbose=False,
        )
        fresh = extract_bed_detection(seg_res[0], fh, fw)
        if fresh.get("bbox") is not None or fresh.get("mask") is not None:
            bed = fresh
    bed = bed or {"mask": None, "bbox": None, "source": "none"}
    if roi_bbox is not None:
        bed = apply_bed_roi(bed, roi_bbox, fh, fw)
    if preset.get("bed_zone"):
        bed = build_approx_bed_zone(bed, roi_bbox, fh, fw, preset)

    out: dict = {
        "person_detected": False,
        "in_bed": "NO",
        "seg_attachment": "none",
        "zone_quality": bed.get("zone_quality", "none"),
        "bed_source": bed.get("source", "none"),
        "kpt_on_seg_ratio": 0.0,
        "edge_zone": None,
        "limb_overflow_max": 0.0,
        "risk_level": "SAFE",
        "center_speed": None,
        "pose": "None",
        "pose_conf": 0.0,
        "rail_left_up": False,
        "rail_right_up": False,
        "bed_event": None,
        "fall_score": 0.0,
        "fall_level": "SAFE",
        "fall_status": "IN_BED",
        "rail_risk": 0.0,
        "zone_risk": 0.0,
        "pose_risk": 0.0,
    }

    pose_res = yolo_pose.predict(frame, conf=0.5, device=YOLO_DEVICE, verbose=False)
    person_bbox = None
    kxy = None
    kconf = None
    if len(pose_res) > 0 and len(pose_res[0].keypoints) > 0:
        kp = pose_res[0].keypoints[0]
        kxy = kp.xy.cpu().numpy().reshape(-1, 2)
        if kp.conf is not None:
            kconf = kp.conf.cpu().numpy().flatten()
        else:
            kconf = np.ones(len(kxy), dtype=np.float32)

        feat = enrich_from_keypoints(kxy, kconf, bed, motion_state, t_sec, preset)
        out.update(
            {
                "person_detected": bool(feat["person_detected"]),
                "in_bed": "YES" if feat["in_bed"] else "NO",
                "seg_attachment": feat["seg_attachment"],
                "zone_quality": feat.get("zone_quality", out["zone_quality"]),
                "bed_source": feat.get("bed_source", out["bed_source"]),
                "kpt_on_seg_ratio": float(feat["kpt_on_seg_ratio"]),
                "edge_zone": feat["edge_zone"],
                "limb_overflow_max": float(feat["limb_overflow_max"]),
                "risk_level": feat["risk_level"],
                "center_speed": feat.get("center_speed"),
            }
        )
        events = event_tracker.update(t_sec, feat)
        if events.get("left_bed"):
            out["bed_event"] = "left_bed"
        elif events.get("high_overflow"):
            out["bed_event"] = "high_overflow"
        elif events.get("edge_fast"):
            out["bed_event"] = "edge_fast"

        if pose_res[0].boxes is not None and len(pose_res[0].boxes) > 0:
            person_bbox = pose_res[0].boxes.xyxy[0].cpu().numpy()

        kpts_flat = kxy.flatten()
        if len(kpts_flat) == 34:
            pred = keras_clf.predict(kpts_flat.reshape(1, -1).astype("float32"), verbose=0)
            idx = int(np.argmax(pred))
            out["pose"] = CLASS_NAMES[idx]
            out["pose_conf"] = float(pred[0][idx])

    if rail_cfg is not None:
        rr = detect_both_rails(
            frame,
            rail_cfg,
            person_xyxy=person_bbox,
            prev_left=prev_rail[0],
            prev_right=prev_rail[1],
            pose_label=out["pose"] if out["pose"] != "None" else None,
        )
        out["rail_left_up"] = bool(rr["rail_left_up"])
        out["rail_right_up"] = bool(rr["rail_right_up"])

    if fall_scorer is not None and kxy is not None and kconf is not None:
        bed_bbox = bed.get("bbox")
        feat = {
            "person_detected": out["person_detected"],
            "in_bed": out["in_bed"] == "YES",
            "seg_attachment": out["seg_attachment"],
            "edge_zone": out["edge_zone"],
        }
        pose_ko = out["pose"] if out["pose"] != "None" else None
        apply_fall_scoring(
            feat,
            kxy,
            kconf,
            bed_bbox,
            preset,
            fall_scorer,
            pose_ko=pose_ko if out["pose"] != "None" else None,
            rail_left_up=out["rail_left_up"],
            rail_right_up=out["rail_right_up"],
        )
        for key in ("fall_score", "fall_level", "fall_status", "rail_risk", "zone_risk", "pose_risk"):
            out[key] = feat.get(key, out[key])

    return out, bed


def process_fall_monitor(frame: np.ndarray, monitor) -> dict:
    r = monitor.process(frame)
    f = r.fall
    rails_down = list(f.rails_down) if f.rails_down else []
    return {
        "fall_score": round(float(f.score), 2),
        "fall_level": f.level,
        "fall_status": f.status,
        "rail_risk": round(float(f.rail_risk), 4),
        "zone_risk": round(float(f.zone_risk), 4),
        "pose_risk": round(float(f.pose_risk), 4),
        "in_bed": bool(r.zone.in_bed),
        "zone": r.zone.zone,
        "pose_ko": r.pose_ko,
        "pose_conf": round(float(r.pose_conf), 4),
        "person_detected": r.person is not None,
        "rails_down": rails_down,
        "roi_ok": r.roi_pts is not None,
    }


def compare_row(ps: dict, fm: dict) -> dict:
    ps_in = ps.get("in_bed") == "YES"
    fm_in = bool(fm.get("in_bed"))
    ps_person = bool(ps.get("person_detected"))
    fm_person = bool(fm.get("person_detected"))
    return {
        "in_bed_agree": ps_in == fm_in,
        "person_agree": ps_person == fm_person,
        "in_bed_pose_six": ps_in,
        "in_bed_fall_monitor": fm_in,
        "seg_attachment": ps.get("seg_attachment"),
        "overflow": ps.get("limb_overflow_max"),
        "risk_level": ps.get("risk_level"),
        "fall_score": fm.get("fall_score"),
        "fall_score_pose_six": ps.get("fall_score"),
        "fall_level": fm.get("fall_level"),
        "fall_level_pose_six": ps.get("fall_level"),
        "fall_status": fm.get("fall_status"),
        "fall_status_pose_six": ps.get("fall_status"),
        "pose_pose_six": ps.get("pose"),
        "pose_fall_monitor": fm.get("pose_ko"),
        "pose_agree": ps.get("pose") == fm.get("pose_ko"),
        "zone_fall_monitor": fm.get("zone"),
        "edge_zone_pose_six": ps.get("edge_zone"),
        "zone_quality": ps.get("zone_quality"),
        "bed_event": ps.get("bed_event"),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"frames": 0}
    agree_in = sum(1 for r in rows if r["compare"]["in_bed_agree"])
    agree_person = sum(1 for r in rows if r["compare"]["person_agree"])
    agree_pose = sum(1 for r in rows if r["compare"]["pose_agree"])
    fm_oob = sum(1 for r in rows if r["fall_monitor"].get("fall_status") == "OUT_OF_BED")
    ps_off = sum(1 for r in rows if r["pose_sixclass"].get("seg_attachment") == "off_seg")
    events = sum(1 for r in rows if r["pose_sixclass"].get("bed_event"))
    scores_fm = [r["fall_monitor"]["fall_score"] for r in rows if r["fall_monitor"].get("person_detected")]
    scores_ps = [r["pose_sixclass"]["fall_score"] for r in rows if r["pose_sixclass"].get("person_detected")]
    status_agree = sum(
        1
        for r in rows
        if r["pose_sixclass"].get("person_detected")
        and r["fall_monitor"].get("person_detected")
        and r["pose_sixclass"].get("fall_status") == r["fall_monitor"].get("fall_status")
    )
    status_n = sum(
        1
        for r in rows
        if r["pose_sixclass"].get("person_detected") and r["fall_monitor"].get("person_detected")
    )
    return {
        "frames": n,
        "in_bed_agreement_pct": round(100.0 * agree_in / n, 1),
        "person_agreement_pct": round(100.0 * agree_person / n, 1),
        "pose_agreement_pct": round(100.0 * agree_pose / n, 1),
        "fall_monitor_out_of_bed_frames": fm_oob,
        "pose_six_off_seg_frames": ps_off,
        "pose_six_bed_event_frames": events,
        "fall_score_min": min(scores_fm) if scores_fm else None,
        "fall_score_max": max(scores_fm) if scores_fm else None,
        "fall_score_mean": round(sum(scores_fm) / len(scores_fm), 2) if scores_fm else None,
        "fall_score_pose_six_min": min(scores_ps) if scores_ps else None,
        "fall_score_pose_six_max": max(scores_ps) if scores_ps else None,
        "fall_score_pose_six_mean": round(sum(scores_ps) / len(scores_ps), 2) if scores_ps else None,
        "fall_status_agreement_pct": round(100.0 * status_agree / status_n, 1) if status_n else None,
    }


def run_images(paths: list[Path], out_dir: Path) -> dict:
    preset = load_preset()
    rail_cfg = load_rail_config()
    roi_path = ROOT / "bed_roi" / "bed_roi.json"

    yolo_seg, yolo_pose, keras_clf = load_pose_six_models()
    fm = load_fall_monitor()
    scoring_cfg = preset.get("scoring") or {}
    fall_scorer = FallScorer(scoring_cfg) if scoring_cfg.get("enabled") else None

    motion = MotionState()
    tracker = LiveEventTracker(preset)
    cached_bed = None
    prev_rail = (False, False)
    rows: list[dict] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "baseline_frames.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for i, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            frame = imutils.resize(frame, width=FRAME_WIDTH)
            fh, fw = frame.shape[:2]
            roi_bbox = load_bed_roi(roi_path, fw, fh)

            ps, cached_bed = process_pose_six(
                frame,
                yolo_seg=yolo_seg,
                yolo_pose=yolo_pose,
                keras_clf=keras_clf,
                preset=preset,
                motion_state=motion,
                event_tracker=tracker,
                rail_cfg=rail_cfg,
                roi_bbox=roi_bbox,
                t_sec=float(i) * 0.1,
                cached_bed=cached_bed,
                frame_idx=i + 1,
                prev_rail=prev_rail,
                fall_scorer=fall_scorer,
            )
            prev_rail = (ps["rail_left_up"], ps["rail_right_up"])
            fm_out = process_fall_monitor(frame, fm)
            row = {
                "ts": _utc_now(),
                "source": str(path),
                "pose_sixclass": ps,
                "fall_monitor": fm_out,
                "compare": compare_row(ps, fm_out),
            }
            rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary.update(
        {
            "mode": "images",
            "image_count": len(paths),
            "processed": len(rows),
            "rtsp_url": None,
            "preset": preset.get("preset_id", "default"),
            "captured_at": _utc_now(),
        }
    )
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_rtsp(url: str, duration: float, out_dir: Path, sample_hz: float = 2.0) -> dict:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise SystemExit(f"RTSP open failed: {url}")

    preset = load_preset()
    rail_cfg = load_rail_config()
    roi_path = ROOT / "bed_roi" / "bed_roi.json"

    yolo_seg, yolo_pose, keras_clf = load_pose_six_models()
    fm = load_fall_monitor()
    scoring_cfg = preset.get("scoring") or {}
    fall_scorer = FallScorer(scoring_cfg) if scoring_cfg.get("enabled") else None

    motion = MotionState()
    tracker = LiveEventTracker(preset)
    cached_bed = None
    prev_rail = (False, False)
    rows: list[dict] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "baseline_rtsp.jsonl"
    interval = 1.0 / max(sample_hz, 0.1)
    t_end = time.time() + duration
    frame_idx = 0
    next_sample = time.time()

    with jsonl_path.open("w", encoding="utf-8") as fp:
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            frame_idx += 1
            now = time.time()
            if now < next_sample:
                continue
            next_sample = now + interval

            frame = imutils.resize(frame, width=FRAME_WIDTH)
            fh, fw = frame.shape[:2]
            roi_bbox = load_bed_roi(roi_path, fw, fh)
            t_sec = now - (t_end - duration)

            ps, cached_bed = process_pose_six(
                frame,
                yolo_seg=yolo_seg,
                yolo_pose=yolo_pose,
                keras_clf=keras_clf,
                preset=preset,
                motion_state=motion,
                event_tracker=tracker,
                rail_cfg=rail_cfg,
                roi_bbox=roi_bbox,
                t_sec=t_sec,
                cached_bed=cached_bed,
                frame_idx=frame_idx,
                prev_rail=prev_rail,
                fall_scorer=fall_scorer,
            )
            prev_rail = (ps["rail_left_up"], ps["rail_right_up"])
            fm_out = process_fall_monitor(frame, fm)
            row = {
                "ts": _utc_now(),
                "source": url,
                "frame_idx": frame_idx,
                "pose_sixclass": ps,
                "fall_monitor": fm_out,
                "compare": compare_row(ps, fm_out),
            }
            rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{len(rows):4d}] ps in_bed={ps['in_bed']} attach={ps['seg_attachment']} "
                f"ps_score={ps.get('fall_score')} fm_score={fm_out['fall_score']} "
                f"fm_status={fm_out['fall_status']}"
            )

    cap.release()
    summary = summarize(rows)
    summary.update(
        {
            "mode": "rtsp",
            "duration_sec": duration,
            "sample_hz": sample_hz,
            "samples": len(rows),
            "rtsp_url": url,
            "preset": preset.get("preset_id", "default"),
            "captured_at": _utc_now(),
        }
    )
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_status_poll(url: str, duration: float, sample_hz: float, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "baseline_status.jsonl"
    interval = 1.0 / max(sample_hz, 0.1)
    t_end = time.time() + duration
    rows: list[dict] = []
    next_sample = time.time()

    with jsonl_path.open("w", encoding="utf-8") as fp:
        while time.time() < t_end:
            now = time.time()
            if now < next_sample:
                time.sleep(0.02)
                continue
            next_sample = now + interval
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                print(f"poll error: {exc}")
                continue
            row = {"ts": _utc_now(), "status": data}
            rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{len(rows):4d}] in_bed={data.get('in_bed')} attach={data.get('seg_attachment')} "
                f"risk={data.get('risk_level')} event={data.get('bed_event')}"
            )

    summary = {
        "mode": "status_poll",
        "url": url,
        "duration_sec": duration,
        "samples": len(rows),
        "captured_at": _utc_now(),
    }
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase M0 merge baseline capture")
    ap.add_argument("--images", nargs="*", type=Path, help="Image paths or globs")
    ap.add_argument("--image-dir", type=Path, help="Directory of jpg/png frames")
    ap.add_argument("--rtsp", type=str, default=None, help="RTSP URL for dual-pipeline capture")
    ap.add_argument("--status-url", type=str, default=None, help="Poll pose-sixclass /status")
    ap.add_argument("--duration", type=float, default=120.0, help="Seconds for rtsp/status modes")
    ap.add_argument("--sample-hz", type=float, default=2.0, help="Sample rate for rtsp/status")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "runs" / "merge_baseline",
        help="Output directory",
    )
    ap.add_argument("--max-images", type=int, default=30, help="Cap images processed")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.images:
        for p in args.images:
            if "*" in str(p):
                paths.extend(sorted(Path().glob(str(p))))
            else:
                paths.append(p)
    if args.image_dir and args.image_dir.is_dir():
        paths.extend(sorted(args.image_dir.glob("*.jpg")))
        paths.extend(sorted(args.image_dir.glob("*.png")))
    paths = [p.resolve() for p in paths if p.is_file()][: args.max_images]

    if args.status_url:
        summary = run_status_poll(args.status_url, args.duration, args.sample_hz, args.out)
    elif args.rtsp:
        summary = run_rtsp(args.rtsp, args.duration, args.out, args.sample_hz)
    elif paths:
        summary = run_images(paths, args.out)
    else:
        default_dir = ROOT / "bed_seg" / "rtsp_raw"
        if default_dir.is_dir():
            paths = sorted(default_dir.glob("*.jpg"))[: args.max_images]
            summary = run_images(paths, args.out)
        else:
            ap.error("No input: use --images, --image-dir, --rtsp, or --status-url")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
