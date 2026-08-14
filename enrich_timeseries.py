#!/usr/bin/env python3
"""
Stage B — v1 timeseries CSV → v2 enriched CSV (bed + motion + overflow).

입력: timeseries/{stem}.csv  (extract_raw_timeseries.py)
출력: timeseries_enriched/{stem}.csv

침대 bbox:
  --bed-cache-dir 에 {stem}.json 있으면 사용
  --raw-root + MP4 있으면 영상 첫 프레임 seg 1회 후 캐시
  없으면 bed_roi.json 고정 bbox (fallback)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from bed_monitor.bed_detect import bed_to_jsonable, detect_bed_from_video
from bed_monitor.bed_zone import build_approx_bed_zone
from bed_monitor.config import load_preset
from bed_monitor.features import MotionState
from bed_monitor.geometry import kpts_to_xy_conf
from bed_monitor.live import enrich_from_keypoints
from bed_monitor.scoring import FallScorer
from bed_roi.roi_utils import apply_bed_roi, load_bed_roi

PIPELINE_VERSION = "enrich_v0.2"


def load_bed_cache(cache_path: Path) -> dict | None:
    if not cache_path.is_file():
        return None
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    bbox = data.get("bbox")
    if bbox is None:
        return None
    return {
        "bbox": tuple(int(v) for v in bbox),
        "mask": None,
        "source": data.get("source", "cache"),
    }


def roi_fallback_bed(roi_path: Path, w: int, h: int) -> dict:
    roi = load_bed_roi(roi_path, w, h)
    if roi is None:
        return {"bbox": None, "mask": None, "source": "none"}
    return apply_bed_roi({"bbox": None, "mask": None, "source": "none"}, roi, h, w)


def resolve_bed_for_video(
    stem: str,
    video_path: Path | None,
    seg_model: YOLO | None,
    cache_dir: Path,
    roi_path: Path,
    preset: dict,
    device: str,
    force: bool,
) -> dict:
    cache_path = cache_dir / f"{stem}.json"
    if cache_path.is_file() and not force:
        cached = load_bed_cache(cache_path)
        if cached is not None:
            return cached

    w = int(preset["inference"].get("resize_width", 640))
    h = int(w * 360 / 640)

    if video_path is not None and video_path.is_file() and seg_model is not None:
        bed = detect_bed_from_video(
            video_path,
            seg_model,
            device=device,
            seg_conf=float(preset["inference"].get("bed_seg_conf", 0.01)),
            resize_width=w,
        )
        roi = load_bed_roi(roi_path, bed.get("frame_width", w), bed.get("frame_height", h))
        if roi is not None:
            bed = apply_bed_roi(bed, roi, bed.get("frame_height", h), bed.get("frame_width", w))
        if preset.get("bed_zone"):
            bed = build_approx_bed_zone(
                bed, roi, bed.get("frame_height", h), bed.get("frame_width", w), preset
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(bed_to_jsonable(bed), indent=2), encoding="utf-8")
        return bed

    bed = roi_fallback_bed(roi_path, w, h)
    bed["frame_width"] = w
    bed["frame_height"] = h
    return bed


def enrich_csv(
    csv_path: Path,
    out_path: Path,
    bed: dict,
    preset: dict,
    sample_hz: float,
) -> int:
    df = pd.read_csv(csv_path)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return 0

    bed_bbox = bed.get("bbox")
    bed_mask = bed.get("mask")
    state = MotionState()
    scoring_cfg = preset.get("scoring") or {}
    fall_scorer = FallScorer(scoring_cfg) if scoring_cfg.get("enabled") else None

    rows_out: list[dict] = []
    for _, row in df.iterrows():
        r = row.to_dict()
        xy, conf = kpts_to_xy_conf(r)
        t_sec = float(r.get("timestamp_sec") or 0.0)
        pose_ko = r.get("pose_class")
        if pose_ko is not None and pd.isna(pose_ko):
            pose_ko = None
        feat = enrich_from_keypoints(
            xy,
            conf,
            bed,
            state,
            t_sec,
            preset,
            pose_ko=str(pose_ko) if pose_ko else None,
            scorer=fall_scorer,
        )

        r["coord_space"] = "image"
        r["bed_seg_ok"] = bed_bbox is not None
        r["bed_source"] = bed.get("source", "none")
        if bed_bbox is not None:
            x0, y0, x1, y1 = bed_bbox
            r["bed_bbox_x0"], r["bed_bbox_y0"], r["bed_bbox_x1"], r["bed_bbox_y1"] = x0, y0, x1, y1
        else:
            r["bed_bbox_x0"] = r["bed_bbox_y0"] = r["bed_bbox_x1"] = r["bed_bbox_y1"] = None
        r["person_detected"] = feat["person_detected"]
        r["zone_quality"] = feat.get("zone_quality", "none")
        r["seg_attachment"] = feat["seg_attachment"]
        r["kpt_on_seg_ratio"] = round(feat["kpt_on_seg_ratio"], 6)
        r["limbs_outside_seg"] = feat["limbs_outside_seg"]
        r["in_bed"] = feat["in_bed"]
        r["in_bed_method"] = feat["in_bed_method"]
        r["edge_zone"] = feat["edge_zone"]
        r["limb_overflow_max"] = round(feat["limb_overflow_max"], 6)
        r["risk_level"] = feat["risk_level"]
        r["center_x"] = feat["center_x"]
        r["center_y"] = feat["center_y"]
        r["center_vx"] = state.center_vx
        r["center_vy"] = state.center_vy
        r["center_speed"] = feat["center_speed"]
        r["center_accel_like"] = state.center_accel_like
        r["fall_score"] = feat.get("fall_score", 0.0)
        r["fall_level"] = feat.get("fall_level", "SAFE")
        r["fall_status"] = feat.get("fall_status", "NO_PERSON")
        r["rail_risk"] = feat.get("rail_risk", 0.0)
        r["zone_risk"] = feat.get("zone_risk", 0.0)
        r["pose_risk"] = feat.get("pose_risk", 0.0)
        r["sample_hz"] = sample_hz
        r["pipeline_version"] = PIPELINE_VERSION
        rows_out.append(r)

    out_df = pd.DataFrame(rows_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return len(out_df)


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich v1 timeseries CSV → v2")
    ap.add_argument("--in-dir", type=Path, required=True, help="v1 CSV folder (timeseries or timeseries_10hz)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: {in-dir}_enriched")
    ap.add_argument("--raw-root", type=Path, default=None, help="Raw_data root for MP4 + bed seg cache")
    ap.add_argument("--bed-cache-dir", type=Path, default=None)
    ap.add_argument("--bed-roi", type=Path, default=Path("/home/dmc/AI/DMC_POSE/bed_roi/bed_roi.json"))
    ap.add_argument("--preset", type=Path, default=None)
    ap.add_argument("--sample-hz", type=float, default=None, help="override; else from timeseries_index.json")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--seg-weights", type=Path, default=Path("/home/dmc/AI/DMC_POSE/yolo11n-bed-seg.pt"))
    ap.add_argument("--no-video", action="store_true", help="MP4 seg skip — ROI/cache only")
    ap.add_argument("--force-bed", action="store_true")
    ap.add_argument("--video", type=str, default=None, help="single stem mp4 name")
    args = ap.parse_args()

    in_dir = args.in_dir.resolve()
    out_dir = (args.out_dir or Path(str(in_dir) + "_enriched")).resolve()
    cache_dir = (args.bed_cache_dir or (args.raw_root / "meta" / "bed_cache" if args.raw_root else in_dir / "bed_cache")).resolve()
    preset = load_preset(args.preset)

    sample_hz = args.sample_hz
    index_path = in_dir / "timeseries_index.json"
    if sample_hz is None and index_path.is_file():
        sample_hz = float(json.loads(index_path.read_text()).get("sample_hz", 1.0))
    if sample_hz is None:
        sample_hz = 1.0

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    seg_model = None
    if not args.no_video and args.raw_root is not None:
        import torch

        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for bed seg; use --no-video for ROI-only")
        seg_model = YOLO(str(args.seg_weights))

    video_dir = args.raw_root / "video" if args.raw_root else None
    csv_files = sorted(in_dir.glob("*.csv"))
    if args.video:
        stem = Path(args.video).stem
        csv_files = [in_dir / f"{stem}.csv"]
        if not csv_files[0].is_file():
            raise FileNotFoundError(csv_files[0])

    t0 = time.time()
    results = []
    for csv_path in csv_files:
        stem = csv_path.stem
        vp = video_dir / f"{stem}.mp4" if video_dir is not None else None
        bed = resolve_bed_for_video(
            stem,
            vp,
            seg_model,
            cache_dir,
            args.bed_roi.resolve(),
            preset,
            args.device,
            args.force_bed,
        )
        out_csv = out_dir / csv_path.name
        n = enrich_csv(csv_path, out_csv, bed, preset, sample_hz)
        results.append(
            {
                "video": stem,
                "rows": n,
                "bed_source": bed.get("source"),
                "bed_bbox": bed.get("bbox"),
                "out": str(out_csv),
            }
        )
        print(results[-1])

    index = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "sample_hz": sample_hz,
        "pipeline_version": PIPELINE_VERSION,
        "elapsed_sec": round(time.time() - t0, 2),
        "videos": results,
    }
    (out_dir / "enrich_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
