#!/usr/bin/env python3
"""
Raw MP4 → timestamp + 6-class pose inference → JSON + CSV (시계열용).

입력 (정리된 Raw_data):
  video/*.mp4
  meta/frame_timestamps/{stem}_frame_timestamps.json

출력:
  timeseries/{stem}.json
  timeseries/{stem}.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from ultralytics import YOLO

GROUP_KO = [
    "정면_누움",
    "엎드림_등",
    "옆누움_가까움",
    "옆누움_멀음",
    "앉음_중앙",
    "앉음_가장자리",
]


def class_order_from_csv(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["Pose_Class"])
    _, categories = df["Pose_Class"].factorize()
    return list(categories)


def normalize_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg.startswith("cuda"):
        return device_arg
    if device_arg.isdigit():
        return f"cuda:{device_arg}"
    return device_arg


def load_timestamp_map(ts_path: Path) -> dict[int, float]:
    data = json.loads(ts_path.read_text(encoding="utf-8"))
    out: dict[int, float] = {}
    for fr in data.get("frames", []):
        fid = int(fr["frame_id"])
        out[fid] = float(fr.get("timestamp_sec", fid / float(data.get("fps") or 20.0)))
    return out


def sample_frame_ids(
    frame_count: int, fps: float, sample_hz: float, timestamp_map: dict[int, float]
) -> list[int]:
    if sample_hz <= 0:
        return list(range(frame_count))
    duration = frame_count / fps if fps > 0 else 0.0
    ids: list[int] = []
    sec = 0.0
    while sec <= duration + 1e-6:
        fid = int(round(sec * fps))
        if fid >= frame_count:
            break
        if fid not in ids:
            ids.append(fid)
        sec += 1.0 / sample_hz
    if not ids and frame_count > 0:
        ids = [0]
    return ids


def infer_frame(
    frame: np.ndarray,
    pose_model: YOLO,
    clf_model: tf.keras.Model,
    class_names: list[str],
    device: str,
) -> tuple[str | None, float | None, np.ndarray | None]:
    results = pose_model.predict(frame, device=device, verbose=False)
    if len(results) == 0 or len(results[0].keypoints) == 0:
        return None, None, None
    kpts = results[0].keypoints[0].xy[0].cpu().numpy().flatten()
    if len(kpts) != 34:
        return None, None, None
    probs = clf_model.predict(kpts.reshape(1, -1).astype("float32"), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    if pred_idx >= len(class_names):
        return None, None, None
    return class_names[pred_idx], float(probs[pred_idx]), kpts


def process_one_video(
    video_path: Path,
    ts_path: Path,
    out_json: Path,
    out_csv: Path,
    pose_model: YOLO,
    clf_model: tf.keras.Model,
    class_names: list[str],
    device: str,
    sample_hz: float,
    rotation_bucket: str,
    force: bool,
) -> dict:
    if out_json.exists() and out_csv.exists() and not force:
        return {"video": video_path.name, "status": "skipped", "rows": 0}

    if not ts_path.is_file():
        return {"video": video_path.name, "status": "error", "error": f"missing {ts_path}"}

    ts_meta = json.loads(ts_path.read_text(encoding="utf-8"))
    fps = float(ts_meta.get("fps") or 20.0)
    frame_count = int(ts_meta.get("frame_count") or 0)
    timestamp_map = load_timestamp_map(ts_path)
    frame_ids = sample_frame_ids(frame_count, fps, sample_hz, timestamp_map)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.name, "status": "error", "error": "cannot_open_video"}

    class_to_id = {n: i for i, n in enumerate(class_names)}
    frames_out: list[dict] = []
    rows: list[dict] = []

    for fid in tqdm(frame_ids, desc=video_path.name, leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        t_sec = timestamp_map.get(fid, fid / fps)
        pose_class, pose_conf, kpts = infer_frame(
            frame, pose_model, clf_model, class_names, device
        )
        row = {
            "video_file": video_path.name,
            "frame_idx": fid,
            "timestamp_sec": round(t_sec, 6),
            "fps": round(fps, 6),
            "rotation_bucket": rotation_bucket,
            "pose_class": pose_class or "",
            "pose_class_id": class_to_id.get(pose_class, -1) if pose_class else -1,
            "pose_conf": round(pose_conf, 6) if pose_conf is not None else None,
            "person_detected": pose_class is not None,
        }
        if kpts is not None:
            for i, v in enumerate(kpts):
                row[f"kpt_{i}"] = round(float(v), 4)
        else:
            for i in range(34):
                row[f"kpt_{i}"] = None
        rows.append(row)

        frames_out.append(
            {
                "frame_id": fid,
                "timestamp_sec": row["timestamp_sec"],
                "pose_class": row["pose_class"],
                "pose_class_id": row["pose_class_id"],
                "pose_conf": row["pose_conf"],
                "person_detected": row["person_detected"],
                "keypoints_xy": kpts.tolist() if kpts is not None else None,
            }
        )

    cap.release()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "video_file": video_path.name,
        "video_path": str(video_path.resolve()),
        "fps": fps,
        "frame_count": frame_count,
        "sample_hz": sample_hz,
        "rotation_bucket": rotation_bucket,
        "class_names": class_names,
        "frame_count_sampled": len(frames_out),
        "frames": frames_out,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    return {
        "video": video_path.name,
        "status": "ok",
        "rows": len(rows),
        "json": str(out_json),
        "csv": str(out_csv),
    }


def resolve_out_dir(raw_root: Path, sample_hz: float, out_dir_arg: Path | None) -> Path:
    if out_dir_arg is not None:
        return out_dir_arg
    if sample_hz == 1.0:
        return raw_root / "timeseries"
    hz_label = str(int(sample_hz)) if sample_hz == int(sample_hz) else str(sample_hz).replace(".", "p")
    return raw_root / f"timeseries_{hz_label}hz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract raw video timeseries JSON+CSV")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/dmc/Dataset/Raw_data"),
    )
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("/home/dmc/AI/DMC_POSE/pose_dataset_six.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/dmc/AI/DMC_POSE/my_model_six.keras"),
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/dmc/AI/DMC_POSE/yolo11m-pose.pt",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--sample-hz",
        type=float,
        default=1.0,
        help="Samples per second (1.0 = 1Hz, 5.0 = 5Hz = 0.2초마다 1행)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="출력 폴더 (기본: timeseries 또는 timeseries_5hz 등 sample_hz별)",
    )
    parser.add_argument(
        "--rotation-bucket",
        type=str,
        default="unknown",
        help="straight|backside|side|opposite_side|unknown",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="특정 mp4 파일명만 처리 (예: 'Raw0 (1).mp4')",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    video_dir = raw_root / "video"
    ts_dir = raw_root / "meta" / "frame_timestamps"
    out_dir = resolve_out_dir(raw_root, args.sample_hz, args.out_dir)
    print(f"Output dir: {out_dir} (sample_hz={args.sample_hz})")

    if not video_dir.is_dir():
        raise FileNotFoundError(f"video dir not found: {video_dir}")
    if not args.dataset_csv.is_file():
        raise FileNotFoundError(f"dataset csv not found: {args.dataset_csv}")
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    pose_device = normalize_device(args.device)
    # RTX 50xx (sm_120) 등: TF/Keras GPU 커널 미지원 → YOLO만 GPU, 분류기는 CPU
    tf.config.set_visible_devices([], "GPU")

    if pose_device != "cpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"GPU requested ({pose_device}) but CUDA unavailable. "
                "Check: nvidia-smi, conda activate pose-cuda, or use --device cpu"
            )
        print(f"YOLO GPU: {torch.cuda.get_device_name(0)} ({pose_device})")
    print("Keras classifier: CPU")

    class_names = class_order_from_csv(args.dataset_csv)
    pose_model = YOLO(args.weights)
    clf_model = tf.keras.models.load_model(args.model)

    videos = sorted(video_dir.glob("*.mp4"))
    if args.video:
        videos = [p for p in videos if p.name == args.video]
        if not videos:
            raise FileNotFoundError(f"video not found: {args.video}")

    t0 = time.time()
    results: list[dict] = []
    for vp in videos:
        stem = vp.stem
        ts_path = ts_dir / f"{stem}_frame_timestamps.json"
        out_json = out_dir / f"{stem}.json"
        out_csv = out_dir / f"{stem}.csv"
        results.append(
            process_one_video(
                video_path=vp,
                ts_path=ts_path,
                out_json=out_json,
                out_csv=out_csv,
                pose_model=pose_model,
                clf_model=clf_model,
                class_names=class_names,
                device=pose_device,
                sample_hz=args.sample_hz,
                rotation_bucket=args.rotation_bucket,
                force=args.force,
            )
        )

    index = {
        "raw_root": str(raw_root),
        "sample_hz": args.sample_hz,
        "rotation_bucket_default": args.rotation_bucket,
        "class_names": class_names,
        "elapsed_sec": round(time.time() - t0, 2),
        "videos": results,
    }
    index_path = out_dir / "timeseries_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(index_path)
    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
