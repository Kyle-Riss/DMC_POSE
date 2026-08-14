#!/usr/bin/env python3
"""Extract Phase-10 observed-only pose rows from a temporal manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from person_tracker import MultiPersonTracker, PersonDetection, keypoints_bbox
from temporal_features import FEATURE_SCHEMA_VERSION, labels_at, normalize_pose
from temporal_sequence import decide_observation

DEFAULT_CLASSES = [
    "front_lying",
    "prone_back",
    "side_near",
    "side_far",
    "sitting_center",
    "sitting_edge",
]
SEQUENCE_CONTRACT_VERSION = "observed_only_10hz_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] == width:
        return frame
    scale = width / frame.shape[1]
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def frame_timestamp(cap, frame_idx: int, fps: float) -> tuple[float, str]:
    pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if pos_msec > 0.0:
        return pos_msec / 1000.0, "source_pts"
    if fps > 0.0:
        return frame_idx / fps, "frame_index_fps"
    return float(frame_idx), "frame_index"


def detections_from_result(result) -> list[PersonDetection]:
    if result is None or result.keypoints is None or len(result.keypoints) == 0:
        return []
    xy_all = result.keypoints.xy.cpu().numpy()
    if result.keypoints.conf is None:
        conf_all = np.ones(xy_all.shape[:2], dtype=np.float32)
    else:
        conf_all = result.keypoints.conf.cpu().numpy()
    detections = []
    for xy, conf in zip(xy_all, conf_all):
        xy = xy.astype(np.float32)
        conf = conf.astype(np.float32)
        bbox = keypoints_bbox(xy, conf, min_conf=0.2)
        if bbox is None:
            continue
        visible = (conf >= 0.25) & (xy[:, 0] > 0) & (xy[:, 1] > 0)
        if int(visible.sum()) < 5:
            continue
        confidence = float(conf[visible].mean())
        detections.append(PersonDetection(xy, conf, bbox, confidence, 0.0))
    return detections


def row_from_pose(item: dict, frame_idx: int, timestamp_sec: float, xy: np.ndarray, conf: np.ndarray, pose_probs: np.ndarray, *, track_id: int, sequence_id: int, timestamp_source: str, reset_reason: str) -> dict:
    normalized = normalize_pose(xy, conf)
    target, active = labels_at(timestamp_sec, item.get("intervals", []))
    visible = normalized["visibility"]
    video_path = item.get("video_path") or item.get("source_path")
    row = {
        "video_id": item["video_id"],
        "dataset": item["dataset"],
        "subject_id": item.get("subject_id"),
        "split": item.get("split"),
        "video_file": Path(video_path).name,
        "frame_idx": frame_idx,
        "timestamp_sec": round(timestamp_sec, 6),
        "timestamp_source": timestamp_source,
        "sample_hz": None,
        "person_detected": True,
        "track_id": int(track_id),
        "sequence_id": int(sequence_id),
        "sequence_reset_reason": reset_reason,
        "target": target,
        "active_labels": "|".join(active),
        "pose_center_x": float(normalized["center"][0]),
        "pose_center_y": float(normalized["center"][1]),
        "pose_scale": float(normalized["scale"]),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sequence_contract_version": SEQUENCE_CONTRACT_VERSION,
    }
    xy_norm = normalized["xy_norm"]
    for index in range(17):
        row[f"kpt_{index}_x"] = float(xy[index, 0])
        row[f"kpt_{index}_y"] = float(xy[index, 1])
        row[f"kpt_{index}_conf"] = float(conf[index])
        row[f"kpt_{index}_visible"] = int(visible[index])
        row[f"kpt_{index}_x_norm"] = float(xy_norm[index, 0])
        row[f"kpt_{index}_y_norm"] = float(xy_norm[index, 1])
    for index, value in enumerate(pose_probs):
        row[f"pose_prob_{index}"] = float(value)
    return row


def process_video(item: dict, pose_model, classifier, class_names: list[str], args, out_dir: Path) -> dict:
    video_path = Path(item.get("video_path") or item.get("source_path"))
    split = str(item.get("split") or "unassigned")
    out_path = out_dir / split / (str(item["video_id"]) + ".csv")
    if out_path.exists() and not args.force:
        return {"video_id": item["video_id"], "status": "skipped", "out": str(out_path)}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": item["video_id"], "status": "error", "error": "cannot_open_video"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or item.get("fps") or 0.0)
    tracker = MultiPersonTracker(track_ttl_sec=2.0, primary_switch_margin=0.25)
    rows = []
    frame_idx = -1
    next_probe_ts = 0.0
    probe_period = 1.0 / args.sample_hz
    last_observation_ts = None
    last_track_id = None
    sequence_id = 0
    counters = {"decoded_frames": 0, "pose_probes": 0, "no_primary": 0, "duplicate_skip": 0, "non_monotonic_skip": 0, "gap_reset": 0, "track_reset": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        counters["decoded_frames"] += 1
        timestamp_sec, timestamp_source = frame_timestamp(cap, frame_idx, fps)
        if timestamp_sec + 1e-9 < next_probe_ts:
            continue
        while next_probe_ts <= timestamp_sec + 1e-9:
            next_probe_ts += probe_period
        counters["pose_probes"] += 1
        frame = resize_width(frame, args.frame_width)
        results = pose_model.predict(frame, device=args.device, verbose=False)
        detections = detections_from_result(results[0] if results else None)
        tracking = tracker.update(detections, timestamp_sec, frame_width=frame.shape[1], frame_height=frame.shape[0])
        primary = tracking.primary
        if primary is None:
            counters["no_primary"] += 1
            continue

        reset_reason = ""
        if last_track_id is not None and primary.track_id != last_track_id:
            sequence_id += 1
            last_observation_ts = None
            counters["track_reset"] += 1
            reset_reason = "track_change"
        decision = decide_observation(timestamp_sec, last_observation_ts, min_interval_sec=args.min_interval_sec, max_interval_sec=args.max_interval_sec)
        if decision.action == "duplicate_skip":
            counters["duplicate_skip"] += 1
            continue
        if decision.action == "non_monotonic_skip":
            counters["non_monotonic_skip"] += 1
            continue
        if decision.reset:
            sequence_id += 1
            counters["gap_reset"] += 1
            reset_reason = "observation_gap"
        if sequence_id == 0:
            sequence_id = 1
            reset_reason = "video_start"

        pred = classifier.predict(primary.keypoints_xy.reshape(1, 34).astype(np.float32), verbose=0)[0]
        if len(pred) != len(class_names):
            raise ValueError(f"classifier output must have {len(class_names)} values, got {len(pred)}")
        row = row_from_pose(item, frame_idx, timestamp_sec, primary.keypoints_xy, primary.keypoints_conf, np.asarray(pred, dtype=np.float32), track_id=primary.track_id, sequence_id=sequence_id, timestamp_source=timestamp_source, reset_reason=reset_reason)
        row["sample_hz"] = args.sample_hz
        rows.append(row)
        last_observation_ts = timestamp_sec
        last_track_id = primary.track_id
    cap.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {
        "video_id": item["video_id"],
        "status": "ok",
        "rows": len(rows),
        "sequences": len({row["sequence_id"] for row in rows}),
        "fall_rows": sum(row["target"] == "fall" for row in rows),
        "out": str(out_path),
        **counters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--manifest", type=Path, default=project_root / "external_datasets/manifests/gmdcsa24.json")
    parser.add_argument("--out-dir", type=Path, default=project_root / "external_datasets/features/tcn_109_v2_no_missing/gmdcsa24")
    parser.add_argument("--weights", type=Path, default=project_root / "yolo11m-pose.pt")
    parser.add_argument("--classifier", type=Path, default=project_root / "my_model_six_check.keras")
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--min-interval-sec", type=float, default=0.070)
    parser.add_argument("--max-interval-sec", type=float, default=0.150)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--video-id")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.min_interval_sec <= args.max_interval_sec:
        raise ValueError("sampling interval must satisfy 0 < min <= max")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if args.video_id:
        items = [item for item in items if item["video_id"] == args.video_id]
    if args.split:
        items = [item for item in items if item.get("split") == args.split]
    if args.max_videos is not None:
        items = items[: args.max_videos]
    missing = [item.get("video_path") or item.get("source_path") for item in items if not Path(item.get("video_path") or item.get("source_path")).is_file()]
    if missing:
        raise FileNotFoundError(f"manifest videos missing: {missing[:3]}")
    if args.validate_only:
        print(json.dumps({"manifest": str(args.manifest.resolve()), "selected_videos": len(items), "missing": 0}, indent=2))
        return 0
    if not args.classifier.is_file():
        raise FileNotFoundError("the six-class classifier is required by the 109-feature contract")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    from ultralytics import YOLO

    tf.config.set_visible_devices([], "GPU")
    pose_model = YOLO(str(args.weights))
    classifier = tf.keras.models.load_model(args.classifier)
    out_dir = args.out_dir.resolve()
    started = time.time()
    results = [process_video(item, pose_model, classifier, DEFAULT_CLASSES, args, out_dir) for item in items]
    index = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sequence_contract_version": SEQUENCE_CONTRACT_VERSION,
        "manifest": str(args.manifest.resolve()),
        "sample_hz": args.sample_hz,
        "min_interval_sec": args.min_interval_sec,
        "max_interval_sec": args.max_interval_sec,
        "frame_width": args.frame_width,
        "pose_weights": str(args.weights.resolve()),
        "pose_weights_sha256": file_sha256(args.weights),
        "classifier": str(args.classifier.resolve()),
        "classifier_sha256": file_sha256(args.classifier),
        "elapsed_sec": round(time.time() - started, 2),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "features_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    errors = [result for result in results if result["status"] == "error"]
    print(json.dumps({"videos": len(results), "ok": sum(result["status"] == "ok" for result in results), "errors": errors, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
