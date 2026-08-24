#!/usr/bin/env python3
"""Benchmark end-to-end Ultralytics Pose micro-batches on local imagery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def percentile(values: list[float], value: float) -> float:
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), value)), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, action="append", dest="images")
    parser.add_argument("--batch-size", type=int, action="append", dest="batch_sizes")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    batch_sizes = args.batch_sizes or [1, 2, 3, 6]
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("batch sizes must be positive")
    frames = []
    for image_path in args.images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(image_path)
        frames.append(frame)

    model = YOLO(str(args.weights))
    results = []
    for batch_size in batch_sizes:
        source = [frames[index % len(frames)] for index in range(batch_size)]
        with torch.inference_mode():
            for _ in range(args.warmup):
                model.predict(source=source, imgsz=args.imgsz, device=args.device, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                prediction = model.predict(source=source, imgsz=args.imgsz, device=args.device, verbose=False)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000.0)
        mean_ms = float(np.mean(samples))
        person_detections = sum(len(item.keypoints) if item.keypoints is not None else 0 for item in prediction)
        results.append({
            "batch_size": batch_size,
            "iterations": args.iterations,
            "p50_batch_ms": percentile(samples, 50),
            "p95_batch_ms": percentile(samples, 95),
            "mean_batch_ms": round(mean_ms, 3),
            "mean_frames_per_sec": round(batch_size * 1000.0 / mean_ms, 3),
            "last_batch_person_detections": person_detections,
        })

    report = {
        "benchmark": "ultralytics_pose_microbatch_wall_v1",
        "scope": "decode excluded; preprocessing, GPU inference, and Ultralytics postprocessing included",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "weights": str(args.weights.resolve()),
        "images": [str(path.resolve()) for path in args.images],
        "image_shapes": [list(frame.shape) for frame in frames],
        "imgsz": args.imgsz,
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
