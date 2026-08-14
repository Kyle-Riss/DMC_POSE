#!/usr/bin/env python3
"""Train YOLO11n bed-seg (v1: RTSP manual labels)."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE = Path("/home/dmc/AI/DMC_POSE/yolo11n-seg.pt")
DEFAULT_DATA = ROOT / "dataset_v1" / "data.yaml"
OUT_WEIGHTS = Path("/home/dmc/AI/DMC_POSE/yolo11n-bed-seg.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="bed_seg_v1")
    parser.add_argument("--export", type=Path, default=OUT_WEIGHTS)
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"missing {args.data} — run prepare_rtsp_dataset.py first")

    model = YOLO(str(BASE))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=20,
        plots=True,
        # 고정 카메라: 과한 perspective 완화
        perspective=0.0,
        degrees=0.0,
    )
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.is_file() and args.export:
        import shutil

        shutil.copy2(best, args.export)
        print(f"deployed: {args.export}")
    print(results)


if __name__ == "__main__":
    main()
