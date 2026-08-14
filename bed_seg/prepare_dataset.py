#!/usr/bin/env python3
"""vlcsnap + extracted_frames + RTSP → YOLO bed-seg dataset."""
from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import cv2

from label_utils import bed_polygon, draw_preview, yolo_seg_line

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"


def capture_rtsp(url: str, n: int, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    saved: list[Path] = []
    for i in range(n * 3):
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.1)
            continue
        if frame.shape[:2] != (360, 640):
            frame = cv2.resize(frame, (640, 360))
        if i % 3 != 0:
            continue
        path = out_dir / f"rtsp_{len(saved):04d}.jpg"
        cv2.imwrite(str(path), frame)
        saved.append(path)
        if len(saved) >= n:
            break
    cap.release()
    return saved


def add_image(src: Path, dst_img: Path, dst_lbl: Path, preview_dir: Path | None) -> bool:
    img = cv2.imread(str(src))
    if img is None:
        return False
    if img.shape[:2] != (360, 640):
        img = cv2.resize(img, (640, 360))
    poly = bed_polygon(img)
    if poly is None:
        return False
    h, w = img.shape[:2]
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_img), img)
    dst_lbl.write_text(yolo_seg_line(poly, w, h) + "\n", encoding="utf-8")
    if preview_dir is not None:
        cv2.imwrite(str(preview_dir / f"{dst_img.stem}.jpg"), draw_preview(img, poly))
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-sample", type=int, default=300)
    parser.add_argument("--rtsp-samples", type=int, default=24)
    parser.add_argument(
        "--rtsp-url", default="rtsp://192.168.0.161:8554/stream"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    if DATASET.exists():
        shutil.rmtree(DATASET)
    preview = ROOT / "label_preview"
    if preview.exists():
        shutil.rmtree(preview)
    preview.mkdir(parents=True, exist_ok=True)

    staging: list[Path] = []
    staging.extend(sorted(Path("/home/dmc/labeling").glob("vlcsnap*.png")))

    extracted = sorted(Path("/home/dmc/pose/extracted_frames").rglob("*.jpg"))
    if args.extracted_sample and extracted:
        staging.extend(random.sample(extracted, min(args.extracted_sample, len(extracted))))

    if args.rtsp_samples:
        rtsp_dir = ROOT / "_rtsp_cache"
        if rtsp_dir.exists():
            shutil.rmtree(rtsp_dir)
        staging.extend(capture_rtsp(args.rtsp_url, args.rtsp_samples, rtsp_dir))

    random.shuffle(staging)
    ok_items: list[Path] = []
    for src in staging:
        poly_img = cv2.imread(str(src))
        if poly_img is None:
            continue
        if poly_img.shape[:2] != (360, 640):
            poly_img = cv2.resize(poly_img, (640, 360))
        if bed_polygon(poly_img) is None:
            continue
        ok_items.append(src)

    if len(ok_items) < 12:
        raise SystemExit(f"too few labeled images: {len(ok_items)}")

    n_val = max(4, int(len(ok_items) * 0.15))
    val_set = set(ok_items[:n_val])
    counts = {"train": 0, "val": 0}
    for src in ok_items:
        split = "val" if src in val_set else "train"
        stem = f"{split}_{counts[split]:05d}"
        counts[split] += 1
        dst_img = DATASET / "images" / split / f"{stem}.jpg"
        dst_lbl = DATASET / "labels" / split / f"{stem}.txt"
        add_image(src, dst_img, dst_lbl, preview if split == "val" else None)

    yaml = f"""path: {DATASET}
train: images/train
val: images/val
names:
  0: bed
"""
    (DATASET / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"dataset: {DATASET}")
    print(f"train={counts['train']} val={counts['val']}")
    print(f"preview: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
