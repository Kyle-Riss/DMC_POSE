#!/usr/bin/env python3
"""RTSP 수동 라벨 → YOLO bed-seg v1 dataset (실시간 카메라 전용)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import imutils

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset_v1"
RTSP_RAW = ROOT / "rtsp_raw"
MANUAL_LABELS = ROOT / "manual_labels"
VLCSNAP_DIR = Path("/home/dmc/labeling")
VLCSNAP_LABELS = VLCSNAP_DIR / "labels"


def load_pairs(
    image_dir: Path,
    label_dir: Path,
    resize_width: int | None,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(image_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = label_dir / f"{img.stem}.txt"
        if lbl.is_file():
            pairs.append((img, lbl))
    return pairs


def copy_pair(
    src_img: Path,
    src_lbl: Path,
    dst_img: Path,
    dst_lbl: Path,
    resize_width: int | None,
) -> bool:
    img = cv2.imread(str(src_img))
    if img is None:
        return False
    if resize_width:
        img = imutils.resize(img, width=resize_width)
    h, w = img.shape[:2]
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_img), img)
    # 라벨은 정규화 좌표라 리사이즈해도 그대로 사용 가능
    dst_lbl.write_text(src_lbl.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-vlcsnap", action="store_true", default=True)
    parser.add_argument("--no-vlcsnap", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)

    pairs: list[tuple[Path, Path, str]] = []

    rtsp_pairs = load_pairs(RTSP_RAW, MANUAL_LABELS, args.resize_width)
    for img, lbl in rtsp_pairs:
        pairs.append((img, lbl, "rtsp"))

    if args.include_vlcsnap and not args.no_vlcsnap and VLCSNAP_LABELS.is_dir():
        for img, lbl in load_pairs(VLCSNAP_DIR, VLCSNAP_LABELS, args.resize_width):
            if img.name.startswith("vlcsnap"):
                pairs.append((img, lbl, "vlcsnap"))

    if len(pairs) < 8:
        raise SystemExit(
            f"labeled pairs too few: {len(pairs)}\n"
            f"  1) bash capture_rtsp_frames.py\n"
            f"  2) label: python /home/dmc/labeling/label_bed_polygon.py "
            f"--images {RTSP_RAW} --labels {MANUAL_LABELS}\n"
            f"  3) rerun this script"
        )

    if DATASET.exists():
        shutil.rmtree(DATASET)

    random.shuffle(pairs)
    n_val = max(4, int(len(pairs) * args.val_ratio))
    # RTSP는 train 우선, val은 마지막 n_val
    val_set = {p[0] for p in pairs[-n_val:]}

    counts = {"train": 0, "val": 0, "rtsp": 0, "vlcsnap": 0}
    for src_img, src_lbl, src_tag in pairs:
        split = "val" if src_img in val_set else "train"
        stem = f"{split}_{counts[split]:05d}"
        counts[split] += 1
        counts[src_tag] += 1
        copy_pair(
            src_img,
            src_lbl,
            DATASET / "images" / split / f"{stem}.jpg",
            DATASET / "labels" / split / f"{stem}.txt",
            args.resize_width,
        )

    yaml = f"""path: {DATASET}
train: images/train
val: images/val
names:
  0: bed
"""
    (DATASET / "data.yaml").write_text(yaml, encoding="utf-8")
    summary = {
        "total": len(pairs),
        "train": counts["train"],
        "val": counts["val"],
        "rtsp_labeled": counts["rtsp"],
        "vlcsnap_labeled": counts["vlcsnap"],
        "resize_width": args.resize_width,
    }
    (DATASET / "dataset_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
