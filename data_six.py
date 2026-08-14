"""
extracted_frames/*/.../labels.json + 동일 폴더의 JPG를 읽어
라벨 1~12 → 6묶음 한글 클래스로 매핑한 뒤 YOLO Pose로 키포인트 CSV 생성.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

# 라벨 1~2 → 0, 3~4 → 1, … (label-1)//2
GROUP_KO = [
    "정면_누움",
    "엎드림_등",
    "옆누움_가까움",
    "옆누움_멀음",
    "앉음_중앙",
    "앉음_가장자리",
]


def label12_to_group_ko(label: int) -> str | None:
    if label is None or label < 1 or label > 12:
        return None
    return GROUP_KO[(int(label) - 1) // 2]


def iter_frame_tasks(frames_root: Path, skip_transition: bool):
    """yield (jpg_path, group_ko)"""
    for labels_path in sorted(frames_root.rglob("labels.json")):
        folder = labels_path.parent
        with open(labels_path, encoding="utf-8") as f:
            data = json.load(f)
        for fr in data.get("frames", []):
            if skip_transition and fr.get("is_transition"):
                continue
            label = fr.get("label")
            gko = label12_to_group_ko(label)
            if gko is None:
                continue
            fn = fr.get("filename")
            if not fn:
                continue
            jpg = folder / fn
            if not jpg.is_file():
                continue
            yield jpg, gko


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dmc/pose/extracted_frames"),
        help="extracted_frames 루트",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/dmc/AI/DMC_POSE/pose_dataset_six.csv"),
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="yolo11m-pose.pt",
        help="YOLO pose 가중치 경로",
    )
    ap.add_argument("--device", type=str, default="0", help="cuda:0 또는 cpu")
    ap.add_argument(
        "--include-transition",
        action="store_true",
        help="is_transition=True 인 프레임도 포함 (기본은 제외)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="디버그용 최대 샘플 수 (미설정 시 전체)",
    )
    args = ap.parse_args()
    skip_transition = not args.include_transition

    if not args.root.is_dir():
        print(f"[error] 폴더 없음: {args.root}")
        return

    tasks = list(iter_frame_tasks(args.root, skip_transition))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[INFO] 유효 프레임(파일 존재): {len(tasks)} (transition 제외={skip_transition})")

    dev = args.device
    if dev != "cpu" and not dev.startswith("cuda"):
        dev = f"cuda:{dev}" if dev.isdigit() else dev

    model = YOLO(args.weights)
    data_list = []

    for jpg_path, gko in tqdm(tasks, desc="YOLO pose"):
        results = model.predict(str(jpg_path), device=dev, verbose=False)
        if len(results) == 0 or len(results[0].keypoints) == 0:
            continue
        kpts = results[0].keypoints[0].xy[0].cpu().numpy().flatten()
        if len(kpts) != 34:
            continue
        data_list.append(np.concatenate([[gko], kpts]))

    columns = ["Pose_Class"] + [f"kpt_{i}" for i in range(34)]
    df = pd.DataFrame(data_list, columns=columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[ok] 샘플 {len(df)} 저장 → {args.out}")
    print(df["Pose_Class"].value_counts().sort_index())


if __name__ == "__main__":
    main()
