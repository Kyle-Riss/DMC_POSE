"""
이미지 단위 end-to-end 검증:
labels.json GT(12->6 매핑) -> YOLO Pose 키포인트 -> Keras 6클래스 분류 예측 비교
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
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


def label12_to_group_ko(label: int | None) -> str | None:
    if label is None:
        return None
    if label < 1 or label > 12:
        return None
    return GROUP_KO[(int(label) - 1) // 2]


def normalize_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg.startswith("cuda"):
        return device_arg
    if device_arg.isdigit():
        return f"cuda:{device_arg}"
    return device_arg


def collect_eval_items(frames_root: Path, skip_transition: bool) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for labels_path in sorted(frames_root.rglob("labels.json")):
        folder = labels_path.parent
        with open(labels_path, encoding="utf-8") as f:
            data = json.load(f)
        for fr in data.get("frames", []):
            if skip_transition and fr.get("is_transition"):
                continue
            gt = label12_to_group_ko(fr.get("label"))
            if gt is None:
                continue
            fn = fr.get("filename")
            if not fn:
                continue
            jpg = folder / fn
            if not jpg.is_file():
                continue
            items.append((jpg, gt))
    return items


def class_order_from_csv(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, usecols=["Pose_Class"])
    _, categories = df["Pose_Class"].factorize()
    return list(categories)


def write_report(
    report_path: Path,
    summary: dict[str, str | int | float],
    class_names: list[str],
    gt_indices: list[int],
    pred_indices: list[int],
) -> None:
    lines: list[str] = []
    for k, v in summary.items():
        lines.append(f"{k}={v}")

    lines.append("")
    lines.append("[labels_order]")
    for idx, name in enumerate(class_names):
        lines.append(f"{idx}={name}")

    if gt_indices:
        lines.append("")
        lines.append("[classification_report]")
        lines.append(
            classification_report(
                gt_indices,
                pred_indices,
                labels=list(range(len(class_names))),
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )
        lines.append("")
        lines.append("[confusion_matrix rows=true cols=pred]")
        cm = confusion_matrix(
            gt_indices,
            pred_indices,
            labels=list(range(len(class_names))),
        )
        lines.append(np.array2string(cm, max_line_width=200))

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dmc/pose/extracted_frames"),
        help="extracted_frames 루트",
    )
    ap.add_argument(
        "--dataset-csv",
        type=Path,
        default=Path("/home/dmc/AI/DMC_POSE/pose_dataset_six.csv"),
        help="학습 시 사용한 CSV (클래스 인덱스 순서 복원용)",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("/home/dmc/AI/DMC_POSE/my_model_six.keras"),
        help="Keras 분류기 경로",
    )
    ap.add_argument(
        "--weights",
        type=str,
        default="yolo11m-pose.pt",
        help="YOLO pose 가중치 경로",
    )
    ap.add_argument("--device", type=str, default="0", help="cuda:0 또는 cpu")
    ap.add_argument("--sample-size", type=int, default=300, help="검증 샘플 수")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    ap.add_argument(
        "--save-images",
        type=int,
        default=40,
        help="예측 오버레이 저장 이미지 수",
    )
    ap.add_argument(
        "--include-transition",
        action="store_true",
        help="is_transition=True 프레임 포함",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/dmc/AI/DMC_POSE/runs/e2e_validate"),
        help="리포트/시각화 저장 폴더",
    )
    args = ap.parse_args()

    if not args.root.is_dir():
        raise FileNotFoundError(f"frames root not found: {args.root}")
    if not args.dataset_csv.is_file():
        raise FileNotFoundError(f"dataset csv not found: {args.dataset_csv}")
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")

    tf.config.set_visible_devices([], "GPU")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    class_names = class_order_from_csv(args.dataset_csv)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print("[INFO] collecting valid labeled frames...")
    all_items = collect_eval_items(args.root, skip_transition=not args.include_transition)
    if not all_items:
        raise RuntimeError("no valid labeled image found")

    random.seed(args.seed)
    sample_size = min(args.sample_size, len(all_items))
    sampled_items = random.sample(all_items, sample_size)
    print(f"[INFO] sampled: {sample_size} / total valid labeled: {len(all_items)}")

    pose_model = YOLO(args.weights)
    clf_model = tf.keras.models.load_model(args.model)
    device = normalize_device(args.device)

    gt_indices: list[int] = []
    pred_indices: list[int] = []
    skip_no_pose = 0
    saved = 0
    pred_counter: Counter[str] = Counter()
    gt_counter: Counter[str] = Counter()

    per_sample_lines: list[str] = []

    for img_path, gt_name in tqdm(sampled_items, desc="e2e validate"):
        gt_counter[gt_name] += 1

        results = pose_model.predict(str(img_path), device=device, verbose=False)
        if len(results) == 0 or len(results[0].keypoints) == 0:
            skip_no_pose += 1
            continue

        kpts = results[0].keypoints[0].xy[0].cpu().numpy().flatten()
        if len(kpts) != 34:
            skip_no_pose += 1
            continue

        probs = clf_model.predict(kpts.reshape(1, -1).astype("float32"), verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        conf = float(probs[pred_idx])

        gt_idx = class_to_idx[gt_name]
        gt_indices.append(gt_idx)
        pred_indices.append(pred_idx)
        pred_counter[pred_name] += 1

        match = pred_name == gt_name
        per_sample_lines.append(f"src={img_path}")
        per_sample_lines.append(
            f"pred={pred_name}, conf={conf:.4f}, gt={gt_name}, match={match}"
        )
        per_sample_lines.append("")

        if saved < args.save_images:
            image = cv2.imread(str(img_path))
            if image is not None:
                boxes = (
                    results[0].boxes.xyxy.cpu().numpy().astype(int)
                    if results[0].boxes is not None and len(results[0].boxes) > 0
                    else []
                )
                if len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    image,
                    f"Pred: {pred_name} ({conf:.3f})",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    f"GT: {gt_name} [{'OK' if match else 'MISS'}]",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                out_name = (
                    f"{saved + 1:03d}__pred-{pred_name}__gt-{gt_name}__conf-{conf:.3f}.jpg"
                )
                cv2.imwrite(str(args.out_dir / out_name), image)
                saved += 1

    used = len(gt_indices)
    correct = int(sum(1 for g, p in zip(gt_indices, pred_indices) if g == p))
    accuracy = (correct / used) if used else 0.0

    summary: dict[str, str | int | float] = {
        "root": str(args.root),
        "dataset_csv": str(args.dataset_csv),
        "model": str(args.model),
        "weights": args.weights,
        "sampled": sample_size,
        "used_for_eval": used,
        "skipped_no_pose": skip_no_pose,
        "correct": correct,
        "accuracy": f"{accuracy:.6f}",
    }

    report_path = args.out_dir / "report.txt"
    write_report(report_path, summary, class_names, gt_indices, pred_indices)

    # 샘플 상세 로그 추가
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n\n[gt_counts]\n")
        for name in class_names:
            f.write(f"{name}={gt_counter[name]}\n")
        f.write("\n[pred_counts]\n")
        for name in class_names:
            f.write(f"{name}={pred_counter[name]}\n")
        f.write("\n[per_sample]\n")
        f.write("\n".join(per_sample_lines))

    print(f"[OK] report: {report_path}")
    print(f"[OK] saved_images: {saved} in {args.out_dir}")
    print(f"[OK] accuracy: {accuracy * 100:.2f}% ({correct}/{used})")


if __name__ == "__main__":
    main()
