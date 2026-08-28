#!/usr/bin/env python3
"""Partially fine-tune Swin3D-B with positive-session-held-out cross-validation."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import Swin3D_B_Weights, swin3d_b


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_reviewed_ai_runner_delta_embeddings import load_labels
from scripts.train_swin3d_verifier_probe import metrics, operating_point


EVENT_RE = re.compile(r"^(bed_[^_]+)_fall_(\d{8})_(\d{6})$")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_indices(start: int, end: int, count: int = 16) -> list[int]:
    return np.rint(np.linspace(start, end, count)).astype(int).tolist()


def decode_video(path: Path, start: int, end: int) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        for index in sample_indices(start, end):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"failed decoding frame {index}: {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def decode_event(path: Path) -> torch.Tensor:
    frame_paths = sorted((path / "frames").glob("*.jpg"))
    if len(frame_paths) < 16:
        raise ValueError(f"event has fewer than 16 frames: {path}")
    frames = []
    for index in sample_indices(0, len(frame_paths) - 1):
        frame = cv2.imread(str(frame_paths[index]))
        if frame is None:
            raise ValueError(f"failed decoding: {frame_paths[index]}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


class ClipDataset(Dataset):
    def __init__(self, rows: list[dict], *, train: bool) -> None:
        self.rows = rows
        self.train = train
        self.transform = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        if row["source"] == "staged":
            clip = decode_video(
                Path(row["video_path"]), int(row["start_frame"]), int(row["end_frame"])
            )
        else:
            clip = decode_event(Path(row["event_path"]))
        if self.train and random.random() < 0.5:
            clip = torch.flip(clip, dims=(-1,))
        tensor = self.transform(clip)
        return tensor, torch.tensor(float(row["label"])), row["id"]


def staged_rows(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        {
            "source": "staged",
            "id": row["clip_id"],
            "label": int(row["label"]),
            "video_path": row["video_path"],
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
        }
        for row in manifest["clips"]
        if row["split"] == "train"
    ]


def reviewed_rows(label_paths: list[Path], events_root: Path) -> list[dict]:
    rows = []
    for event_id, (label, confidence) in load_labels(label_paths).items():
        match = EVENT_RE.match(event_id)
        if match is None:
            continue
        rows.append(
            {
                "source": "reviewed",
                "id": event_id,
                "label": int(label == "fall"),
                "review_confidence": confidence,
                "session_group": f"{match.group(2)}_{match.group(3)[:2]}",
                "event_path": str((events_root / event_id).resolve()),
            }
        )
    return rows


def build_model(weight_path: Path, probe_path: Path, device: torch.device) -> nn.Module:
    model = swin3d_b(weights=None)
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.head = nn.Linear(model.head.in_features, 1)
    probe = np.load(probe_path)
    coefficient = probe["coefficient"].reshape(-1).astype(np.float32)
    mean = probe["mean"].astype(np.float32)
    scale = probe["scale"].astype(np.float32)
    raw_weight = coefficient / scale
    raw_bias = float(probe["intercept"].reshape(-1)[0] - np.dot(coefficient, mean / scale))
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(raw_weight).reshape(1, -1))
        model.head.bias.fill_(raw_bias)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.features[6][1].parameters():
        parameter.requires_grad = True
    for parameter in model.norm.parameters():
        parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    return model.to(device)


def forward_logits(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch).reshape(-1)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    model.eval()
    targets, scores, ids = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for clips, target, sample_ids in loader:
            clips = clips.to(device, non_blocking=True)
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = forward_logits(model, clips)
            scores.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
            targets.extend(target.numpy().astype(np.int64).tolist())
            ids.extend(sample_ids)
    elapsed = time.perf_counter() - started
    return (
        np.asarray(targets, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        ids,
        elapsed,
    )


def train_fold(
    fold: int,
    train_rows: list[dict],
    test_rows: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model = build_model(args.weight, args.initial_probe, device)
    backbone_parameters = list(model.features[6][1].parameters()) + list(model.norm.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    positives = sum(int(row["label"]) for row in train_rows)
    negatives = len(train_rows) - positives
    pos_weight = torch.tensor([negatives / max(1, positives)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(
        ClipDataset(train_rows, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        ClipDataset(test_rows, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, (clips, target, _) in enumerate(train_loader, start=1):
            clips = clips.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = forward_logits(model, clips)
                loss = criterion(logits, target) / args.accumulate
            scaler.scale(loss).backward()
            if step % args.accumulate == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_size = len(target)
            total_loss += float(loss.detach()) * args.accumulate * batch_size
            seen += batch_size
        y, probability, _, eval_seconds = evaluate(model, test_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "train_seconds": time.perf_counter() - started,
            "held_auc": float(roc_auc_score(y, probability)),
            "held_at_0_5": metrics(y, probability, 0.5),
            "eval_seconds": eval_seconds,
        }
        history.append(epoch_row)
        print(json.dumps({"fold": fold, **epoch_row}, ensure_ascii=False), flush=True)
    y, probability, ids, eval_seconds = evaluate(model, test_loader, device)
    fold_dir = args.out_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "swin3d_b_last_block_binary_v1",
            "trainable_scope": ["features.6.1", "norm", "head"],
            "fold": fold,
        },
        fold_dir / "model.pt",
    )
    return {
        "fold": fold,
        "history": history,
        "train_events": len(train_rows),
        "held_events": len(test_rows),
        "held_targets": y.tolist(),
        "held_probabilities": probability.tolist(),
        "held_ids": ids,
        "eval_seconds": eval_seconds,
        "model": str((fold_dir / "model.pt").resolve()),
    }


def main() -> int:
    review_dir = PROJECT_ROOT / "runtime_data/ai_runner_fall_review_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, nargs="+", default=sorted(review_dir.glob("reviewed_labels*.json"))
    )
    parser.add_argument(
        "--events-root", type=Path, default=Path("/home/dmc/AI/AI_runner/data/events/fall")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "external_datasets/manifests/swin3d_verifier_staged_v1.json",
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=PROJECT_ROOT / "external_models/torchvision/swin3d_b_22k-7c6ae6fa.pth",
    )
    parser.add_argument(
        "--initial-probe",
        type=Path,
        default=PROJECT_ROOT
        / "runs/video_verifier/swin3d_b_staged_linear_v1_20260828/linear_probe.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "runs/video_verifier/swin3d_b_partial_ft_cv_v1_20260828",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-cv-recall", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    seed_everything(args.seed)
    device = torch.device(args.device)

    staged = staged_rows(args.manifest)
    reviewed = reviewed_rows(args.labels, args.events_root)
    positive_groups = sorted(
        {row["session_group"] for row in reviewed if row["label"] == 1}
    )
    if len(positive_groups) != 2:
        raise ValueError(f"expected two positive session groups, got {positive_groups}")
    negative_groups = sorted(
        {
            row["session_group"]
            for row in reviewed
            if row["label"] == 0 and row["session_group"] not in positive_groups
        }
    )
    negative_fold = {group: index % 2 for index, group in enumerate(negative_groups)}
    folds = []
    for fold, positive_group in enumerate(positive_groups):
        held_reviewed = [
            row
            for row in reviewed
            if row["session_group"] == positive_group
            or (
                row["label"] == 0
                and row["session_group"] not in positive_groups
                and negative_fold.get(row["session_group"]) == fold
            )
        ]
        held_ids = {row["id"] for row in held_reviewed}
        train_reviewed = [row for row in reviewed if row["id"] not in held_ids]
        folds.append(
            train_fold(
                fold,
                staged + train_reviewed,
                held_reviewed,
                args,
                device,
            )
        )
        torch.cuda.empty_cache()

    held_y = np.concatenate(
        [np.asarray(row["held_targets"], dtype=np.int64) for row in folds]
    )
    held_probability = np.concatenate(
        [np.asarray(row["held_probabilities"], dtype=np.float64) for row in folds]
    )
    point = operating_point(held_y, held_probability, args.min_cv_recall)
    threshold = float(point["threshold"])
    trainable_model = build_model(args.weight, args.initial_probe, torch.device("cpu"))
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainable_model.parameters()
        if parameter.requires_grad
    )
    report = {
        "schema_version": "dmc_swin3d_b_partial_finetune_cv_v1",
        "architecture": "swin3d_b_last_block_binary_v1",
        "trainable_scope": ["features.6.1", "norm", "head"],
        "trainable_parameters": int(trainable_parameters),
        "frozen_parameters": int(
            sum(parameter.numel() for parameter in trainable_model.parameters())
            - trainable_parameters
        ),
        "positive_session_groups": positive_groups,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "accumulate": args.accumulate,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
        },
        "threshold": threshold,
        "held_session_cv": metrics(held_y, held_probability, threshold),
        "held_session_at_0_5": metrics(held_y, held_probability, 0.5),
        "folds": folds,
        "promotion_eligible": False,
        "authority": "offline_diagnostic_only",
        "warnings": [
            "only two independent reviewed positive sessions are available",
            "fold checkpoints are evaluation artifacts, not deployment weights",
            "a new untouched positive session is required before shadow integration",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "trainable_parameters": report["trainable_parameters"],
        "threshold": threshold,
        "held_session_cv": report["held_session_cv"],
        "held_session_at_0_5": report["held_session_at_0_5"],
        "output": str((args.out_dir / "report.json").resolve()),
        "promotion_eligible": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
