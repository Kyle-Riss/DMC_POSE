#!/usr/bin/env python3
"""Fine-tune the frozen DMC Swin3D-B base for one fixed-camera site.

The input manifest is deliberately small and explicit.  Training clips may
change weights, validation clips select the epoch and threshold, and test
clips are read only after model selection.  A checkpoint without a test split
is always marked non-promotable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_swin3d_b_reviewed_cv import (
    ClipDataset,
    build_model,
    evaluate,
    forward_logits,
    seed_everything,
)
from scripts.train_swin3d_verifier_probe import metrics, operating_point


SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict, dict[str, list[dict]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dmc_site_finetune_manifest_v1":
        raise ValueError("manifest schema_version must be dmc_site_finetune_manifest_v1")
    if not str(payload.get("site_id", "")).strip():
        raise ValueError("manifest site_id is required")

    rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    group_splits: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    for index, raw in enumerate(payload.get("clips", [])):
        split = str(raw.get("split", ""))
        if split not in SPLITS:
            raise ValueError(f"clip {index}: split must be train, val, or test")
        clip_id = str(raw.get("clip_id", "")).strip()
        if not clip_id or clip_id in ids:
            raise ValueError(f"clip {index}: clip_id is missing or duplicated: {clip_id!r}")
        ids.add(clip_id)
        label = int(raw.get("label", -1))
        if label not in (0, 1):
            raise ValueError(f"clip {clip_id}: label must be 0 or 1")
        video_path = Path(str(raw.get("video_path", ""))).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"clip {clip_id}: video does not exist: {video_path}")
        start = int(raw.get("start_frame", -1))
        end = int(raw.get("end_frame", -1))
        if start < 0 or end <= start:
            raise ValueError(f"clip {clip_id}: invalid frame range {start}:{end}")
        group_id = str(raw.get("group_id", "")).strip()
        if not group_id:
            raise ValueError(f"clip {clip_id}: group_id is required to prevent leakage")
        group_splits[group_id].add(split)
        rows[split].append(
            {
                "source": "staged",
                "id": clip_id,
                "label": label,
                "video_path": str(video_path),
                "start_frame": start,
                "end_frame": end,
                "group_id": group_id,
            }
        )

    leaked = {group: sorted(parts) for group, parts in group_splits.items() if len(parts) > 1}
    if leaked:
        raise ValueError(f"group leakage across splits: {leaked}")
    for split in ("train", "val"):
        labels = {row["label"] for row in rows[split]}
        if labels != {0, 1}:
            raise ValueError(f"{split} must contain both fall and non-fall clips")
    if rows["test"] and {row["label"] for row in rows["test"]} != {0, 1}:
        raise ValueError("test must contain both classes when supplied")
    return payload, rows


def loader(rows: list[dict], args: argparse.Namespace, *, train: bool) -> DataLoader:
    return DataLoader(
        ClipDataset(rows, train=train),
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = build_model(args.weight, args.initial_probe, device)
    if args.base_checkpoint:
        checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def class_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {"clips": len(rows), "non_fall": counts[0], "fall": counts[1]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "runs/video_verifier/swin3d_b_partial_ft_all_reviewed_base_v1_20260828/model.pt",
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-val-recall", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    payload, rows = load_manifest(args.manifest)
    summary = {split: class_counts(rows[split]) for split in SPLITS}
    if args.validate_only:
        print(json.dumps({"site_id": payload["site_id"], "splits": summary}, indent=2))
        return 0
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")

    seed_everything(args.seed)
    device = torch.device(args.device)
    model = load_model(args, device)
    train_loader = loader(rows["train"], args, train=True)
    val_loader = loader(rows["val"], args, train=False)
    test_loader = loader(rows["test"], args, train=False) if rows["test"] else None

    backbone = list(model.features[6][1].parameters()) + list(model.norm.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    positives = summary["train"]["fall"]
    negatives = summary["train"]["non_fall"]
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(1, positives)], device=device)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / "model.pt"
    history, best_key = [], None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen = 0
        started = time.perf_counter()
        for step, (clips, target, _) in enumerate(train_loader, start=1):
            clips = clips.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(forward_logits(model, clips), target) / args.accumulate
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
            count = len(target)
            total_loss += float(loss.detach()) * args.accumulate * count
            seen += count

        y_val, p_val, _, _ = evaluate(model, val_loader, device)
        point = operating_point(y_val, p_val, args.min_val_recall)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "seconds": time.perf_counter() - started,
            "validation": metrics(y_val, p_val, float(point["threshold"])),
        }
        history.append(row)
        key = (row["validation"]["f1"], row["validation"]["precision"], row["validation"]["roc_auc"])
        if best_key is None or key > best_key:
            best_key = key
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": "swin3d_b_last_block_binary_v1",
                    "trainable_scope": ["features.6.1", "norm", "head"],
                    "site_id": payload["site_id"],
                    "selected_epoch": epoch,
                    "threshold": float(point["threshold"]),
                },
                best_path,
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    y_val, p_val, _, _ = evaluate(model, val_loader, device)
    threshold = float(checkpoint["threshold"])
    report = {
        "schema_version": "dmc_site_finetune_report_v1",
        "site_id": payload["site_id"],
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "base_checkpoint": str(args.base_checkpoint.resolve()) if args.base_checkpoint else None,
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "threshold_policy": f"max precision with validation recall >= {args.min_val_recall}",
        "splits": summary,
        "validation": metrics(y_val, p_val, threshold),
        "history": history,
        "model": str(best_path),
        "model_sha256": sha256(best_path),
        "authority": "offline_site_candidate",
        "promotion_eligible": False,
    }
    if test_loader is not None:
        y_test, p_test, ids_test, _ = evaluate(model, test_loader, device)
        report["test"] = metrics(y_test, p_test, threshold)
        np.savez_compressed(
            args.out_dir / "test_predictions.npz",
            y=y_test,
            probability=p_test,
            clip_id=np.asarray(ids_test),
        )
    else:
        report["warnings"] = ["no untouched site test split; shadow deployment is blocked"]
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
