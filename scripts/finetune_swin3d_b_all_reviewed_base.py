#!/usr/bin/env python3
"""Build a one-epoch all-reviewed Swin3D-B adaptation base without claiming evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.finetune_swin3d_b_reviewed_cv import (
    ClipDataset,
    build_model,
    forward_logits,
    reviewed_rows,
    seed_everything,
    staged_rows,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        / "runs/video_verifier/swin3d_b_partial_ft_all_reviewed_base_v1_20260828",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    seed_everything(args.seed)
    device = torch.device(args.device)

    rows = staged_rows(args.manifest) + reviewed_rows(args.labels, args.events_root)
    loader = DataLoader(
        ClipDataset(rows, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = build_model(args.weight, args.initial_probe, device)
    backbone_parameters = list(model.features[6][1].parameters()) + list(model.norm.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    positives = sum(int(row["label"]) for row in rows)
    negatives = len(rows) - positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(1, positives)], device=device)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, seen = 0.0, 0
    started = time.perf_counter()
    for step, (clips, target, _) in enumerate(loader, start=1):
        clips = clips.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(
            device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            loss = criterion(forward_logits(model, clips), target) / args.accumulate
        scaler.scale(loss).backward()
        if step % args.accumulate == 0 or step == len(loader):
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "swin3d_b_last_block_binary_v1",
            "trainable_scope": ["features.6.1", "norm", "head"],
            "base_weight": str(args.weight.resolve()),
            "initial_probe": str(args.initial_probe.resolve()),
            "epochs": 1,
        },
        model_path,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = {
        "schema_version": "dmc_swin3d_b_all_reviewed_adaptation_base_v1",
        "architecture": "swin3d_b_last_block_binary_v1",
        "purpose": "future fixed-camera domain-adaptation warm start",
        "trainable_scope": ["features.6.1", "norm", "head"],
        "trainable_parameters": int(trainable_parameters),
        "training": {
            "epochs": 1,
            "samples": len(rows),
            "positive": positives,
            "negative": negatives,
            "loss": total_loss / max(1, seen),
            "seconds": time.perf_counter() - started,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
        },
        "model": str(model_path.resolve()),
        "sha256": sha256(model_path),
        "promotion_eligible": False,
        "authority": "offline_warm_start_only",
        "warnings": [
            "all reviewed events were used for adaptation, so this checkpoint has no untouched local test",
            "use the frozen checkpoint as fallback",
            "validate on a new installation session before any shadow integration",
        ],
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
