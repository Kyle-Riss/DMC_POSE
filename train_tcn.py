#!/usr/bin/env python3
"""Train a selected temporal model on subject-disjoint pose windows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from temporal_model import MODEL_ARCHITECTURES, architecture_from_checkpoint, build_temporal_model

def load_split(root: Path, split: str):
    data = np.load(root / f"{split}.npz")
    return data["x"].astype(np.float32), data["y"].astype(np.float32)


def report_metrics(y, probability, threshold):
    prediction = (probability >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    unique_classes = np.unique(y)
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y, prediction)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(y, probability)), 4) if len(unique_classes) == 2 else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def choose_threshold(y, probability, min_recall):
    feasible = []
    for threshold in np.unique(np.concatenate(([0.0], probability, [1.0]))):
        result = report_metrics(y, probability, float(threshold))
        if result["recall"] >= min_recall:
            feasible.append((result["precision"], result["f1"], float(threshold)))
    return max(feasible)[2] if feasible else 0.5


def probabilities(model, x, device, batch_size=256):
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
    output = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            output.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent
    parser.add_argument("--windows-dir", type=Path, default=project_root / "external_datasets/windows/tcn_109_v2_no_missing/gmdcsa24_3s")
    parser.add_argument("--out-dir", type=Path, default=project_root / "runs/temporal_tcn/gmdcsa24_tcn_v2_observed_only")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-val-recall", type=float, default=0.90)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--architecture", choices=MODEL_ARCHITECTURES, default="causal_tcn_v1")
    parser.add_argument("--run-purpose", choices=("candidate", "shadow_candidate", "smoke"), default="candidate")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)

    window_index = json.loads((args.windows_dir / "window_index.json").read_text(encoding="utf-8"))
    synthetic_smoke_fixture = bool(window_index.get("synthetic_smoke_fixture"))
    if synthetic_smoke_fixture and args.run_purpose != "smoke":
        raise ValueError("synthetic smoke fixtures require --run-purpose smoke")
    promotion_eligible = bool(
        args.run_purpose == "candidate"
        and not synthetic_smoke_fixture
        and window_index.get("promotion_eligible", True)
    )
    sequence_contract = str(window_index.get("sequence_contract_version") or "")
    if not sequence_contract.startswith("observed_only_"):
        raise ValueError("training requires a versioned observed-only window contract")
    train_x, train_y = load_split(args.windows_dir, "train")
    val_x, val_y = load_split(args.windows_dir, "val")
    test_x, test_y = load_split(args.windows_dir, "test")
    window_rows = int(window_index["window_rows"])
    sample_hz = float(window_index["sample_hz"])
    if train_x.ndim != 3 or train_x.shape[1] != window_rows:
        raise ValueError("training tensor does not match window_index window_rows")
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std[std < 1e-5] = 1.0
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    test_x = (test_x - mean) / std

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = build_temporal_model(args.architecture, train_x.shape[2]).to(device)
    initialization = "random"
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if int(initial.get("feature_count", train_x.shape[2])) != train_x.shape[2]:
            raise ValueError("initial checkpoint feature count mismatch")
        if initial.get("sequence_contract_version") != window_index["sequence_contract_version"]:
            raise ValueError("initial checkpoint sequence contract mismatch")
        if architecture_from_checkpoint(initial) != args.architecture:
            raise ValueError("initial checkpoint architecture mismatch")
        model.load_state_dict(initial["state_dict"])
        initialization = f"warm_start:{args.init_checkpoint.resolve()}"
    positive = float(train_y.sum())
    negative = float(len(train_y) - positive)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / max(positive, 1.0), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / "model.pt"
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_probability = probabilities(model, val_x, device)
        val_loss = float(nn.functional.binary_cross_entropy(torch.from_numpy(val_probability), torch.from_numpy(val_y)).item())
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(train_losses)), 6), "val_loss": round(val_loss, 6)})
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            stale = 0
            torch.save({
                "state_dict": model.state_dict(),
                "feature_count": train_x.shape[2],
                "feature_schema_version": window_index["feature_schema_version"],
                "sequence_contract_version": window_index["sequence_contract_version"],
                "architecture": args.architecture,
                "window_rows": window_rows,
                "sample_hz": sample_hz,
                "window_sec": float(window_index["window_sec"]),
                "run_purpose": args.run_purpose,
                "data_provenance": window_index.get("data_provenance"),
                "promotion_eligible": promotion_eligible,
                "mean": mean,
                "std": std,
            }, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_probability = probabilities(model, val_x, device)
    threshold = choose_threshold(val_y.astype(np.int64), val_probability, args.min_val_recall)
    test_probability = probabilities(model, test_x, device)
    report = {
        "model": args.architecture,
        "feature_schema_version": window_index["feature_schema_version"],
        "sequence_contract_version": window_index["sequence_contract_version"],
        "window_rows": window_rows,
        "sample_hz": sample_hz,
        "window_sec": float(window_index["window_sec"]),
        "windows_dir": str(args.windows_dir.resolve()),
        "device": str(device),
        "initialization": initialization,
        "run_purpose": args.run_purpose,
        "data_provenance": window_index.get("data_provenance"),
        "promotion_eligible": promotion_eligible,
        "accuracy_claim": promotion_eligible,
        "data_warnings": list(window_index.get("warnings", [])),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_completed": len(history),
        "best_val_loss": round(best_loss, 6),
        "threshold_policy": f"max precision on validation with recall >= {args.min_val_recall}",
        "validation": report_metrics(val_y.astype(np.int64), val_probability, threshold),
        "test": report_metrics(test_y.astype(np.int64), test_probability, threshold),
        "history": history,
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "device", "run_purpose", "promotion_eligible", "parameter_count", "epochs_completed", "best_val_loss", "validation", "test")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
