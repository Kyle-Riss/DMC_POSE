#!/usr/bin/env python3
"""Export a FallTCN checkpoint to TorchScript and verify numerical parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from temporal_model import FallTCN


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class NormalizedTCN(torch.nn.Module):
    def __init__(self, model: FallTCN, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((x - self.mean) / self.std))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT / "runs/temporal_tcn/gmdcsa24_tcn/model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "artifacts/edge/rpi5/tcn_baseline_v1",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_count = int(checkpoint["feature_count"])
    model = FallTCN(feature_count).eval()
    model.load_state_dict(checkpoint["state_dict"])
    wrapped = NormalizedTCN(
        model,
        torch.as_tensor(checkpoint["mean"], dtype=torch.float32),
        torch.as_tensor(checkpoint["std"], dtype=torch.float32),
    ).eval()
    example = torch.from_numpy(np.random.default_rng(42).normal(size=(4, 30, feature_count)).astype(np.float32))
    with torch.inference_mode():
        reference = wrapped(example)
        traced = torch.jit.trace(wrapped, example, strict=True)
        candidate = traced(example)
    max_abs_error = float(torch.max(torch.abs(reference - candidate)))
    if max_abs_error > 1e-6:
        raise RuntimeError(f"TorchScript parity failed: max_abs_error={max_abs_error}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "fall_tcn_normalized.ts"
    traced.save(str(model_path))
    loaded = torch.jit.load(str(model_path), map_location="cpu").eval()
    with torch.inference_mode():
        reload_error = float(torch.max(torch.abs(reference - loaded(example))))
    if reload_error > 1e-6:
        raise RuntimeError(f"TorchScript reload parity failed: max_abs_error={reload_error}")

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": sha256(args.checkpoint),
        "artifact": str(model_path.resolve()),
        "artifact_sha256": sha256(model_path),
        "bytes": model_path.stat().st_size,
        "format": "torchscript",
        "input": ["batch", 30, feature_count],
        "output": "fall_probability",
        "normalization_embedded": True,
        "sequence_contract_version": checkpoint.get("sequence_contract_version"),
        "feature_schema_version": checkpoint.get("feature_schema_version"),
        "max_abs_error_export": max_abs_error,
        "max_abs_error_reload": reload_error,
        "parity_passed": True,
        "activation_status": "benchmark_required",
    }
    report_path = args.output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
