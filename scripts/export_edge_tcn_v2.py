#!/usr/bin/env python3
"""Export a normalized FallTCN to TorchScript with reload parity verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from temporal_model import FallTCN  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


class NormalizedTCN(torch.nn.Module):
    def __init__(self, model: FallTCN, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model((inputs - self.mean) / self.std))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "runs/temporal_tcn/gmdcsa24_tcn/model.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "artifacts/edge/rpi5/tcn_baseline_v1")
    args = parser.parse_args()
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features = int(source["feature_count"])
    model = FallTCN(features).eval()
    model.load_state_dict(source["state_dict"])
    wrapped = NormalizedTCN(
        model,
        torch.as_tensor(source["mean"], dtype=torch.float32),
        torch.as_tensor(source["std"], dtype=torch.float32),
    ).eval()
    sample = torch.from_numpy(np.random.default_rng(42).normal(size=(8, 30, features)).astype(np.float32))
    with torch.inference_mode():
        expected = wrapped(sample)
        traced = torch.jit.trace(wrapped, sample, strict=True)
        export_error = float(torch.max(torch.abs(expected - traced(sample))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "fall_tcn_normalized.ts"
    traced.save(str(artifact))
    reloaded = torch.jit.load(str(artifact), map_location="cpu").eval()
    with torch.inference_mode():
        reload_error = float(torch.max(torch.abs(expected - reloaded(sample))))
    if max(export_error, reload_error) > 1e-6:
        raise RuntimeError(f"parity failed: export={export_error} reload={reload_error}")
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": digest(args.checkpoint),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": digest(artifact),
        "bytes": artifact.stat().st_size,
        "format": "torchscript",
        "input_shape": ["batch", 30, features],
        "output": "fall_probability",
        "normalization_embedded": True,
        "sequence_contract_version": source.get("sequence_contract_version"),
        "feature_schema_version": source.get("feature_schema_version"),
        "max_abs_error_export": export_error,
        "max_abs_error_reload": reload_error,
        "parity_passed": True,
        "activation_status": "benchmark_required",
    }
    (args.output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
