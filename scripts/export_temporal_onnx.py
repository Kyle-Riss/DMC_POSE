#!/usr/bin/env python3
"""Export a DMC temporal checkpoint with normalization and verify ONNX parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from temporal_model import architecture_from_checkpoint, build_temporal_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NormalizedProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, -1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        normalized = (sequence - self.mean) / self.std
        return torch.sigmoid(self.model(normalized))


def export_checkpoint(checkpoint_path: Path, out_path: Path, *, allow_non_promotion: bool = False) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    promotion_eligible = bool(checkpoint.get("promotion_eligible", False))
    if not promotion_eligible and not allow_non_promotion:
        raise ValueError("checkpoint is not promotion eligible; pass --allow-non-promotion for smoke export")
    architecture = architecture_from_checkpoint(checkpoint)
    feature_count = int(checkpoint["feature_count"])
    window_rows = int(checkpoint["window_rows"])
    model = build_temporal_model(architecture, feature_count)
    model.load_state_dict(checkpoint["state_dict"])
    wrapper = NormalizedProbabilityModel(model, checkpoint["mean"], checkpoint["std"]).eval()
    generator = torch.Generator().manual_seed(20260824)
    sample = torch.randn(3, window_rows, feature_count, generator=generator)
    with torch.no_grad():
        expected = wrapper(sample).numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample[:1],
        out_path,
        input_names=["pose_sequence"],
        output_names=["fall_probability"],
        dynamic_axes={"pose_sequence": {0: "batch"}, "fall_probability": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    actual = session.run(["fall_probability"], {"pose_sequence": sample.numpy()})[0]
    max_abs_error = float(np.max(np.abs(expected - actual)))
    parity_pass = bool(np.allclose(expected, actual, atol=1e-5, rtol=1e-5))
    if not parity_pass:
        raise ValueError(f"ONNX parity failed: max_abs_error={max_abs_error}")
    report = {
        "schema_version": "dmc_temporal_onnx_export_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "onnx": str(out_path.resolve()),
        "onnx_sha256": sha256(out_path),
        "architecture": architecture,
        "input_name": "pose_sequence",
        "input_shape": ["batch", window_rows, feature_count],
        "output_name": "fall_probability",
        "normalization_embedded": True,
        "sigmoid_embedded": True,
        "run_purpose": checkpoint.get("run_purpose", "legacy_unspecified"),
        "promotion_eligible": promotion_eligible,
        "parity_samples": len(sample),
        "parity_max_abs_error": max_abs_error,
        "parity_pass": parity_pass,
    }
    report_path = out_path.with_suffix(out_path.suffix + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-non-promotion", action="store_true")
    args = parser.parse_args()
    report = export_checkpoint(args.checkpoint.resolve(), args.out.resolve(), allow_non_promotion=args.allow_non_promotion)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
