#!/usr/bin/env python3
"""Evaluate named checkpoints without tuning on the diagnostic dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_temporal_events import evaluate_split


def parse_model(value: str) -> tuple[str, Path, float, int]:
    parts = value.split("::")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("model must be NAME::MODEL_DIR::THRESHOLD::PERSISTENCE")
    return parts[0], Path(parts[1]), float(parts[2]), int(parts[3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merge-gap-sec", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    results = {}
    for name, model_dir, threshold, persistence in args.model:
        report = evaluate_split(
            "test", args.windows_dir, manifest, model_dir / "model.pt",
            threshold, persistence, device, args.merge_gap_sec,
        )
        results[name] = {key: value for key, value in report.items() if key != "per_video"}
    payload = {
        "scope": "FallVision diagnostic only; not promotion eligible",
        "warning": "manual positive pilot calibrated the boundary proposer; subject identity unresolved",
        "windows_dir": str(args.windows_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "operating_point_policy": "frozen on GMDCSA validation before this diagnostic evaluation",
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"diagnostic_report: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
