#!/usr/bin/env python3
"""Export numeric StandardScaler state from a pickle without unpickling it."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickletools
from pathlib import Path

import numpy as np


def extract_arrays(path: Path, feature_count: int) -> dict[str, np.ndarray]:
    payload = path.read_bytes()
    wanted = {"mean_": "mean", "scale_": "scale"}
    active = None
    arrays: dict[str, np.ndarray] = {}
    for opcode, argument, _ in pickletools.genops(payload):
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8"} and argument in wanted:
            active = wanted[str(argument)]
            continue
        if active and opcode.name in {"SHORT_BINBYTES", "BINBYTES", "BINBYTES8"}:
            raw = bytes(argument)
            if len(raw) == feature_count * 8:
                arrays[active] = np.frombuffer(raw, dtype="<f8").copy()
                active = None
    if set(arrays) != {"mean", "scale"}:
        raise ValueError(f"could not safely locate scaler arrays: {sorted(arrays)}")
    if not np.isfinite(arrays["mean"]).all() or not np.isfinite(arrays["scale"]).all():
        raise ValueError("scaler contains non-finite values")
    if np.any(arrays["scale"] <= 0.0):
        raise ValueError("scaler contains non-positive scale")
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", type=Path, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arrays = extract_arrays(args.pickle, args.features)
    digest = hashlib.sha256(args.pickle.read_bytes()).hexdigest()
    output = {
        "format": "dmc_safe_standard_scaler_v1",
        "source_pickle": str(args.pickle),
        "source_sha256": digest,
        "feature_count": args.features,
        "mean": arrays["mean"].tolist(),
        "scale": arrays["scale"].tolist(),
        "security": "numeric arrays extracted with pickletools; pickle was not executed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("format", "source_sha256", "feature_count", "security")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
