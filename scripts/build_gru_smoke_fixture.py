#!/usr/bin/env python3
"""Create a deterministic synthetic fixture for exercising the GRU trainer only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from temporal_features import FEATURE_SCHEMA_VERSION, temporal_feature_names
from temporal_sequence import observed_sequence_contract


def make_split(count: int, *, seed: int, rows: int = 80, features: int = 109):
    rng = np.random.default_rng(seed)
    y = (np.arange(count) % 2).astype(np.int64)
    x = rng.normal(0.0, 0.15, size=(count, rows, features)).astype(np.float32)
    ramp = np.linspace(0.0, 1.0, rows - 50, dtype=np.float32)
    for index in np.flatnonzero(y == 1):
        x[index, 50:, :8] += ramp[:, None] * 1.5
        x[index, 50:, 74] = 1.0
    return x, y


def build_fixture(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": 96, "val": 32, "test": 32}
    for offset, (split, count) in enumerate(counts.items()):
        x, y = make_split(count, seed=4200 + offset)
        np.savez_compressed(out_dir / f"{split}.npz", x=x, y=y)
        (out_dir / f"{split}_metadata.json").write_text("[]", encoding="utf-8")
    index = {
        "window_schema_version": "pose_gru_smoke_fixture_v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sequence_contract_version": observed_sequence_contract(20.0),
        "data_provenance": "deterministic_synthetic_smoke_fixture",
        "synthetic_smoke_fixture": True,
        "promotion_eligible": False,
        "prohibited_claims": ["accuracy", "recall", "precision", "clinical_performance"],
        "window_sec": 4.0,
        "stride_sec": 0.05,
        "sample_hz": 20.0,
        "window_rows": 80,
        "stride_rows": 1,
        "feature_count": 109,
        "feature_names": temporal_feature_names(),
        "splits": {split: {"windows": count, "non_fall": count // 2, "fall": count // 2} for split, count in counts.items()},
    }
    (out_dir / "window_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=project / "external_datasets/windows/smoke/gru_80x109_20hz_v1")
    args = parser.parse_args()
    index = build_fixture(args.out_dir)
    print(json.dumps({"out_dir": str(args.out_dir.resolve()), "promotion_eligible": index["promotion_eligible"], "splits": index["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
