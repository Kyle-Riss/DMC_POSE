#!/usr/bin/env python3
"""Append train-only augmentation windows while freezing base val/test."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")


def load_npz(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(root / f"{split}.npz")
    return payload["x"].astype(np.float32), payload["y"].astype(np.int64)


def load_metadata(root: Path, split: str) -> list[dict]:
    return json.loads((root / f"{split}_metadata.json").read_text(encoding="utf-8"))


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=project / "external_datasets/windows/tcn_109_v2_no_missing/gmdcsa24_3s")
    parser.add_argument("--augment", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base_index = json.loads((args.base / "window_index.json").read_text(encoding="utf-8"))
    for root in args.augment:
        index = json.loads((root / "window_index.json").read_text(encoding="utf-8"))
        for key in ("feature_schema_version", "sequence_contract_version", "window_rows", "feature_count"):
            if index.get(key) != base_index.get(key):
                raise ValueError(f"window contract mismatch for {root}: {key}")
        for split in ("val", "test"):
            _, y = load_npz(root, split)
            if len(y):
                raise ValueError(f"augmentation must be train-only: {root}/{split} has {len(y)} windows")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = dict(base_index)
    summary["source_dir"] = str(args.out.resolve())
    summary["merge_policy"] = "base val/test frozen; augment train only"
    summary["base_windows_dir"] = str(args.base.resolve())
    summary["augmentation_windows_dirs"] = [str(path.resolve()) for path in args.augment]
    summary["splits"] = {}
    for split in SPLITS:
        x_parts, y_parts = [], []
        metadata = []
        roots = [args.base] + (args.augment if split == "train" else [])
        for root in roots:
            x, y = load_npz(root, split)
            x_parts.append(x)
            y_parts.append(y)
            metadata.extend(load_metadata(root, split))
        x = np.concatenate(x_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        np.savez_compressed(args.out / f"{split}.npz", x=x, y=y)
        (args.out / f"{split}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        counts = Counter(y.tolist())
        summary["splits"][split] = {
            "windows": len(y), "non_fall": counts.get(0, 0), "fall": counts.get(1, 0),
            "videos": len({row["video_id"] for row in metadata}),
        }
    (args.out / "window_index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
