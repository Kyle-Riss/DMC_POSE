#!/usr/bin/env python3
"""Validate the frozen temporal feature and observed-only sequence contract."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_temporal import TemporalShadowRunner
from temporal_features import FEATURE_COUNT, FEATURE_SCHEMA_VERSION, temporal_feature_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "config" / "temporal_contract_v2.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    names = temporal_feature_names()

    errors: list[str] = []
    if contract["feature_schema"] != FEATURE_SCHEMA_VERSION:
        errors.append("feature_schema_mismatch")
    if contract["feature_count"] != FEATURE_COUNT or len(names) != FEATURE_COUNT:
        errors.append("feature_count_mismatch")

    cursor = 0
    for block in contract["feature_blocks"]:
        if int(block["start"]) != cursor:
            errors.append(f"feature_block_gap:{block['name']}")
        cursor = int(block["end_exclusive"])
    if cursor != FEATURE_COUNT:
        errors.append("feature_blocks_do_not_end_at_feature_count")

    for model_key in ("pose_model", "posture_model"):
        model = contract[model_key]
        path = root / model["path"]
        if not path.is_file():
            errors.append(f"missing_model:{model_key}")
        elif sha256(path) != model["sha256"]:
            errors.append(f"model_hash_mismatch:{model_key}")

    defaults = inspect.signature(TemporalShadowRunner.__init__).parameters
    if float(defaults["min_interval_sec"].default) != float(contract["minimum_interval_sec"]):
        errors.append("live_min_interval_mismatch")
    if float(defaults["max_interval_sec"].default) != float(contract["maximum_interval_sec"]):
        errors.append("live_max_interval_mismatch")
    if contract["missing_observation_row"] != "forbidden":
        errors.append("missing_row_not_forbidden")
    if contract["previous_pose_copy"] is not False:
        errors.append("previous_pose_copy_not_forbidden")

    result = {
        "ok": not errors,
        "contract": str(contract_path),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_count": len(names),
        "feature_first": names[0],
        "feature_last": names[-1],
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
