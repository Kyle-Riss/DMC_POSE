#!/usr/bin/env python3
"""Create a reproducible pre-deployment audit of temporal training assets."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def nonempty(row: dict, fields: tuple[str, ...]) -> bool:
    return all(str(row.get(field, "")).strip() for field in fields)


def manifest_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["video_id"]) for row in payload.get("items", [])}


def window_summary(root: Path) -> dict:
    result = {}
    for split in ("train", "val", "test"):
        payload = np.load(root / f"{split}.npz")
        y = payload["y"].astype(int)
        metadata = json.loads((root / f"{split}_metadata.json").read_text(encoding="utf-8"))
        counts = Counter(y.tolist())
        result[split] = {
            "windows": int(len(y)),
            "non_fall": int(counts.get(0, 0)),
            "fall": int(counts.get(1, 0)),
            "videos": len({str(row["video_id"]) for row in metadata}),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "runs/temporal_tcn/central_predeploy_data_audit_20260807.json",
    )
    args = parser.parse_args()

    annotations = PROJECT / "external_datasets/FallVision/annotations"
    annotation_files = {
        "manual_pilot": annotations / "fallvision_pilot_v1.csv",
        "manual_pilot_complete": annotations / "fallvision_pilot_v1_complete.csv",
        "round2_proposals": annotations / "fallvision_round2_120_v1.csv",
        "weak_train": annotations / "fallvision_weak_train_v1.csv",
        "non_fall_train": annotations / "fallvision_non_fall_train_v1.csv",
    }
    temporal_fields = (
        "fall_onset_frame",
        "impact_frame",
        "post_fall_stable_frame",
        "fall_end_frame",
    )
    annotation_summary = {}
    for name, path in annotation_files.items():
        rows = read_csv(path)
        annotation_summary[name] = {
            "path": str(path),
            "rows": len(rows),
            "status": dict(Counter(str(row.get("annotation_status", "")) for row in rows)),
            "complete_temporal_rows": sum(nonempty(row, temporal_fields) for row in rows),
        }

    manifests = PROJECT / "external_datasets/manifests"
    pilot = manifest_ids(manifests / "fallvision_pilot_manual_diagnostic_v1.json")
    weak = manifest_ids(manifests / "fallvision_weak_train_v1.json")
    negative = manifest_ids(manifests / "fallvision_non_fall_train_v1.json")
    windows_root = PROJECT / "external_datasets/windows/tcn_109_v2_no_missing"
    window_sets = {
        name: window_summary(windows_root / name)
        for name in (
            "gmdcsa24_3s",
            "fallvision_pilot_manual_diagnostic_v1_3s",
            "fallvision_pilot_balanced_diagnostic_v1_3s",
            "fallvision_weak_train_v1_3s",
            "fallvision_non_fall_train_v1_3s",
            "gmdcsa24_fallvision_weak_v1_3s",
        )
    }

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sequence_contract": "observed_only_10hz_v2",
        "annotations": annotation_summary,
        "data_roles": {
            "manual_pilot_24": "frozen external diagnostic; never train",
            "round2_120": "unreviewed proposal/weak label only; never ground truth",
            "fallvision_weak_train": "train-only weak positive augmentation",
            "fallvision_non_fall_train": "train-only video-level negative augmentation",
            "gmdcsa24_val_test": "frozen operating-point selection and test",
        },
        "leakage_checks": {
            "manual_vs_weak_overlap": len(pilot & weak),
            "manual_vs_negative_overlap": len(pilot & negative),
            "weak_vs_negative_overlap": len(weak & negative),
            "passed": not (pilot & weak or pilot & negative or weak & negative),
        },
        "window_sets": window_sets,
        "promotion_decision": {
            "current_live_checkpoint": "keep_shadow_baseline",
            "mixed_weak_models": "not_promotable",
            "reasons": [
                "FallVision participant identity remains unresolved",
                "only 9/24 manual fall videos produced evaluable windows",
                "pre-onset-ready coverage is zero on the frozen pilot",
                "mixed weak-label models increased event false positives",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
