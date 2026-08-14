#!/usr/bin/env python3
"""Correct-path wrapper for the deterministic pre-deployment data audit."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def csv_summary(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = ("fall_onset_frame", "impact_frame", "post_fall_stable_frame", "fall_end_frame")
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "status": dict(Counter(str(row.get("annotation_status", "")) for row in rows)),
        "complete_temporal_rows": sum(all(str(row.get(k, "")).strip() for k in fields) for row in rows),
    }


def ids(name: str) -> set[str]:
    path = PROJECT / "external_datasets/manifests" / f"{name}.json"
    return {str(row["video_id"]) for row in json.loads(path.read_text(encoding="utf-8"))["items"]}


def windows(name: str) -> dict:
    root = PROJECT / "external_datasets/windows/tcn_109_v2_no_missing" / name
    answer = {}
    for split in ("train", "val", "test"):
        labels = np.load(root / f"{split}.npz")["y"].astype(int)
        metadata = json.loads((root / f"{split}_metadata.json").read_text(encoding="utf-8"))
        count = Counter(labels.tolist())
        answer[split] = {
            "windows": int(len(labels)), "non_fall": int(count.get(0, 0)),
            "fall": int(count.get(1, 0)), "videos": len({str(row["video_id"]) for row in metadata}),
        }
    return answer


def main() -> int:
    ann = PROJECT / "external_datasets/annotations"
    annotations = {
        key: csv_summary(ann / filename)
        for key, filename in {
            "manual_pilot": "fallvision_pilot_v1.csv",
            "manual_pilot_complete": "fallvision_pilot_v1_complete.csv",
            "round2_proposals": "fallvision_round2_120_v1.csv",
            "weak_train": "fallvision_weak_train_v1.csv",
            "non_fall_train": "fallvision_non_fall_train_v1.csv",
        }.items()
    }
    pilot, weak, negative = (
        ids("fallvision_pilot_manual_diagnostic_v1"), ids("fallvision_weak_train_v1"),
        ids("fallvision_non_fall_train_v1"),
    )
    names = (
        "gmdcsa24_3s", "fallvision_pilot_manual_diagnostic_v1_3s",
        "fallvision_pilot_balanced_diagnostic_v1_3s", "fallvision_weak_train_v1_3s",
        "fallvision_non_fall_train_v1_3s", "gmdcsa24_fallvision_weak_v1_3s",
    )
    report = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sequence_contract": "observed_only_10hz_v2",
        "annotations": annotations,
        "data_roles": {
            "manual_pilot_24": "frozen external diagnostic; never train",
            "round2_120": "unreviewed proposal/weak label only; never ground truth",
            "weak_positive_72": "train-only; proposal boundary",
            "non_fall_36": "train-only; video-level negative",
        },
        "leakage_checks": {
            "manual_vs_weak_overlap": len(pilot & weak),
            "manual_vs_negative_overlap": len(pilot & negative),
            "weak_vs_negative_overlap": len(weak & negative),
            "passed": not bool((pilot & weak) or (pilot & negative) or (weak & negative)),
        },
        "window_sets": {name: windows(name) for name in names},
        "promotion_decision": {
            "live_checkpoint": "keep current shadow baseline",
            "weak_mixed_models": "not promotable",
            "reasons": [
                "participant identity unresolved", "manual pilot evaluable windows only 9/24",
                "manual pilot pre-onset-ready coverage zero", "mixed weak models increased event false positives",
            ],
        },
    }
    output = PROJECT / "runs/temporal_tcn/central_predeploy_data_audit_20260807.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
