#!/usr/bin/env python3
"""Extract full-event Swin3D embeddings from all reviewed AI_runner events."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_reviewed_ai_runner_delta_embeddings import load_labels, read_frames
from swin3d_verifier import Swin3DVerifierService


EVENT_RE = re.compile(r"^(bed_[^_]+)_fall_(\d{8})_(\d{6})$")


def main() -> int:
    review_dir = PROJECT_ROOT / "runtime_data/ai_runner_fall_review_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, nargs="+", default=sorted(review_dir.glob("reviewed_labels*.json"))
    )
    parser.add_argument(
        "--events-root", type=Path, default=Path("/home/dmc/AI/AI_runner/data/events/fall")
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=PROJECT_ROOT / "external_models/torchvision/swin3d_b_22k-7c6ae6fa.pth",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=PROJECT_ROOT
        / "runs/video_verifier/swin3d_b_staged_linear_v1_20260828/linear_probe.npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "external_datasets/features/swin3d_b_verifier/ai_runner_reviewed_single_v1.npz",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels)
    service = Swin3DVerifierService(args.weight, args.probe, device=args.device)
    features, targets, event_ids, bed_ids, groups, confidences = [], [], [], [], [], []
    for event_id, (label, confidence) in labels.items():
        match = EVENT_RE.match(event_id)
        if match is None:
            continue
        frames = read_frames(args.events_root / event_id)
        if len(frames) < 16:
            continue
        features.append(service._embedding(frames))
        targets.append(int(label == "fall"))
        event_ids.append(event_id)
        bed_ids.append(match.group(1))
        groups.append(f"{match.group(2)}_{match.group(3)[:2]}")
        confidences.append(confidence)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(targets, dtype=np.int64),
        event_id=np.asarray(event_ids),
        bed_id=np.asarray(bed_ids),
        session_group=np.asarray(groups),
        review_confidence=np.asarray(confidences),
    )
    summary = {
        "schema_version": "dmc_ai_runner_reviewed_single_embeddings_v1",
        "feature_mode": "single_embedding_v1",
        "events": len(event_ids),
        "positive": int(sum(targets)),
        "negative": int(len(targets) - sum(targets)),
        "session_groups": sorted(set(groups)),
        "output": str(args.out.resolve()),
        "promotion_eligible": False,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
