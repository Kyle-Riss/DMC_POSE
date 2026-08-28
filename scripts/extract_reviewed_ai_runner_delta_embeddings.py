#!/usr/bin/env python3
"""Extract first-to-last Swin3D delta embeddings from reviewed AI_runner events."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swin3d_verifier import Swin3DVerifierService


EVENT_RE = re.compile(r"^(bed_[^_]+)_fall_(\d{8})_(\d{6})$")


def read_frames(event_dir: Path) -> list[np.ndarray]:
    frames = []
    for path in sorted((event_dir / "frames").glob("*.jpg")):
        frame = cv2.imread(str(path))
        if frame is not None:
            frames.append(frame)
    return frames


def load_labels(paths: list[Path]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for event_id, value in document["labels"].items():
            previous = labels.get(event_id)
            if previous is not None and previous != value:
                raise ValueError(f"conflicting labels for {event_id}: {previous} != {value}")
            labels[event_id] = value
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        default=sorted(
            (PROJECT_ROOT / "runtime_data/ai_runner_fall_review_v1").glob(
                "reviewed_labels*.json"
            )
        ),
    )
    parser.add_argument(
        "--events-root",
        type=Path,
        default=Path("/home/dmc/AI/AI_runner/data/events/fall"),
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
        / "runs/video_verifier/swin3d_b_staged_delta_v2_20260828/delta_probe.npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "external_datasets/features/swin3d_b_verifier/ai_runner_reviewed_v1.npz",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels)
    service = Swin3DVerifierService(args.weight, args.probe, device=args.device)
    features = []
    targets = []
    event_ids = []
    bed_ids = []
    session_groups = []
    review_confidences = []
    excluded = []
    for event_id, (label, review_confidence) in labels.items():
        match = EVENT_RE.match(event_id)
        if match is None:
            excluded.append({"event_id": event_id, "reason": "invalid_event_id"})
            continue
        frames = read_frames(args.events_root / event_id)
        if len(frames) < 32:
            excluded.append({"event_id": event_id, "reason": "fewer_than_32_frames"})
            continue
        baseline = service._embedding(frames[:16])
        post = service._embedding(frames[-16:])
        features.append(post - baseline)
        targets.append(int(label == "fall"))
        event_ids.append(event_id)
        bed_ids.append(match.group(1))
        date, time = match.group(2), match.group(3)
        session_groups.append(f"{date}_{time[:2]}")
        review_confidences.append(review_confidence)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=np.asarray(features, dtype=np.float32),
        y=np.asarray(targets, dtype=np.int64),
        event_id=np.asarray(event_ids),
        bed_id=np.asarray(bed_ids),
        session_group=np.asarray(session_groups),
        review_confidence=np.asarray(review_confidences),
    )
    summary = {
        "schema_version": "dmc_ai_runner_reviewed_delta_embeddings_v1",
        "feature_mode": "delta_embedding_v1",
        "labels": [str(path.resolve()) for path in args.labels],
        "events": len(event_ids),
        "positive": int(sum(targets)),
        "negative": int(len(targets) - sum(targets)),
        "session_groups": sorted(set(session_groups)),
        "excluded": excluded,
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
