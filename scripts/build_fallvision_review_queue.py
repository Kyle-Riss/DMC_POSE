#!/usr/bin/env python3
"""Select human-review clips that can satisfy a temporal persistence gate."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def possible_event_windows(
    onset_frame: int,
    end_frame: int,
    fps: float,
    *,
    window_rows: int = 30,
    stride_rows: int = 5,
    sample_hz: float = 10.0,
) -> int:
    """Upper bound before Pose gaps, using the actual 10 Hz window-end grid."""
    onset_sec = onset_frame / fps
    end_sec = end_frame / fps
    first_end = (window_rows - 1) / sample_hz
    stride_sec = stride_rows / sample_hz
    if end_sec < max(onset_sec, first_end):
        return 0
    first_index = max(0, math.ceil((onset_sec - first_end) / stride_sec - 1e-9))
    last_index = math.floor((end_sec - first_end) / stride_sec + 1e-9)
    return max(0, last_index - first_index + 1)


def build_queue(
    annotations: list[dict[str, str]],
    proposals: list[dict[str, str]],
    *,
    min_persistence: int = 2,
    limit: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict]]:
    proposal_by_id = {row["video_id"]: row for row in proposals}
    candidates = []
    for row in annotations:
        if row.get("annotation_status") != "unreviewed":
            continue
        proposal = proposal_by_id.get(row["video_id"])
        if proposal is None or proposal.get("proposal_status") != "review_required":
            continue
        capacity = possible_event_windows(
            int(proposal["proposed_fall_onset_frame"]),
            int(proposal["proposed_fall_end_frame"]),
            float(row["fps"]),
        )
        if capacity < min_persistence:
            continue
        candidates.append((
            -capacity,
            row["scene_id"],
            -float(row["duration_sec"]),
            row["video_id"],
            row,
            proposal,
            capacity,
        ))
    candidates.sort(key=lambda value: value[:4])
    if limit > 0:
        candidates = candidates[:limit]
    queue = [value[4] for value in candidates]
    queue_proposals = [value[5] for value in candidates]
    audit = [{
        "video_id": value[4]["video_id"],
        "scene_id": value[4]["scene_id"],
        "chunk_id": value[4]["chunk_id"],
        "duration_sec": float(value[4]["duration_sec"]),
        "proposed_fall_onset_frame": int(value[5]["proposed_fall_onset_frame"]),
        "proposed_fall_end_frame": int(value[5]["proposed_fall_end_frame"]),
        "persistence_capacity_upper_bound": value[6],
    } for value in candidates]
    return queue, queue_proposals, audit


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=project / "external_datasets/annotations/fallvision_round2_120_v1.csv")
    parser.add_argument("--proposals", type=Path, default=project / "external_datasets/annotations/fallvision_round2_120_v1_proposals.csv")
    parser.add_argument("--out-annotations", type=Path, default=project / "external_datasets/annotations/fallvision_round2_persistence20_v1.csv")
    parser.add_argument("--out-proposals", type=Path, default=project / "external_datasets/annotations/fallvision_round2_persistence20_v1_proposals.csv")
    parser.add_argument("--report", type=Path, default=project / "external_datasets/annotations/fallvision_round2_persistence20_v1_report.json")
    parser.add_argument("--min-persistence", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    annotations = read_csv(args.annotations)
    proposals = read_csv(args.proposals)
    queue, queue_proposals, audit = build_queue(
        annotations, proposals,
        min_persistence=args.min_persistence,
        limit=args.limit,
    )
    if not queue:
        raise ValueError("no eligible unreviewed clips")
    write_csv(args.out_annotations, queue, list(annotations[0]))
    write_csv(args.out_proposals, queue_proposals, list(proposals[0]))
    scenes = Counter(row["scene_id"] for row in queue)
    report = {
        "schema_version": "fallvision_human_review_queue_v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "purpose": "human review queue; proposals are never ground truth",
        "source_annotations": str(args.annotations.resolve()),
        "source_proposals": str(args.proposals.resolve()),
        "output_annotations": str(args.out_annotations.resolve()),
        "output_proposals": str(args.out_proposals.resolve()),
        "min_persistence": args.min_persistence,
        "selected_count": len(queue),
        "scene_counts": dict(sorted(scenes.items())),
        "pre_onset_context_warning": "no selected Round 2 clip has 3 seconds of pre-onset context",
        "selection_metric": "upper bound only; actual observed-only Pose extraction may reduce capacity",
        "items": audit,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_count", "scene_counts", "min_persistence", "pre_onset_context_warning")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
