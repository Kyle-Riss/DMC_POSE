#!/usr/bin/env python3
"""Create review-only multiview median proposals and flag camera disagreement."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


BOUNDARIES = (
    "proposed_fall_onset_frame",
    "proposed_impact_frame",
    "proposed_post_fall_stable_frame",
)
SPREAD_LIMIT_SEC = {
    "proposed_fall_onset_frame": 0.25,
    "proposed_impact_frame": 0.25,
    "proposed_post_fall_stable_frame": 0.50,
}


def build_consensus(annotation_rows: list[dict], proposal_rows: list[dict]) -> tuple[list[dict], dict]:
    annotations = {row["video_id"]: row for row in annotation_rows}
    if {row["video_id"] for row in proposal_rows} != set(annotations):
        raise ValueError("proposal and annotation video sets do not match")
    groups: dict[str, list[dict]] = defaultdict(list)
    for proposal in proposal_rows:
        annotation = annotations[proposal["video_id"]]
        groups[proposal["recording_id"]].append({"proposal": proposal, "annotation": annotation})

    outputs = []
    event_reports = []
    for recording_id, members in sorted(groups.items()):
        cameras = {member["annotation"]["camera_id"] for member in members}
        if len(members) != 3 or len(cameras) != 3:
            raise ValueError(f"recording {recording_id} must have exactly three camera views")
        consensus_sec = {}
        spread_sec = {}
        for boundary in BOUNDARIES:
            values = [int(member["proposal"][boundary]) / float(member["annotation"]["fps"]) for member in members]
            consensus_sec[boundary] = float(statistics.median(values))
            spread_sec[boundary] = float(max(values) - min(values))
        consistent = all(spread_sec[name] <= SPREAD_LIMIT_SEC[name] for name in BOUNDARIES)
        status = "multiview_consistent" if consistent else "needs_adjudication"
        for member in members:
            proposal = member["proposal"]
            annotation = member["annotation"]
            fps = float(annotation["fps"])
            frame_count = int(float(annotation["frame_count"]))
            converted = {name: min(frame_count - 1, max(0, round(seconds * fps))) for name, seconds in consensus_sec.items()}
            onset = converted["proposed_fall_onset_frame"]
            impact = max(onset, converted["proposed_impact_frame"])
            stable = max(impact, converted["proposed_post_fall_stable_frame"])
            uncertainty = max(1, round(0.15 * fps))
            outputs.append({
                "video_id": proposal["video_id"],
                "recording_id": recording_id,
                "scene_id": proposal["scene_id"],
                "proposed_fall_onset_frame": onset,
                "proposed_impact_frame": impact,
                "proposed_post_fall_stable_frame": stable,
                "proposed_fall_end_frame": frame_count - 1,
                "proposed_onset_earliest_frame": max(0, onset - uncertainty),
                "proposed_onset_latest_frame": min(impact, onset + uncertainty),
                "proposal_method": "multiview_median_motion_review_v1",
                "proposal_status": "review_required",
                "multiview_status": status,
                "onset_spread_sec": round(spread_sec["proposed_fall_onset_frame"], 6),
                "impact_spread_sec": round(spread_sec["proposed_impact_frame"], 6),
                "stable_spread_sec": round(spread_sec["proposed_post_fall_stable_frame"], 6),
            })
        event_reports.append({
            "recording_id": recording_id,
            "cameras": sorted(cameras),
            "status": status,
            "consensus_sec": {name: round(value, 6) for name, value in consensus_sec.items()},
            "spread_sec": {name: round(value, 6) for name, value in spread_sec.items()},
        })
    status_counts = Counter(row["status"] for row in event_reports)
    report = {
        "schema_version": "dmc_multiview_temporal_proposals_v1",
        "purpose": "review-only multiview consensus; never ground truth",
        "video_count": len(outputs),
        "recording_count": len(event_reports),
        "status_counts": dict(sorted(status_counts.items())),
        "spread_limits_sec": SPREAD_LIMIT_SEC,
        "events": event_reports,
        "limitations": [
            "median assumes clips share a recording-relative time origin",
            "camera disagreement is flagged rather than silently averaged into ground truth",
            "every boundary still requires human review",
        ],
    }
    return outputs, report


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1.csv")
    parser.add_argument("--proposals", type=Path, default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1_proposals.csv")
    parser.add_argument("--out", type=Path, default=project / "external_datasets/annotations/usb_sim_falldown_temporal_v1_multiview_proposals.csv")
    parser.add_argument("--report", type=Path, default=project / "docs/usb_sim_falldown_multiview_proposals_20260824.json")
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8-sig") as handle:
        annotations = list(csv.DictReader(handle))
    with args.proposals.open(newline="", encoding="utf-8-sig") as handle:
        proposals = list(csv.DictReader(handle))
    outputs, report = build_consensus(annotations, proposals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0]))
        writer.writeheader(); writer.writerows(outputs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("video_count", "recording_count", "status_counts", "spread_limits_sec")}, ensure_ascii=False, indent=2))
    print(f"proposals: {args.out.resolve()}")
    print(f"report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
