#!/usr/bin/env python3
"""Apply an explicit human/protocol review to one temporal session manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path


NEGATIVE_LABELS = {
    "NORMAL_ENTRY_PRESENCE_EXIT",
    "NORMAL_SIT",
    "NORMAL_LIE",
    "CROUCH",
    "FLOOR_SIT",
    "PICKUP",
    "ASSISTED_MOVEMENT",
}
POSITIVE_LABELS = {"FALL", "BED_EXIT_FALL"}
LABELS = sorted(NEGATIVE_LABELS | POSITIVE_LABELS)


def reviewed_manifest(
    manifest: dict,
    *,
    label: str,
    reviewer: str,
    notes: str = "",
    onset_sec: float | None = None,
    impact_sec: float | None = None,
    end_sec: float | None = None,
) -> dict:
    if label not in LABELS:
        raise ValueError(f"unsupported label: {label}")
    if not reviewer.strip():
        raise ValueError("reviewer must be non-empty")
    duration = float(manifest.get("duration_sec", 0.0))
    is_fall = label in POSITIVE_LABELS
    boundaries = (onset_sec, impact_sec, end_sec)
    if is_fall:
        if any(value is None for value in boundaries):
            raise ValueError("fall labels require onset, impact, and end seconds")
        onset, impact, end = (float(value) for value in boundaries)
        if not 0.0 <= onset <= impact <= end <= duration:
            raise ValueError("fall boundaries must satisfy 0 <= onset <= impact <= end <= duration")
    elif any(value is not None for value in boundaries):
        raise ValueError("non-fall labels must not contain fall boundaries")

    result = dict(manifest)
    previous_review = {
        "label": result.get("label"),
        "binary_fall_label": result.get("binary_fall_label"),
        "review_status": result.get("review_status", "unreviewed"),
        "reviewed_at": result.get("reviewed_at"),
        "reviewer": result.get("reviewer"),
        "review_notes": result.get("review_notes", ""),
    }
    history = list(result.get("review_history", []))
    if previous_review["label"] != "UNREVIEWED" or previous_review["reviewed_at"]:
        history.append(previous_review)
    reviewed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    result.update({
        "label": label,
        "binary_fall_label": int(is_fall),
        "review_status": "reviewed",
        "reviewed_at": reviewed_at,
        "reviewer": reviewer.strip(),
        "review_notes": notes.strip(),
        "fall_onset_sec": float(onset_sec) if onset_sec is not None else None,
        "impact_sec": float(impact_sec) if impact_sec is not None else None,
        "fall_end_sec": float(end_sec) if end_sec is not None else None,
        # Review does not bypass cadence/track curation and split assignment.
        "training_eligible": False,
        "training_blockers": ["curation_pending", "split_assignment_pending"],
        "review_history": history,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--onset-sec", type=float)
    parser.add_argument("--impact-sec", type=float)
    parser.add_argument("--end-sec", type=float)
    args = parser.parse_args()

    manifest_path = args.session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated = reviewed_manifest(
        manifest,
        label=args.label,
        reviewer=args.reviewer,
        notes=args.notes,
        onset_sec=args.onset_sec,
        impact_sec=args.impact_sec,
        end_sec=args.end_sec,
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    print(json.dumps({
        "session_id": updated.get("session_id"),
        "label": updated["label"],
        "binary_fall_label": updated["binary_fall_label"],
        "review_status": updated["review_status"],
        "training_eligible": updated["training_eligible"],
        "training_blockers": updated["training_blockers"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
