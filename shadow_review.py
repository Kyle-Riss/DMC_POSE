"""Stable review artifacts and operational metrics for shadow alerts."""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable

CANDIDATE_LABELS = {
    "pending",
    "true_fall",
    "false_alarm",
    "staged_fall",
    "uncertain",
}
ACTUAL_EVENT_TYPES = {"actual_fall", "staged_fall"}

REVIEW_FIELDS = (
    "candidate_id", "camera_id", "started_at", "ended_at", "peak_risk",
    "evidence", "track_ids", "policy_versions", "label", "reviewer", "labeled_at", "note",
)
EVENT_FIELDS = (
    "event_id", "camera_id", "occurred_at", "event_type",
    "policy_version", "matched_candidate_id", "reviewer", "note",
)


def candidate_id(candidate: dict[str, Any]) -> str:
    """Return an ID that remains stable while the event end time grows."""
    source = f"{candidate.get('camera_id', '')}|{candidate.get('started_at', '')}"
    return "sha_" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fields: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_review_rows(
    candidates: list[dict[str, Any]],
    existing_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Merge current candidates with an editable review CSV without losing labels."""
    existing = {
        row.get("candidate_id", ""): row
        for row in (existing_rows or [])
        if row.get("candidate_id")
    }
    current_ids: set[str] = set()
    output: list[dict[str, str]] = []
    for item in sorted(
        candidates,
        key=lambda value: (str(value.get("started_at", "")), str(value.get("camera_id", ""))),
    ):
        item_id = candidate_id(item)
        current_ids.add(item_id)
        old = existing.get(item_id, {})
        label = old.get("label", "pending") or "pending"
        if label not in CANDIDATE_LABELS:
            label = "uncertain"
        output.append({
            "candidate_id": item_id,
            "camera_id": str(item.get("camera_id", "")),
            "started_at": str(item.get("started_at", "")),
            "ended_at": str(item.get("ended_at", "")),
            "peak_risk": f"{float(item.get('peak_risk') or 0.0):.6f}",
            "evidence": "|".join(sorted(map(str, item.get("evidence") or []))),
            "track_ids": "|".join(map(str, item.get("track_ids") or [])),
            "policy_versions": "|".join(map(str, item.get("policy_versions") or [])),
            "label": label,
            "reviewer": old.get("reviewer", ""),
            "labeled_at": old.get("labeled_at", ""),
            "note": old.get("note", ""),
        })

    # Preserve orphaned reviewed rows for auditability if source logs are moved.
    for item_id, old in existing.items():
        if item_id not in current_ids and old.get("label", "pending") != "pending":
            output.append({field: old.get(field, "") for field in REVIEW_FIELDS})
    return output


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator > 0 else None


def evaluate_operations(
    summary: dict[str, Any],
    review_rows: list[dict[str, str]],
    actual_events: list[dict[str, str]] | None = None,
    *,
    min_bed_hours: float = 168.0,
    max_false_alarms_per_bed_hour: float = 0.01,
    min_sensitivity: float = 0.90,
    policy_version: str | None = None,
) -> dict[str, Any]:
    """Compute honest alert burden and incident detection gates."""
    all_review_rows = review_rows
    all_events = actual_events or []
    if policy_version:
        review_rows = [
            row for row in all_review_rows
            if policy_version in (row.get("policy_versions") or "").split("|")
        ]
        events = [
            row for row in all_events if row.get("policy_version") == policy_version
        ]
    else:
        events = all_events
    reviews_by_id = {
        row.get("candidate_id", ""): row
        for row in review_rows
        if row.get("candidate_id")
    }
    camera_ids = set((summary.get("cameras") or {}).keys())
    camera_ids.update(row.get("camera_id", "") for row in review_rows)
    camera_ids.update(row.get("camera_id", "") for row in events)
    camera_ids.discard("")

    review_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    event_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in review_rows:
        review_groups[row.get("camera_id", "")].append(row)
    for row in events:
        event_groups[row.get("camera_id", "")].append(row)

    def metrics_for(camera_id: str, rows: list[dict[str, str]],
                    incident_rows: list[dict[str, str]], bed_hours: float) -> dict[str, Any]:
        labels = defaultdict(int)
        invalid_labels = 0
        for row in rows:
            label = row.get("label", "pending") or "pending"
            if label not in CANDIDATE_LABELS:
                invalid_labels += 1
                label = "uncertain"
            labels[label] += 1
        real_labeled = labels["true_fall"] + labels["false_alarm"]
        actual = [row for row in incident_rows if row.get("event_type") == "actual_fall"]
        staged = [row for row in incident_rows if row.get("event_type") == "staged_fall"]

        def matched(items: list[dict[str, str]], expected_label: str) -> int:
            return sum(
                1 for row in items
                if (
                    row.get("matched_candidate_id") in reviews_by_id
                    and reviews_by_id[row["matched_candidate_id"]].get("label") == expected_label
                )
            )

        actual_matched = matched(actual, "true_fall")
        staged_matched = matched(staged, "staged_fall")
        confirmed_false_alarm_rate = _safe_rate(labels["false_alarm"], bed_hours)
        unresolved = labels["pending"] + labels["uncertain"] + invalid_labels
        false_alarm_rate = confirmed_false_alarm_rate if unresolved == 0 else None
        sensitivity = _safe_rate(actual_matched, len(actual))
        staged_sensitivity = _safe_rate(staged_matched, len(staged))

        if bed_hours < min_bed_hours or unresolved:
            false_alarm_gate = "NOT_READY"
        elif false_alarm_rate is not None and false_alarm_rate <= max_false_alarms_per_bed_hour:
            false_alarm_gate = "PASS"
        else:
            false_alarm_gate = "FAIL"

        if not actual:
            detection_gate = "NOT_MEASURED"
        elif sensitivity is not None and sensitivity >= min_sensitivity:
            detection_gate = "PASS"
        else:
            detection_gate = "FAIL"

        return {
            "bed_hours": round(bed_hours, 6),
            "candidate_count": len(rows),
            "labels": dict(sorted(labels.items())),
            "invalid_label_count": invalid_labels,
            "review_completion": _safe_rate(
                len(rows) - labels["pending"] - labels["uncertain"] - invalid_labels,
                len(rows),
            ),
            "alert_precision_real_only": _safe_rate(labels["true_fall"], real_labeled),
            "false_alarms_per_bed_hour": false_alarm_rate,
            "confirmed_false_alarms_per_bed_hour_lower_bound": confirmed_false_alarm_rate,
            "actual_fall_count": len(actual),
            "actual_fall_matched": actual_matched,
            "sensitivity_actual_falls": sensitivity,
            "staged_fall_count": len(staged),
            "staged_fall_matched": staged_matched,
            "sensitivity_staged_falls": staged_sensitivity,
            "false_alarm_gate": false_alarm_gate,
            "detection_gate": detection_gate,
        }

    cameras = {}
    for camera_id in sorted(camera_ids):
        camera_summary = (summary.get("cameras") or {}).get(camera_id, {})
        if policy_version:
            bed_hours = float(
                (camera_summary.get("policy_bed_hours") or {}).get(policy_version) or 0.0
            )
        else:
            bed_hours = float(camera_summary.get("recorded_bed_hours") or 0.0)
        cameras[camera_id] = metrics_for(
            camera_id, review_groups[camera_id], event_groups[camera_id], bed_hours
        )

    if policy_version:
        total_bed_hours = float(
            (summary.get("policy_bed_hours") or {}).get(policy_version) or 0.0
        )
    else:
        total_bed_hours = float(summary.get("total_bed_hours") or 0.0)
    overall = metrics_for("ALL", review_rows, events, total_bed_hours)
    if overall["false_alarm_gate"] == "PASS" and overall["detection_gate"] == "PASS":
        readiness = "PASS"
    elif "FAIL" in (overall["false_alarm_gate"], overall["detection_gate"]):
        readiness = "FAIL"
    else:
        readiness = "NOT_READY"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy_version": policy_version or "all",
        "targets": {
            "scope": "engineering trial targets; not a clinical standard",
            "min_bed_hours": min_bed_hours,
            "max_false_alarms_per_bed_hour": max_false_alarms_per_bed_hour,
            "min_sensitivity": min_sensitivity,
        },
        "overall": overall,
        "cameras": cameras,
        "readiness": readiness,
        "limitations": [
            "Unlogged actual falls cannot be counted as misses.",
            "Staged falls are reported separately from real operational falls.",
            "A shadow candidate is not a false alarm until reviewed.",
        ],
    }

