#!/usr/bin/env python3
"""Materialize reviewed live sessions from privacy-preserving shadow logs.

The shadow log intentionally does not contain images, keypoints, or the 109-D
temporal feature vector.  Curated outputs are therefore suitable for runtime
and fusion calibration, but never for temporal-model training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from typing import Iterable


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def candidate_log_paths(shadow_dir: Path, start: datetime, end: datetime) -> Iterable[Path]:
    current = start.date()
    while current <= end.date():
        yield shadow_dir / f"shadow_features_{current:%Y%m%d}.jsonl"
        current = current.fromordinal(current.toordinal() + 1)


def row_time(row: dict) -> datetime | None:
    value = row.get("recorded_at") or row.get("timestamp")
    if not value:
        return None
    try:
        return parse_utc(str(value))
    except ValueError:
        return None


def select_rows(session: dict, shadow_dir: Path) -> tuple[list[dict], list[str]]:
    start = parse_utc(str(session["start_utc"]))
    end = parse_utc(str(session["end_utc"]))
    if end < start:
        raise ValueError(f"session {session.get('session_id')} ends before it starts")
    camera_id = str(session["camera_id"])
    selected = []
    sources = []
    for path in candidate_log_paths(shadow_dir, start, end):
        if not path.exists():
            continue
        sources.append(str(path.resolve()))
        for row in read_jsonl(path):
            timestamp = row_time(row)
            if row.get("camera_id") == camera_id and timestamp is not None and start <= timestamp <= end:
                selected.append(row)
    selected.sort(key=lambda row: row_time(row) or start)
    return selected, sources


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def finite_numbers(rows: list[dict], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize(session: dict, rows: list[dict], sources: list[str]) -> dict:
    times = [row_time(row) for row in rows]
    valid_times = [value for value in times if value is not None]
    intervals_ms = [
        (right - left).total_seconds() * 1000.0
        for left, right in zip(valid_times, valid_times[1:])
        if right >= left
    ]
    probabilities = finite_numbers(rows, "tcn_fall_probability")
    queue_latency = finite_numbers(rows, "scheduler_queue_latency_ms")
    observed_track_ids = [
        row.get("primary_track_id")
        for row in rows
        if row.get("primary_track_observed") and row.get("primary_track_id") is not None
    ]
    unique_track_ids = sorted(set(observed_track_ids), key=str)
    candidate_count = sum(bool(row.get("tcn_alert_candidate")) for row in rows)
    fusion_alert_count = sum(
        str(row.get("fusion_phase") or "").upper() in {"ALERT", "CONFIRMED_FALL"}
        for row in rows
    )
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return {
        "schema_version": "curated_shadow_session_v1",
        "session_id": session["session_id"],
        "camera_id": session["camera_id"],
        "label": session.get("event_label"),
        "binary_fall_label": session.get("binary_fall_label"),
        "start_utc": session["start_utc"],
        "end_utc": session["end_utc"],
        "actions": list(session.get("actions") or []),
        "annotation_precision": session.get("annotation_precision"),
        "source_logs": sources,
        "matched_rows": len(rows),
        "first_row_utc": valid_times[0].isoformat().replace("+00:00", "Z") if valid_times else None,
        "last_row_utc": valid_times[-1].isoformat().replace("+00:00", "Z") if valid_times else None,
        "row_interval_ms": {
            "median": statistics.median(intervals_ms) if intervals_ms else None,
            "p95": percentile(intervals_ms, 0.95),
            "max": max(intervals_ms) if intervals_ms else None,
        },
        "tracking": {
            "unique_observed_track_ids": unique_track_ids,
            "continuous_single_track": len(unique_track_ids) == 1 and bool(observed_track_ids),
            "unobserved_rows": sum(not bool(row.get("primary_track_observed")) for row in rows),
        },
        "tcn": {
            "ready_rows": sum(bool(row.get("tcn_shadow_ready")) for row in rows),
            "candidate_rows": candidate_count,
            "max_probability": max(probabilities) if probabilities else None,
            "sources": dict(Counter(str(row.get("tcn_source") or "unknown") for row in rows)),
        },
        "fusion": {
            "alert_rows": fusion_alert_count,
            "suppressed_all_tcn_candidates": candidate_count > 0 and fusion_alert_count == 0,
            "phases": dict(Counter(str(row.get("fusion_phase") or "unknown") for row in rows)),
        },
        "runtime": {
            "max_scheduler_queue_latency_ms": max(queue_latency) if queue_latency else None,
            "capture_disconnect_rows": sum(row.get("capture_connected") is False for row in rows),
            "decode_error_delta": (
                max(finite_numbers(rows, "capture_decode_error_total"))
                - min(finite_numbers(rows, "capture_decode_error_total"))
                if finite_numbers(rows, "capture_decode_error_total") else None
            ),
        },
        "integrity": {"canonical_rows_sha256": hashlib.sha256(encoded).hexdigest()},
        "usage_contract": {
            "fusion_calibration_eligible": bool(rows),
            "temporal_training_eligible": False,
            "temporal_training_blockers": [
                "shadow_rows_do_not_contain_109d_feature_vector",
                "session_level_annotation_has_no_per_action_boundaries",
            ],
            "contains_images": False,
            "contains_keypoints": False,
        },
    }


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=project / "runtime_data/annotations/hard_negative_sessions.jsonl")
    parser.add_argument("--shadow-dir", type=Path, default=project / "runtime_data/shadow_features")
    parser.add_argument("--output-dir", type=Path, default=project / "runtime_data/curated_sessions")
    args = parser.parse_args()

    sessions = read_jsonl(args.sessions)
    results = []
    for session in sessions:
        rows, sources = select_rows(session, args.shadow_dir)
        report = summarize(session, rows, sources)
        session_dir = args.output_dir / str(session["session_id"])
        write_jsonl_atomic(session_dir / "shadow_rows.jsonl", rows)
        write_json_atomic(session_dir / "report.json", report)
        results.append(report)
    index = {
        "schema_version": "curated_shadow_session_index_v1",
        "session_count": len(results),
        "sessions": results,
    }
    write_json_atomic(args.output_dir / "index.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0 if all(item["matched_rows"] > 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
