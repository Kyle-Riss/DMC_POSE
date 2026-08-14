from __future__ import annotations

REQUIRED_TEMPORAL_FIELDS = (
    "raw_video_exists",
    "decode_ok",
    "temporal_annotation_complete",
    "split_group_resolved",
    "observed_pose_samples_available",
    "pre_context_sufficient",
)


def evaluate_temporal_eligibility(record: dict) -> dict:
    checks = {field: bool(record.get(field, False)) for field in REQUIRED_TEMPORAL_FIELDS}
    missing = [field for field, ready in checks.items() if not ready]
    return {
        "temporal_tcn_eligible": not missing,
        "eligibility_checks": checks,
        "temporal_tcn_blockers": missing,
        "pre_onset_observed_samples": record.get("pre_onset_observed_samples"),
        "pre_onset_ready": bool(record.get("pre_onset_ready", False)),
        "event_evaluable": bool(record.get("event_evaluable", False)),
    }
