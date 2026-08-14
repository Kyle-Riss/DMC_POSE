import pytest

from scripts.review_temporal_session import reviewed_manifest


def base_manifest():
    return {
        "session_id": "bed_161_test",
        "label": "UNREVIEWED",
        "binary_fall_label": None,
        "duration_sec": 12.0,
        "training_eligible": False,
        "training_blockers": ["label_unreviewed"],
    }


def test_negative_review_stays_blocked_until_curation():
    result = reviewed_manifest(
        base_manifest(),
        label="NORMAL_ENTRY_PRESENCE_EXIT",
        reviewer="operator",
        notes="controlled protocol",
    )
    assert result["binary_fall_label"] == 0
    assert result["review_status"] == "reviewed"
    assert result["training_eligible"] is False
    assert result["training_blockers"] == [
        "curation_pending", "split_assignment_pending"
    ]


def test_fall_requires_ordered_boundaries():
    with pytest.raises(ValueError):
        reviewed_manifest(base_manifest(), label="FALL", reviewer="operator")
    with pytest.raises(ValueError):
        reviewed_manifest(
            base_manifest(), label="FALL", reviewer="operator",
            onset_sec=5, impact_sec=4, end_sec=6,
        )
    result = reviewed_manifest(
        base_manifest(), label="FALL", reviewer="operator",
        onset_sec=4, impact_sec=5, end_sec=7,
    )
    assert result["binary_fall_label"] == 1
    assert result["fall_onset_sec"] == 4


def test_negative_rejects_fall_boundaries():
    with pytest.raises(ValueError):
        reviewed_manifest(
            base_manifest(), label="NORMAL_SIT", reviewer="operator",
            onset_sec=1,
        )
