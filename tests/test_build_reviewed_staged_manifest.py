import pytest

from scripts.build_reviewed_staged_manifest import build_manifest, identity_template


def annotation(camera="c1", recording="sim0101", status="complete"):
    return {
        "video_id": f"{camera}_{recording}", "dataset": "usb", "scene_id": "bed",
        "camera_id": camera, "recording_id": recording,
        "local_video_path": f"/data/{camera}_{recording}.mp4", "fps": "20",
        "frame_count": "240", "duration_sec": "12", "width": "640", "height": "360",
        "fall_onset_frame": "100", "impact_frame": "110",
        "post_fall_stable_frame": "120", "fall_end_frame": "180",
        "annotation_status": status, "annotation_confidence": "high", "annotator": "reviewer",
    }


def test_reviewed_multiview_builds_one_locked_group():
    rows = [annotation(camera) for camera in ("c1", "c2", "c3")]
    identity = {"recordings": {"sim0101": {
        "subject_id": "actor_1", "session_id": "session_1", "split": "train"
    }}}
    result = build_manifest(rows, identity)
    assert result["video_count"] == 3
    assert result["recording_count"] == 1
    assert result["subject_count"] == 1
    assert result["split_counts"] == {"train": 3}
    assert all(item["fall_start_sec"] == 5.0 for item in result["items"])
    assert len({item["multiview_group"] for item in result["items"]}) == 1


def test_unreviewed_annotation_fails_closed():
    with pytest.raises(ValueError, match="unresolved"):
        build_manifest([annotation(status="unreviewed")], {"recordings": {}})


def test_excluded_and_adjudication_rows_are_held_out():
    rows = [
        annotation(recording="sim0101"),
        annotation(recording="sim0102", status="excluded"),
        annotation(recording="sim0103", status="needs_adjudication"),
    ]
    identity = {"recordings": {"sim0101": {
        "subject_id": "actor_1", "session_id": "session_1", "split": "train"
    }}}
    result = build_manifest(rows, identity)
    assert result["video_count"] == 1
    assert result["review_summary"]["recording_status_counts"] == {
        "complete": 1, "excluded": 1, "needs_adjudication": 1,
    }


def test_multiview_status_disagreement_fails_closed():
    rows = [annotation("c1"), annotation("c2", status="excluded")]
    with pytest.raises(ValueError, match="status disagreement"):
        build_manifest(rows, {"recordings": {}})


def test_unknown_subject_mapping_fails_closed():
    with pytest.raises(ValueError, match="subject/session"):
        build_manifest([annotation()], {"recordings": {"sim0101": {
            "subject_id": None, "session_id": "s1", "split": "train"
        }}})


def test_same_subject_cannot_cross_splits():
    rows = [annotation(recording="sim0101"), annotation(recording="sim0102")]
    identity = {"recordings": {
        "sim0101": {"subject_id": "actor", "session_id": "s1", "split": "train"},
        "sim0102": {"subject_id": "actor", "session_id": "s2", "split": "test"},
    }}
    with pytest.raises(ValueError, match="subject split leakage"):
        build_manifest(rows, identity)


def test_identity_template_groups_three_views():
    rows = [annotation(camera) for camera in ("c1", "c2", "c3")]
    template = identity_template(rows)
    assert list(template["recordings"]) == ["sim0101"]
    assert template["recordings"]["sim0101"]["camera_views"] == ["c1", "c2", "c3"]
    assert template["recordings"]["sim0101"]["subject_id"] is None


def test_identity_template_omits_noncomplete_recordings():
    rows = [annotation(recording="sim0101"), annotation(recording="sim0102", status="excluded")]
    template = identity_template(rows)
    assert list(template["recordings"]) == ["sim0101"]
