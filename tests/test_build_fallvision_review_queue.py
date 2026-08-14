from scripts.build_fallvision_review_queue import build_queue, possible_event_windows


def test_possible_event_windows_uses_29th_sample_as_first_end():
    assert possible_event_windows(0, 89, 30.0) == 1
    assert possible_event_windows(0, 119, 30.0) == 3
    assert possible_event_windows(100, 119, 30.0) == 2


def test_queue_accepts_only_unreviewed_review_required_with_capacity():
    base = {
        "scene_id": "bed", "chunk_id": "1", "duration_sec": "4",
        "fps": "30", "annotation_status": "unreviewed",
    }
    annotations = [
        {**base, "video_id": "ok"},
        {**base, "video_id": "done", "annotation_status": "complete"},
        {**base, "video_id": "short"},
    ]
    proposals = [
        {"video_id": "ok", "proposal_status": "review_required", "proposed_fall_onset_frame": "0", "proposed_fall_end_frame": "119"},
        {"video_id": "done", "proposal_status": "review_required", "proposed_fall_onset_frame": "0", "proposed_fall_end_frame": "119"},
        {"video_id": "short", "proposal_status": "review_required", "proposed_fall_onset_frame": "0", "proposed_fall_end_frame": "89"},
    ]
    queue, queue_proposals, audit = build_queue(annotations, proposals, min_persistence=2)
    assert [row["video_id"] for row in queue] == ["ok"]
    assert [row["video_id"] for row in queue_proposals] == ["ok"]
    assert audit[0]["persistence_capacity_upper_bound"] == 3
