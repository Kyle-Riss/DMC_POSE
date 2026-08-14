from pose_candidate_filter import accept_pose_candidate, select_tracking_bbox


BASE = {
    "strong_box_conf": 0.5,
    "strong_min_area_ratio": 0.016,
    "weak_box_conf": 0.5,
    "weak_min_area_ratio": 0.025,
    "weak_min_visible": 8,
    "weak_min_keypoint_mean": 0.25,
}


def test_accepts_calibrated_low_pose_person():
    assert accept_pose_candidate(
        box_confidence=0.882,
        area_ratio=0.0199,
        visible_count=10,
        keypoint_mean=0.5,
        **BASE,
    )


def test_rejects_largest_empty_room_false_candidate():
    assert not accept_pose_candidate(
        box_confidence=0.738,
        area_ratio=0.0132,
        visible_count=12,
        keypoint_mean=0.5,
        **BASE,
    )


def test_weak_candidate_keeps_strict_area_and_structure_gate():
    weak_only = {**BASE, "strong_box_conf": 0.8}
    assert not accept_pose_candidate(
        box_confidence=0.5,
        area_ratio=0.0249,
        visible_count=17,
        keypoint_mean=0.9,
        **weak_only,
    )
    assert accept_pose_candidate(
        box_confidence=0.5,
        area_ratio=0.025,
        visible_count=8,
        keypoint_mean=0.25,
        **weak_only,
    )


def test_tracking_prefers_detector_box_over_pose_dependent_keypoint_box():
    detector = (10.0, 20.0, 110.0, 220.0)
    collapsed_keypoints = (70.0, 140.0, 105.0, 210.0)
    assert select_tracking_bbox(detector, collapsed_keypoints) == detector


def test_tracking_falls_back_when_detector_box_is_invalid():
    keypoints = (20.0, 30.0, 60.0, 90.0)
    assert select_tracking_bbox((10.0, 10.0, 5.0, 5.0), keypoints) == keypoints
    assert select_tracking_bbox(None, keypoints) == keypoints
