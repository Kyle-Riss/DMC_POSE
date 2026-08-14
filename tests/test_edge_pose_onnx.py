import numpy as np

from edge_pose_onnx import decode_pose, nms


def test_nms_keeps_highest_overlapping_box():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [30, 30, 40, 40]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5) == [0, 2]


def test_decode_pose_maps_letterbox_coordinates_back():
    output = np.zeros((1, 56, 2), dtype=np.float32)
    output[0, :5, 0] = [160, 160, 100, 100, 0.9]
    for keypoint in range(17):
        start = 5 + keypoint * 3
        output[0, start:start + 3, 0] = [160, 160, 0.8]
    detections = decode_pose(
        output, original_shape=(180, 320), scale=1.0, pad=(0, 70), confidence=0.25
    )
    assert len(detections) == 1
    assert detections[0]["visible_keypoints"] == 17
    assert detections[0]["keypoints_xyc"][0][:2] == [160.0, 90.0]


def test_decode_pose_filters_low_confidence():
    output = np.zeros((1, 56, 1), dtype=np.float32)
    output[0, 4, 0] = 0.1
    assert decode_pose(
        output, original_shape=(180, 320), scale=1.0, pad=(0, 70)
    ) == []

