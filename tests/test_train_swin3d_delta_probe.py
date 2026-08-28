import numpy as np

from scripts.train_swin3d_delta_probe import paired_embeddings


def row(video, source, label, start, flip=False):
    return {
        "video_id": video, "label_source": source, "label": label,
        "start_frame": start, "horizontal_flip": flip,
    }


def test_builds_positive_delta_and_ordered_hard_negative_deltas():
    x = np.asarray([[1, 1], [3, 4], [5, 7], [0, 0], [1, 2], [4, 6]], dtype=np.float32)
    metadata = [
        row("fall", "same_recording_prefall", 0, 0),
        row("fall", "reviewed_fall_transition", 1, 10),
        row("fall", "reviewed_fall_transition", 1, 20),
        row("normal", "reviewed_hard_negative", 0, 0),
        row("normal", "reviewed_hard_negative", 0, 10),
        row("normal", "reviewed_hard_negative", 0, 20),
    ]
    features, labels = paired_embeddings(x, metadata)
    assert labels.tolist() == [1, 1, 0, 0, 0]
    np.testing.assert_array_equal(features[0], [2, 3])
    np.testing.assert_array_equal(features[-1], [4, 6])
