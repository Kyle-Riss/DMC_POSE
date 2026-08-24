from scripts.benchmark_pose_batch import percentile


def test_percentile_is_stable_and_rounded():
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
