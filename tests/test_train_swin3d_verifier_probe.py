import numpy as np

from scripts.train_swin3d_verifier_probe import metrics, operating_point


def test_operating_point_honors_minimum_recall():
    labels = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    point = operating_point(labels, probability, 1.0)
    assert point["recall"] == 1.0
    assert point["threshold"] == 0.6


def test_metrics_confusion_counts():
    result = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]), 0.5)
    assert result["confusion"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
