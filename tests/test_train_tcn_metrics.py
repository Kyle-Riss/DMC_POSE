import numpy as np

from train_tcn import report_metrics


def test_single_class_split_has_no_roc_auc_claim():
    report = report_metrics(np.ones(4, dtype=np.int64), np.array([0.6, 0.7, 0.8, 0.9]), 0.5)
    assert report["roc_auc"] is None
    assert report["recall"] == 1.0


def test_two_class_split_reports_roc_auc():
    report = report_metrics(np.array([0, 1]), np.array([0.1, 0.9]), 0.5)
    assert report["roc_auc"] == 1.0
