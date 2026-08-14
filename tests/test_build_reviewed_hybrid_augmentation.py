import json
from argparse import Namespace

import numpy as np
import pytest

from scripts.build_reviewed_hybrid_augmentation import build, load_local_negatives


def write_fallvision(tmp_path):
    root = tmp_path / "fv"
    root.mkdir()
    index = {
        "feature_schema_version": "pose_temporal_109_v1",
        "sequence_contract_version": "observed_only_10hz_v2",
        "window_rows": 30,
        "feature_count": 109,
        "splits": {},
    }
    (root / "window_index.json").write_text(json.dumps(index))
    x = np.zeros((3, 30, 109), dtype=np.float32)
    y = np.asarray([1, 1, 0], dtype=np.int64)
    np.savez_compressed(root / "test.npz", x=x, y=y)
    metadata = [
        {"video_id": "fall_train", "split": "test", "track_id": 1},
        {"video_id": "fall_hold", "split": "test", "track_id": 1},
        {"video_id": "nonfall", "split": "test", "track_id": 1},
    ]
    (root / "test_metadata.json").write_text(json.dumps(metadata))
    items = [
        {"video_id": "fall_train", "split_group": "fall:g1", "annotation_scope": "manual_interval_diagnostic_only", "annotation_source": "manual_pilot", "split": "test", "duration_sec": 3, "intervals": [{"label": "fall", "start_sec": 0, "end_sec": 2.9}]},
        {"video_id": "fall_hold", "split_group": "fall:g2", "annotation_scope": "manual_interval_diagnostic_only", "annotation_source": "manual_pilot", "split": "test", "duration_sec": 3, "intervals": [{"label": "fall", "start_sec": 0, "end_sec": 2.9}]},
        {"video_id": "nonfall", "split_group": "nonfall:g3", "annotation_scope": "video_non_fall_diagnostic_only", "annotation_source": "official", "split": "test", "duration_sec": 3, "intervals": []},
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": items}))
    return root, manifest


def write_local(tmp_path, labels=(0,)):
    path = tmp_path / "local.npz"
    n = len(labels)
    np.savez_compressed(
        path,
        x=np.zeros((n, 30, 109), dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        session_ids=np.asarray([f"s{i}" for i in range(n)]),
        track_ids=np.ones(n, dtype=np.int64),
        segment_indices=np.zeros(n, dtype=np.int64),
        start_sec=np.zeros(n),
        end_sec=np.full(n, 2.9),
        mean_quality=np.ones(n, dtype=np.float32),
    )
    report = tmp_path / "local_report.json"
    report.write_text(json.dumps({"schema_version": "curated_temporal_sessions_v1"}))
    return path, report


def test_build_keeps_archive_groups_disjoint_and_outputs_train_only(tmp_path):
    fv, manifest = write_fallvision(tmp_path)
    local, local_report = write_local(tmp_path)
    out = tmp_path / "out"
    report = build(Namespace(
        fallvision_windows=fv,
        fallvision_manifest=manifest,
        local_npz=local,
        local_report=local_report,
        out=out,
        holdout_fall_group=["fall:g2"],
    ))
    train = np.load(out / "augmentation_train_only/train.npz")
    diagnostic = np.load(out / "fallvision_archive_holdout_diagnostic/test.npz")
    assert train["x"].shape == (2, 30, 109)
    assert train["y"].tolist() == [1, 0]
    assert diagnostic["y"].tolist() == [1, 0]
    assert report["archive_group_overlap"] == []
    assert report["promotion_eligible"] is False
    assert np.load(out / "augmentation_train_only/val.npz")["x"].shape == (0, 30, 109)


def test_local_positive_is_rejected(tmp_path):
    local, report = write_local(tmp_path, labels=(1,))
    with pytest.raises(ValueError, match="non-fall only"):
        load_local_negatives(local, report)
