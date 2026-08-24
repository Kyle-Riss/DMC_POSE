import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.prepare_realbiomfall_manifest import (
    RestrictedRealBiomUnpickler,
    annotation_time,
    build_manifest,
    parse_video_name,
)


def test_annotation_time_accepts_scalar_and_single_bbox_entry():
    assert annotation_time(1.25) == 1.25
    assert annotation_time([{"t": 2.5, "bbox": {"left": 1}}]) == 2.5
    assert annotation_time(False) is None
    assert annotation_time([]) is None
    assert annotation_time([{"t": 1.0}, {"t": 2.0}]) is None


def test_parse_video_name_preserves_underscored_source_id():
    source, start, end = parse_video_name("_UfI_SNDkqY_63.5_65.9.mp4")
    assert source == "_UfI_SNDkqY"
    assert start == 63.5
    assert end == 65.9


def test_restricted_unpickler_rejects_unexpected_global(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(Path("unsafe")))
    with path.open("rb") as handle, pytest.raises(pickle.UnpicklingError):
        RestrictedRealBiomUnpickler(handle).load()


def test_manifest_is_train_augmentation_only_and_reports_official_leakage(tmp_path):
    labels = tmp_path / "labels"
    videos = tmp_path / "videos"
    labels.mkdir()
    videos.mkdir()
    filenames = ["source_A_0.0_5.0.mp4", "source_A_6.0_11.0.mp4"]
    semantic = {
        filenames[0]: {"start": 2.0, "reaching lowest position": 3.0, "end of fall": False},
        filenames[1]: {"start": [{"t": 4.0, "bbox": {}}], "reaching lowest position": None, "end of fall": None},
    }
    coarse = {
        filenames[0]: {"subset": "training"},
        filenames[1]: {"subset": "testing"},
    }
    fine = {name: {} for name in filenames}
    for name, data in (
        ("labels_semantical.pkl", semantic),
        ("labels_temporal_coarse.pkl", coarse),
        ("labels_temporal_finegrained.pkl", fine),
    ):
        (labels / name).write_bytes(pickle.dumps(data))
    for filename in filenames:
        (videos / filename).touch()

    meta = {"readable": True, "fps": 30.0, "frame_count": 150, "duration_sec": 5.0, "width": 640, "height": 360}
    with patch("scripts.prepare_realbiomfall_manifest.video_metadata", return_value=meta):
        manifest = build_manifest(labels, videos)

    assert manifest["video_count"] == 2
    assert manifest["training_eligible_count"] == 2
    assert manifest["four_second_precontext_count"] == 1
    assert manifest["official_split_leaking_source_groups"] == ["source_A"]
    assert all(item["split"] == "train" for item in manifest["items"])
    assert all(item["augmentation_train_only"] for item in manifest["items"])
    assert all(not item["promotion_metric_eligible"] for item in manifest["items"])
    assert manifest["items"][0]["intervals"][0]["label"] == "non_fall"
    assert manifest["items"][0]["intervals"][1]["label"] == "fall"
