import json

import pandas as pd

from scripts.summarize_temporal_extraction import summarize


def test_summary_is_quality_only_and_camera_scoped(tmp_path):
    output = tmp_path / "diagnostic" / "video.csv"
    output.parent.mkdir()
    frame = {f"kpt_{index}_visible": [1, 1] for index in range(17)}
    pd.DataFrame(frame).to_csv(output, index=False)
    index = {
        "feature_schema_version": "pose_temporal_109_v1",
        "sequence_contract_version": "observed_only_20hz_v1",
        "sample_hz": 20.0,
        "elapsed_sec": 5.0,
        "results": [{
            "video_id": "video",
            "status": "ok",
            "out": str(output),
            "decoded_frames": 4,
            "pose_probes": 4,
            "rows": 2,
            "no_primary": 2,
            "gap_reset": 1,
        }],
    }
    (tmp_path / "features_index.json").write_text(json.dumps(index), encoding="utf-8")
    manifest = {"items": [{"video_id": "video", "camera_id": "c1", "duration_sec": 10.0}]}
    report = summarize(tmp_path, manifest)
    assert report["accuracy_claim"] is False
    assert report["offline_realtime_factor"] == 2.0
    assert report["totals"]["pose_observation_coverage"] == 0.5
    assert report["by_camera"]["c1"]["no_primary_rate"] == 0.5
    assert report["mean_visible_joints_per_observation"] == 17.0


def test_summary_counts_empty_observation_csv(tmp_path):
    output = tmp_path / "train" / "empty.csv"
    output.parent.mkdir()
    output.touch()
    index = {
        "sample_hz": 20.0,
        "elapsed_sec": 1.0,
        "results": [{
            "video_id": "empty",
            "status": "ok",
            "out": str(output),
            "pose_probes": 20,
            "rows": 0,
            "no_primary": 20,
        }],
    }
    (tmp_path / "features_index.json").write_text(json.dumps(index), encoding="utf-8")
    manifest = {"items": [{"video_id": "empty", "duration_sec": 1.0}]}
    report = summarize(tmp_path, manifest)
    assert report["csv_count"] == 1
    assert report["empty_csv_count"] == 1
    assert report["totals"]["pose_observation_coverage"] == 0.0
