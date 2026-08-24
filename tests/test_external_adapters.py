import json

import numpy as np
import pytest

from external_adapters.fallvision_adapter import fallvision_source_features, prepare_fallvision_bed_window
from external_adapters.wardy_adapter import prepare_wardy_window
from external_temporal_contracts import ExternalContractError
from scripts.export_sklearn_scaler_safe import extract_arrays


def test_fallvision_transform_reproduces_missing_joint_source_behavior(tmp_path):
    xy = np.zeros((60, 17, 2), dtype=np.float32)
    conf = np.zeros((60, 17), dtype=np.float32)
    xy[:, 11] = [10.0, 20.0]
    xy[:, 12] = [14.0, 24.0]
    xy[:, 5] = [15.0, 30.0]
    conf[:, [5, 11, 12]] = 0.9
    features = fallvision_source_features(xy, conf)
    assert features.shape == (60, 24)
    assert features[0, :2].tolist() == pytest.approx([3.0, 8.0])
    # Missing elbow is zero-filled before hip translation in upstream source.
    assert features[0, 4:6].tolist() == pytest.approx([-12.0, -22.0])

    scaler = tmp_path / "scaler.json"
    scaler.write_text(json.dumps({"format": "dmc_safe_standard_scaler_v1", "mean": [0.0] * 24, "scale": [2.0] * 24}), encoding="utf-8")
    tensor = prepare_fallvision_bed_window(xy, conf, scaler_json=scaler)
    assert tensor.shape == (1, 60, 24)
    assert tensor[0, 0, :2].tolist() == pytest.approx([1.5, 4.0])


def test_fallvision_wrong_rows_fail_closed(tmp_path):
    scaler = tmp_path / "scaler.json"
    scaler.write_text(json.dumps({"format": "dmc_safe_standard_scaler_v1", "mean": [0.0] * 24, "scale": [1.0] * 24}), encoding="utf-8")
    with pytest.raises(ExternalContractError):
        prepare_fallvision_bed_window(np.zeros((30, 17, 2), np.float32), np.ones((30, 17), np.float32), scaler_json=scaler)


def test_wardy_metadata_normalization(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "input_name": "features", "output_name": "logits", "input_shape": ["batch", 20, 80],
        "target_fps": 10.0, "window_seconds": 2.0, "feature_names": [f"f{i}" for i in range(80)],
        "feature_mean": [1.0] * 80, "feature_std": [2.0] * 80,
    }), encoding="utf-8")
    tensor = prepare_wardy_window(np.full((20, 80), 3.0, np.float32), np.arange(20) / 10.0, metadata_path=metadata)
    assert tensor.shape == (1, 20, 80)
    assert np.all(tensor == 1.0)


def test_real_bed_scaler_is_extracted_without_unpickling():
    path = __import__('pathlib').Path('external_models/posture-aware-fall-detection/models/bed_scaler.pkl')
    if not path.is_file():
        pytest.skip('local external model not present')
    arrays = extract_arrays(path, 24)
    assert arrays['mean'].shape == (24,)
    assert arrays['scale'].shape == (24,)
