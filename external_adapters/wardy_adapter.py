"""Offline ONNX adapter for Wardy M-04 when exact raw 80D features exist."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from external_temporal_contracts import WARDY_M04, ExternalContractError, validate_feature_window, validate_observed_cadence


def load_wardy_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("input_name") != "features" or payload.get("output_name") != "logits":
        raise ExternalContractError("Wardy: unsupported model I/O names")
    if payload.get("input_shape") != ["batch", 20, 80]:
        raise ExternalContractError(f"Wardy: unexpected metadata input shape {payload.get('input_shape')}")
    if float(payload.get("target_fps")) != 10.0 or float(payload.get("window_seconds")) != 2.0:
        raise ExternalContractError("Wardy: unexpected cadence metadata")
    if len(payload.get("feature_names", [])) != 80:
        raise ExternalContractError("Wardy: expected 80 feature names")
    return payload


def prepare_wardy_window(raw_features: np.ndarray, timestamps: np.ndarray, *, metadata_path: Path) -> np.ndarray:
    features = validate_feature_window(raw_features, WARDY_M04)
    validate_observed_cadence(timestamps, WARDY_M04)
    metadata = load_wardy_metadata(metadata_path)
    mean = np.asarray(metadata.get("feature_mean"), dtype=np.float32)
    std = np.asarray(metadata.get("feature_std"), dtype=np.float32)
    if mean.shape != (80,) or std.shape != (80,) or np.any(std <= 0.0):
        raise ExternalContractError("Wardy: invalid 80D normalization metadata")
    normalized = (features - mean[None, :]) / std[None, :]
    if not np.isfinite(normalized).all():
        raise ExternalContractError("Wardy: normalized input contains NaN/Inf")
    return np.ascontiguousarray(normalized[None, ...], dtype=np.float32)


class WardyAdapter:
    """Execute Wardy only on exact externally built 80D observations."""

    def __init__(self, model_path: Path, metadata_path: Path):
        import onnxruntime as ort

        self.metadata_path = Path(metadata_path)
        load_wardy_metadata(self.metadata_path)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "features" or inputs[0].shape[1:] != [20, 80]:
            raise ExternalContractError("Wardy: ONNX input contract mismatch")
        if len(outputs) != 1 or outputs[0].name != "logits":
            raise ExternalContractError("Wardy: ONNX output contract mismatch")

    def predict(self, raw_features: np.ndarray, timestamps: np.ndarray) -> float:
        tensor = prepare_wardy_window(raw_features, timestamps, metadata_path=self.metadata_path)
        logits = np.asarray(self.session.run(["logits"], {"features": tensor})[0]).reshape(-1)
        if logits.shape != (1,) or not np.isfinite(logits[0]):
            raise ExternalContractError(f"Wardy: unexpected output {logits}")
        return 1.0 / (1.0 + math.exp(-float(logits[0])))
