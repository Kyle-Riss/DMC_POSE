"""Offline adapter for the shipped FallVision bed LSTM artifact.

The repository's temporal/FPS provenance is inconsistent. This adapter only
implements the verifiable spatial transform, scaler, and tensor/model I/O. A
caller must not claim a valid zero-shot temporal comparison until the 60-row
time contract is resolved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from external_temporal_contracts import FALLVISION_FALL, ExternalContractError, validate_feature_window


FALLVISION_MIN_CONFIDENCE = 0.2
FALLVISION_FALL_JOINTS = tuple(range(5, 17))


def fallvision_source_features(keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> np.ndarray:
    """Reproduce the checked-in source's confidence filter and hip translation."""
    xy = np.asarray(keypoints_xy)
    confidence = np.asarray(keypoints_conf)
    if xy.ndim < 3 or xy.shape[-2:] != (17, 2):
        raise ExternalContractError(f"FallVision: expected (..., 17, 2), got {xy.shape}")
    if confidence.shape != xy.shape[:-1]:
        raise ExternalContractError(f"FallVision: expected confidence {xy.shape[:-1]}, got {confidence.shape}")
    if not np.issubdtype(xy.dtype, np.floating) or not np.issubdtype(confidence.dtype, np.floating):
        raise ExternalContractError("FallVision: floating XY and confidence required")
    finite = np.isfinite(xy).all(axis=-1) & np.isfinite(confidence)
    present = finite & (confidence > FALLVISION_MIN_CONFIDENCE)
    source_xy = np.where(present[..., None], xy, 0.0)
    midhip = (source_xy[..., 11, :] + source_xy[..., 12, :]) * 0.5
    selected = source_xy[..., FALLVISION_FALL_JOINTS, :] - midhip[..., None, :]
    return np.ascontiguousarray(selected.reshape(*xy.shape[:-2], 24), dtype=np.float32)


def load_scaler_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "dmc_safe_standard_scaler_v1":
        raise ExternalContractError("FallVision: unsupported safe scaler format")
    mean = np.asarray(payload.get("mean"), dtype=np.float32)
    scale = np.asarray(payload.get("scale"), dtype=np.float32)
    if mean.shape != (24,) or scale.shape != (24,):
        raise ExternalContractError("FallVision: scaler must contain 24 mean/scale values")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ExternalContractError("FallVision: invalid scaler values")
    return mean, scale


def prepare_fallvision_bed_window(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    *,
    scaler_json: Path,
) -> np.ndarray:
    features = fallvision_source_features(keypoints_xy, keypoints_conf)
    features = validate_feature_window(features, FALLVISION_FALL)
    mean, scale = load_scaler_json(scaler_json)
    scaled = (features - mean[None, :]) / scale[None, :]
    if not np.isfinite(scaled).all():
        raise ExternalContractError("FallVision: scaled input contains NaN/Inf")
    return np.ascontiguousarray(scaled[None, ...], dtype=np.float32)


class FallVisionBedAdapter:
    """Load and execute the HDF5 model after DMC-owned preprocessing."""

    def __init__(self, model_path: Path, scaler_json: Path):
        import tensorflow as tf

        self.model_path = Path(model_path)
        self.scaler_json = Path(scaler_json)
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        input_shape = tuple(self.model.input_shape)
        if input_shape[-2:] != (60, 24):
            raise ExternalContractError(f"FallVision: unexpected model input {input_shape}")

    def predict(self, keypoints_xy: np.ndarray, keypoints_conf: np.ndarray) -> float:
        tensor = prepare_fallvision_bed_window(
            keypoints_xy, keypoints_conf, scaler_json=self.scaler_json
        )
        output = np.asarray(self.model.predict(tensor, verbose=0)).reshape(-1)
        if output.shape != (1,) or not np.isfinite(output[0]):
            raise ExternalContractError(f"FallVision: unexpected output {output}")
        return float(output[0])
