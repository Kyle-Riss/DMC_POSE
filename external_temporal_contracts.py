"""Fail-closed contracts for third-party temporal model experiments.

This module deliberately contains no model loader and never unpickles external
artifacts.  It is the admission boundary between DMC observations and offline
third-party baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ExternalContractError(ValueError):
    """Raised when external-model input cannot be proven contract-compatible."""


@dataclass(frozen=True)
class ExternalTemporalContract:
    name: str
    rows: int
    features: int
    target_hz: float | None
    window_seconds: float | None
    status: str


FALLVISION_FALL = ExternalTemporalContract(
    name="fallvision_fall_lstm",
    rows=60,
    features=24,
    target_hz=None,
    window_seconds=None,
    status="quarantined",
)

WARDY_M04 = ExternalTemporalContract(
    name="wardy_m04_gru",
    rows=20,
    features=80,
    target_hz=10.0,
    window_seconds=2.0,
    status="offline_baseline_only",
)

FALLVISION_JOINT_INDICES = tuple(range(5, 17))


def validate_feature_window(window: np.ndarray, contract: ExternalTemporalContract) -> np.ndarray:
    """Return a float32 window only when its exact external contract is met."""
    value = np.asarray(window)
    expected = (contract.rows, contract.features)
    if value.shape != expected:
        raise ExternalContractError(f"{contract.name}: expected {expected}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ExternalContractError(f"{contract.name}: floating dtype required, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ExternalContractError(f"{contract.name}: NaN/Inf input is forbidden")
    return np.ascontiguousarray(value, dtype=np.float32)


def validate_observed_cadence(
    timestamps: np.ndarray,
    contract: ExternalTemporalContract,
    *,
    minimum_interval_sec: float = 0.070,
    maximum_interval_sec: float = 0.150,
) -> np.ndarray:
    """Validate real 10 Hz observations without interpolation or copied rows."""
    if contract.target_hz != 10.0:
        raise ExternalContractError(f"{contract.name}: cadence is unresolved or unsupported")
    value = np.asarray(timestamps, dtype=np.float64)
    if value.shape != (contract.rows,) or not np.isfinite(value).all():
        raise ExternalContractError(f"{contract.name}: expected {contract.rows} finite timestamps")
    intervals = np.diff(value)
    if np.any(intervals <= 0.0):
        raise ExternalContractError(f"{contract.name}: timestamps must be strictly monotonic")
    if np.any(intervals < minimum_interval_sec) or np.any(intervals > maximum_interval_sec):
        raise ExternalContractError(f"{contract.name}: observed cadence is outside 10 Hz bounds")
    return value


def fallvision_midhip_pixel_features(keypoints_xy: np.ndarray) -> np.ndarray:
    """Reproduce the checked-in FallVision coordinate transform, not its scaler.

    Input is COCO-17 pixel XY with shape ``(..., 17, 2)``. The output contains
    shoulders through ankles (COCO indices 5..16), translated by the midpoint
    of hips 11 and 12. Body-scale normalization is intentionally not invented.
    """
    value = np.asarray(keypoints_xy)
    if value.ndim < 2 or value.shape[-2:] != (17, 2):
        raise ExternalContractError(f"FallVision: expected (..., 17, 2), got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ExternalContractError("FallVision: finite floating COCO-17 XY required")
    midhip = (value[..., 11, :] + value[..., 12, :]) * 0.5
    selected = value[..., FALLVISION_JOINT_INDICES, :] - midhip[..., None, :]
    return np.ascontiguousarray(selected.reshape(*value.shape[:-2], 24), dtype=np.float32)


def wardy_from_dmc_109(_: np.ndarray) -> np.ndarray:
    """Reject the unsafe assumption that Wardy 80D is a DMC 109D slice."""
    raise ExternalContractError(
        "Wardy 80D requires its own verified geometry/motion feature builder; "
        "direct conversion from DMC 109D is forbidden"
    )
