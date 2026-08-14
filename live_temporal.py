"""Thread-safe TCN model service and per-camera shadow sequence runner."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from threading import Lock

import numpy as np
import torch

from temporal_features import temporal_feature_vector
from temporal_model import FallTCN
from temporal_sequence import decide_observation


class TemporalModelService:
    """Load one immutable TCN artifact shared by all camera threads."""

    def __init__(self, model_path: Path, report_path: Path, *, device: str = "cpu", threshold: float | None = None):
        self.model_path = Path(model_path).resolve()
        self.report_path = Path(report_path).resolve()
        self.device = torch.device(device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.feature_count = int(checkpoint["feature_count"])
        if self.feature_count != 109:
            raise ValueError(f"TCN feature count must be 109, got {self.feature_count}")
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(1, 1, self.feature_count)
        self.std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(1, 1, self.feature_count)
        if np.any(self.std <= 0):
            raise ValueError("TCN checkpoint contains non-positive standard deviations")
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        self.threshold = float(threshold if threshold is not None else report["validation"]["threshold"])
        self.model = FallTCN(self.feature_count).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.lock = Lock()

    def predict(self, window: np.ndarray) -> float:
        array = np.asarray(window, dtype=np.float32)
        if array.shape != (30, self.feature_count):
            raise ValueError(f"TCN window must be (30, {self.feature_count}), got {array.shape}")
        normalized = (array[None, ...] - self.mean) / self.std
        tensor = torch.from_numpy(normalized).to(self.device)
        with self.lock, torch.inference_mode():
            return float(torch.sigmoid(self.model(tensor))[0].cpu())


class TemporalShadowRunner:
    """Build a TCN v1 window from consecutive, actually observed poses.

    The deployed 109-feature checkpoint was not trained with an explicit
    missing-row mask. Therefore no synthetic zero skeleton is inserted.
    Observations faster than the accepted cadence are skipped and a late
    observation starts a fresh window.
    """

    def __init__(
        self,
        service: TemporalModelService,
        *,
        sample_hz: float = 10.0,
        inference_stride: int = 5,
        persistence: int = 2,
        min_interval_sec: float = 0.070,
        max_interval_sec: float = 0.150,
    ):
        if sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        self.service = service
        self.sample_hz = float(sample_hz)
        self.period = 1.0 / self.sample_hz
        self.inference_stride = max(1, int(inference_stride))
        self.persistence = max(1, int(persistence))
        self.min_interval_sec = float(min_interval_sec)
        self.max_interval_sec = float(max_interval_sec)
        if not 0.0 < self.min_interval_sec <= self.max_interval_sec:
            raise ValueError("sampling interval must satisfy 0 < min <= max")
        self.features = deque(maxlen=30)
        self.last_sample_ts: float | None = None
        self.last_observation_ts: float | None = None
        self.previous_xy_norm: np.ndarray | None = None
        self.previous_visibility: np.ndarray | None = None
        self.samples_since_inference = 0
        self.probability = 0.0
        self.consecutive_positive = 0
        self.prediction_count = 0
        # Retained as zero-valued compatibility fields for the current API.
        self.missing_samples_total = 0
        self.gap_reset_total = 0
        self.duplicate_skip_total = 0
        self.non_monotonic_skip_total = 0
        self.last_dt_sec: float | None = None
        self.last_action = "not_started"
        self.timestamp_source = "unknown"
        self.latest_feature: np.ndarray | None = None

    def reset(self) -> None:
        self.features.clear()
        self.last_sample_ts = None
        self.last_observation_ts = None
        self.previous_xy_norm = None
        self.previous_visibility = None
        self.samples_since_inference = 0
        self.probability = 0.0
        self.consecutive_positive = 0
        self.last_dt_sec = None
        self.latest_feature = None

    def _reset_for_gap(self) -> None:
        self.gap_reset_total += 1
        self.reset()
        self.last_action = "gap_reset"

    def observe_gap(self, timestamp: float) -> None:
        """Invalidate history when no primary pose arrives within 150 ms."""
        timestamp = float(timestamp)
        if (
            self.last_sample_ts is not None
            and timestamp - self.last_sample_ts > self.max_interval_sec
        ):
            self._reset_for_gap()

    def _append_sample(
        self,
        timestamp: float,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        pose_probs: np.ndarray,
        *,
        dt: float,
    ) -> None:
        vector, xy_norm, visibility = temporal_feature_vector(
            keypoints_xy,
            keypoints_conf,
            pose_probs,
            previous_xy_norm=self.previous_xy_norm,
            previous_visibility=self.previous_visibility,
            dt=dt,
        )
        self.previous_xy_norm = xy_norm
        self.previous_visibility = visibility
        self.features.append(vector)
        self.latest_feature = np.asarray(vector, dtype=np.float32).copy()
        self.last_sample_ts = float(timestamp)
        self.last_observation_ts = float(timestamp)
        self.last_dt_sec = float(dt)
        self.last_action = "append"
        self.samples_since_inference += 1

    def push(
        self,
        timestamp: float,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        pose_probs: np.ndarray,
        *,
        timestamp_source: str = "decode_mono_ts",
    ) -> dict:
        timestamp = float(timestamp)
        self.timestamp_source = str(timestamp_source)

        decision = decide_observation(
            timestamp,
            self.last_sample_ts,
            min_interval_sec=self.min_interval_sec,
            max_interval_sec=self.max_interval_sec,
        )
        if decision.action == "non_monotonic_skip":
            self.non_monotonic_skip_total += 1
            self.last_action = "non_monotonic_skip"
            self.last_dt_sec = decision.dt_sec
            return self.status()
        if decision.action == "duplicate_skip":
            self.duplicate_skip_total += 1
            self.last_action = "duplicate_skip"
            self.last_dt_sec = decision.dt_sec
            return self.status()
        if decision.reset:
            self._reset_for_gap()

        self._append_sample(
            timestamp,
            keypoints_xy,
            keypoints_conf,
            pose_probs,
            dt=self.period if decision.dt_sec is None or decision.reset else decision.dt_sec,
        )
        return self._predict_if_due()

    def latest_observation(self) -> tuple[float, np.ndarray] | None:
        """Return the most recently appended observed-only row for recording."""
        if self.last_sample_ts is None or self.latest_feature is None:
            return None
        return float(self.last_sample_ts), self.latest_feature.copy()

    def _predict_if_due(self) -> dict:
        if (
            len(self.features) == 30
            and self.samples_since_inference >= self.inference_stride
        ):
            self.probability = self.service.predict(np.stack(self.features))
            self.prediction_count += 1
            self.samples_since_inference = 0
            if self.probability >= self.service.threshold:
                self.consecutive_positive += 1
            else:
                self.consecutive_positive = 0
        return self.status()

    def status(self, sample_ts: float | None = None) -> dict:
        return {
            "ready": len(self.features) == 30,
            "probability": self.probability,
            "candidate": self.consecutive_positive >= self.persistence,
            "threshold": self.service.threshold,
            "samples": len(self.features),
            "prediction_count": self.prediction_count,
            "sample_timestamp": (
                self.last_sample_ts if sample_ts is None else sample_ts
            ),
            "sample_hz": self.sample_hz,
            "missing_samples_window": 0,
            "missing_samples_total": 0,
            "gap_reset_total": int(self.gap_reset_total),
            "duplicate_skip_total": int(self.duplicate_skip_total),
            "non_monotonic_skip_total": int(self.non_monotonic_skip_total),
            "last_dt_sec": self.last_dt_sec,
            "last_action": self.last_action,
            "timestamp_source": self.timestamp_source,
            "sampling_contract": (
                f"observed_only_{self.min_interval_sec * 1000:.0f}_"
                f"{self.max_interval_sec * 1000:.0f}ms"
            ),
        }
