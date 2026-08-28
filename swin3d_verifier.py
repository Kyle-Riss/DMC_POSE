"""Candidate-only Swin3D-B fall verifier with a frozen linear probe."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import Swin3D_B_Weights, swin3d_b


WEIGHT_SHA256 = "7c6ae6fa165f481a9c71156644a7c0e61bb393e470ca3671b8d24a30d365ffc6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def progressive_decision(
    baseline: float, post_scores: Sequence[float], *, absolute_threshold: float, delta_threshold: float,
) -> dict:
    if not post_scores:
        return {"ready": False, "candidate": False, "baseline": float(baseline), "post_max": None, "delta": None}
    post_max = max(float(value) for value in post_scores)
    delta = post_max - float(baseline)
    return {
        "ready": True,
        "candidate": post_max >= absolute_threshold and delta >= delta_threshold,
        "baseline": float(baseline),
        "post_max": post_max,
        "delta": delta,
        "absolute_threshold": float(absolute_threshold),
        "delta_threshold": float(delta_threshold),
    }


@dataclass(frozen=True)
class VerifierPrediction:
    probability: float
    latency_ms: float
    source_frames: int
    sampled_frames: int


@dataclass(frozen=True)
class PairVerifierPrediction:
    probability: float
    latency_ms: float
    baseline_frames: int
    post_frames: int
    sampled_frames_per_clip: int


class Swin3DVerifierService:
    """Load one shared RGB backbone and score bounded frame snapshots."""

    def __init__(self, weight_path: str | Path, probe_path: str | Path, *, device: str = "cuda", verify_hash: bool = True):
        self.weight_path = Path(weight_path).resolve()
        self.probe_path = Path(probe_path).resolve()
        if verify_hash and sha256(self.weight_path) != WEIGHT_SHA256:
            raise ValueError("Swin3D-B weight hash mismatch")
        self.device = torch.device(device)
        model = swin3d_b(weights=None)
        state = torch.load(self.weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.embedding_size = int(model.head.in_features)
        model.head = nn.Identity()
        self.model = model.eval().to(self.device)
        self.transform = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1.transforms()
        probe = np.load(self.probe_path)
        self.feature_mode = str(
            probe["feature_mode"].reshape(-1)[0]
            if "feature_mode" in probe.files else "single_embedding_v1"
        )
        self.threshold = float(
            probe["threshold"].reshape(-1)[0]
            if "threshold" in probe.files else 0.5
        )
        self.mean = probe["mean"].astype(np.float32)
        self.scale = probe["scale"].astype(np.float32)
        self.coefficient = probe["coefficient"].reshape(-1).astype(np.float32)
        self.intercept = float(probe["intercept"].reshape(-1)[0])
        if not all(len(value) == self.embedding_size for value in (self.mean, self.scale, self.coefficient)):
            raise ValueError("linear probe dimension mismatch")
        self.lock = Lock()

    @staticmethod
    def _sample(frames: Sequence[np.ndarray], count: int = 16) -> list[np.ndarray]:
        if len(frames) < count:
            raise ValueError(f"verifier needs at least {count} frames, got {len(frames)}")
        indices = np.rint(np.linspace(0, len(frames) - 1, count)).astype(int)
        return [frames[index] for index in indices]

    def _embedding(self, frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
        sampled = self._sample(frames_bgr)
        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in sampled])
        clip = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        tensor = self.transform(clip).unsqueeze(0).to(self.device, non_blocking=True)
        with self.lock, torch.inference_mode(), torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            embedding = self.model(tensor)[0].float().cpu().numpy()
        return embedding

    def _probability(self, feature: np.ndarray) -> float:
        normalized = (feature - self.mean) / self.scale
        logit = float(np.dot(normalized, self.coefficient) + self.intercept)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))

    def predict(self, frames_bgr: Sequence[np.ndarray]) -> VerifierPrediction:
        if self.feature_mode != "single_embedding_v1":
            raise ValueError(f"predict is unavailable for {self.feature_mode}")
        started = time.perf_counter()
        embedding = self._embedding(frames_bgr)
        return VerifierPrediction(
            probability=self._probability(embedding),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            source_frames=len(frames_bgr),
            sampled_frames=16,
        )

    def predict_pair(
        self,
        baseline_frames_bgr: Sequence[np.ndarray],
        post_frames_bgr: Sequence[np.ndarray],
    ) -> PairVerifierPrediction:
        if self.feature_mode != "delta_embedding_v1":
            raise ValueError(f"predict_pair is unavailable for {self.feature_mode}")
        started = time.perf_counter()
        baseline = self._embedding(baseline_frames_bgr)
        post = self._embedding(post_frames_bgr)
        return PairVerifierPrediction(
            probability=self._probability(post - baseline),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            baseline_frames=len(baseline_frames_bgr),
            post_frames=len(post_frames_bgr),
            sampled_frames_per_clip=16,
        )
