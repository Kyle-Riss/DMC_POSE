"""Pi-oriented adapter joining observed-only TCN windows to candidate recording."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from live_temporal import TemporalShadowRunner
from temporal_candidate_recorder import TemporalCandidateRecorder


class EdgeTemporalRuntime:
    """Own one camera/track temporal buffer and automatically retain candidates."""

    def __init__(
        self,
        runner: TemporalShadowRunner,
        recorder: TemporalCandidateRecorder,
        *,
        node_id: str,
        camera_id: str,
        boot_id: str,
        model_bundle_version: str,
    ):
        self.runner = runner
        self.recorder = recorder
        self.node_id = str(node_id)
        self.camera_id = str(camera_id)
        self.boot_id = str(boot_id)
        self.model_bundle_version = str(model_bundle_version)
        self.track_id: int | None = None
        self.sample_timestamps: deque[float] = deque(maxlen=30)

    def reset(self, reason: str = "explicit_reset") -> dict[str, Any]:
        self.runner.reset()
        self.sample_timestamps.clear()
        self.track_id = None
        status = self.runner.status()
        status["edge_reset_reason"] = reason
        return status

    def observe_no_person(self, timestamp: float) -> dict[str, Any]:
        previous_resets = self.runner.gap_reset_total
        self.runner.observe_gap(float(timestamp))
        if self.runner.gap_reset_total != previous_resets:
            self.sample_timestamps.clear()
            self.track_id = None
        return self.runner.status()

    def push(
        self,
        *,
        timestamp: float,
        captured_at: str,
        frame_seq: int,
        track_id: int,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        pose_probs: np.ndarray,
        roi_version: int,
        fusion_risk: float,
        fusion_phase: str,
        quality: float,
        evidence: list[str],
        timestamp_source: str = "capture_mono_ts",
    ) -> dict[str, Any]:
        parsed = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("captured_at must include timezone")
        incoming_track = int(track_id)
        if self.track_id is not None and incoming_track != self.track_id:
            self.runner.reset()
            self.sample_timestamps.clear()
        self.track_id = incoming_track

        resets_before = self.runner.gap_reset_total
        status = self.runner.push(
            float(timestamp), keypoints_xy, keypoints_conf, pose_probs,
            timestamp_source=timestamp_source,
        )
        if self.runner.gap_reset_total != resets_before:
            self.sample_timestamps.clear()
        if status["last_action"] == "append":
            self.sample_timestamps.append(float(timestamp))

        status = dict(status)
        status.update({
            "candidate_recorded": False,
            "candidate_id": None,
            "candidate_record_reason": None,
        })
        if status["candidate"] and status["ready"] and len(self.sample_timestamps) == 30:
            receipt = self.recorder.submit(
                np.stack(self.runner.features),
                {
                    "node_id": self.node_id,
                    "camera_id": self.camera_id,
                    "boot_id": self.boot_id,
                    "frame_seq": int(frame_seq),
                    "captured_at": captured_at,
                    "track_id": incoming_track,
                    "model_bundle_version": self.model_bundle_version,
                    "roi_version": int(roi_version),
                    "trigger": "temporal_candidate",
                    "temporal_probability": float(status["probability"]),
                    "fusion_risk": float(fusion_risk),
                    "fusion_phase": str(fusion_phase),
                    "quality": float(quality),
                    "evidence": list(evidence),
                    "sample_timestamps": list(self.sample_timestamps),
                },
            )
            status["candidate_recorded"] = receipt.accepted
            status["candidate_id"] = receipt.candidate_id
            status["candidate_record_reason"] = receipt.reason
        return status

