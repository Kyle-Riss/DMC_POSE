"""Strict, versioned wire contracts for DMC_POSE edge nodes.

Only inference facts and event metadata cross the control plane. RTSP
credentials and continuous raw video are deliberately absent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EDGE_CONTRACT_VERSION = "dmc_pose_edge_v1"
EDGE_FEATURE_SCHEMA = "pose_temporal_109_v1"

FusionPhase = Literal[
    "NO_PERSON", "INSUFFICIENT", "WARMING", "SAFE", "CANDIDATE",
    "VERIFY", "SHADOW_ALERT",
]
RuntimeMode = Literal["EMPTY", "OCCUPIED", "BURST", "DEGRADED"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _timezone_required(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value


class EdgeCapabilities(StrictModel):
    camera_capture: bool = True
    rtsp_publish: bool = True
    motion_watcher: bool = True
    automatic_bed_roi: bool = False
    pose_inference: bool = False
    temporal_inference: bool = False
    fusion: bool = False
    event_frame_upload: bool = False


class EdgeHeartbeat(StrictModel):
    contract_version: Literal[EDGE_CONTRACT_VERSION] = EDGE_CONTRACT_VERSION
    node_id: str = Field(min_length=1, max_length=128)
    camera_id: str = Field(min_length=1, max_length=128)
    boot_id: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    sent_at: datetime
    software_version: str = Field(min_length=1, max_length=128)
    model_bundle_version: str | None = Field(default=None, max_length=128)
    uptime_sec: float = Field(ge=0)
    capture_connected: bool
    capture_fps: float = Field(ge=0, le=240)
    watcher_fps: float = Field(ge=0, le=240)
    runtime_mode: RuntimeMode
    roi_state: Literal["UNAVAILABLE", "BOOTSTRAP", "READY", "DEGRADED"]
    roi_version: int = Field(ge=0)
    spool_depth: int = Field(ge=0)
    spool_bytes: int = Field(ge=0)
    storage_free_mb: float = Field(ge=0)
    capabilities: EdgeCapabilities

    _sent_at_tz = field_validator("sent_at")(_timezone_required)


class EdgeInferenceResult(StrictModel):
    contract_version: Literal[EDGE_CONTRACT_VERSION] = EDGE_CONTRACT_VERSION
    node_id: str = Field(min_length=1, max_length=128)
    camera_id: str = Field(min_length=1, max_length=128)
    boot_id: str = Field(min_length=1, max_length=128)
    frame_seq: int = Field(ge=0)
    captured_at: datetime
    model_bundle_version: str = Field(min_length=1, max_length=128)
    roi_version: int = Field(ge=0)
    primary_track_id: int | None = Field(default=None, ge=0)
    person_present: bool
    body_in_bed_ratio: float = Field(ge=0, le=1)
    pose_label: str | None = Field(default=None, max_length=64)
    pose_confidence: float = Field(ge=0, le=1)
    temporal_ready: bool
    temporal_samples: int = Field(ge=0, le=30)
    temporal_probability: float = Field(ge=0, le=1)
    temporal_candidate: bool
    fusion_phase: FusionPhase
    fusion_risk: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, max_length=32)
    quality: float = Field(ge=0, le=1)
    inference_latency_ms: float = Field(ge=0)

    _captured_at_tz = field_validator("captured_at")(_timezone_required)

    @model_validator(mode="after")
    def temporal_state_is_consistent(self):
        if self.temporal_ready and self.temporal_samples != 30:
            raise ValueError("temporal_ready requires exactly 30 observed samples")
        if not self.person_present and self.primary_track_id is not None:
            raise ValueError("primary_track_id requires person_present")
        return self


class EdgeEventStart(StrictModel):
    contract_version: Literal[EDGE_CONTRACT_VERSION] = EDGE_CONTRACT_VERSION
    event_id: str = Field(min_length=1, max_length=160)
    node_id: str = Field(min_length=1, max_length=128)
    camera_id: str = Field(min_length=1, max_length=128)
    boot_id: str = Field(min_length=1, max_length=128)
    started_at: datetime
    start_frame_seq: int = Field(ge=0)
    event_type: Literal["BED_EXIT", "FALL", "BED_EXIT_FALL", "CANDIDATE"]
    model_bundle_version: str = Field(min_length=1, max_length=128)
    roi_version: int = Field(ge=0)
    pre_event_frames_available: int = Field(ge=0)
    pre_event_coverage_sec: float = Field(ge=0)
    peak_risk: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list, max_length=32)

    _started_at_tz = field_validator("started_at")(_timezone_required)


class EdgeEventEnd(StrictModel):
    contract_version: Literal[EDGE_CONTRACT_VERSION] = EDGE_CONTRACT_VERSION
    event_id: str = Field(min_length=1, max_length=160)
    node_id: str = Field(min_length=1, max_length=128)
    camera_id: str = Field(min_length=1, max_length=128)
    boot_id: str = Field(min_length=1, max_length=128)
    ended_at: datetime
    end_frame_seq: int = Field(ge=0)
    peak_risk: float = Field(ge=0, le=1)
    uploaded_frame_count: int = Field(ge=0)
    close_reason: Literal["completed", "cancelled", "timeout", "shutdown"]

    _ended_at_tz = field_validator("ended_at")(_timezone_required)


class ModelArtifact(StrictModel):
    role: Literal["bed_seg", "pose", "posture", "temporal", "fusion_config"]
    filename: str = Field(min_length=1, max_length=200)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    format: Literal["onnx", "tflite", "ncnn", "torchscript", "json"]


class EdgeModelBundle(StrictModel):
    contract_version: Literal[EDGE_CONTRACT_VERSION] = EDGE_CONTRACT_VERSION
    bundle_version: str = Field(min_length=1, max_length=128)
    status: Literal["benchmark_required", "shadow", "production"]
    created_at: datetime
    target: Literal["rpi4", "rpi5", "rpi5_hailo", "generic_arm64"]
    feature_schema: Literal[EDGE_FEATURE_SCHEMA] = EDGE_FEATURE_SCHEMA
    sample_hz: float = Field(gt=0)
    temporal_rows: Literal[30] = 30
    artifacts: list[ModelArtifact]

    _created_at_tz = field_validator("created_at")(_timezone_required)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
