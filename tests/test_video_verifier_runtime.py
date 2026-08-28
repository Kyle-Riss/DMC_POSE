import time

import numpy as np

from swin3d_verifier import VerifierPrediction
from video_verifier_runtime import (
    CandidateVideoVerifierRuntime,
    verifier_trigger_active,
)


class FakeService:
    def __init__(self):
        self.calls = 0

    def predict(self, frames):
        self.calls += 1
        probability = 0.2 if self.calls == 1 else 0.7
        return VerifierPrediction(probability, 1.0, len(frames), 16)


class FakeDeltaService:
    feature_mode = "delta_embedding_v1"
    threshold = 0.4

    def predict_pair(self, baseline, post):
        from swin3d_verifier import PairVerifierPrediction
        return PairVerifierPrediction(0.8, 2.0, len(baseline), len(post), 16)


def test_runtime_uses_existing_frames_and_triggers_only_on_rising_edge():
    service = FakeService()
    runtime = CandidateVideoVerifierRuntime("bed_test", service, sample_hz=4, frame_width=64, rearm_sec=0)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for index in range(41):
        runtime.observe(frame, frame_seq=index, mono_ts=index * 0.25)
    assert runtime.update_trigger(True, mono_ts=10.0)
    deadline = time.monotonic() + 2
    while runtime.status()["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    status = runtime.status()
    assert status["ready"]
    assert status["candidate"]
    assert status["completed_total"] == 1
    assert service.calls == 2
    assert not runtime.update_trigger(True, mono_ts=10.1)


def test_active_signal_is_not_consumed_before_ring_is_ready():
    service = FakeService()
    runtime = CandidateVideoVerifierRuntime(
        "bed_test", service, sample_hz=4, frame_width=64, rearm_sec=0,
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for index in range(8):
        runtime.observe(frame, frame_seq=index, mono_ts=index * 0.25)
    assert not runtime.update_trigger(True, mono_ts=1.75)
    for index in range(8, 41):
        runtime.observe(frame, frame_seq=index, mono_ts=index * 0.25)
    assert runtime.update_trigger(True, mono_ts=10.0)
    runtime.close()


def test_motion_or_fusion_can_trigger_without_temporal_readiness():
    assert verifier_trigger_active(rapid_motion=True, fusion_phase="WARMING")
    assert verifier_trigger_active(rapid_motion=False, fusion_phase="VERIFY")
    assert not verifier_trigger_active(rapid_motion=False, fusion_phase="WARMING")


def test_delta_service_uses_learned_pair_probability():
    runtime = CandidateVideoVerifierRuntime(
        "bed_test", FakeDeltaService(), sample_hz=5, frame_width=64, rearm_sec=0,
    )
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for index in range(51):
        runtime.observe(frame, frame_seq=index, mono_ts=index * 0.2)
    assert runtime.update_trigger(True, mono_ts=10.0)
    deadline = time.monotonic() + 2
    while runtime.status()["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    status = runtime.status()
    assert status["candidate"]
    assert status["pair_probability"] == 0.8
    assert status["decision_mode"] == "delta_embedding_v1"
