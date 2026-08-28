import numpy as np
import pytest

from swin3d_verifier import Swin3DVerifierService, progressive_decision


def test_progressive_gate_requires_absolute_and_delta_evidence():
    assert progressive_decision(0.2, [0.5, 0.7], absolute_threshold=0.6, delta_threshold=0.3)["candidate"]
    assert not progressive_decision(0.6, [0.7], absolute_threshold=0.6, delta_threshold=0.3)["candidate"]
    assert not progressive_decision(0.2, [0.5], absolute_threshold=0.6, delta_threshold=0.3)["candidate"]


def test_progressive_gate_is_not_ready_without_post_clip():
    result = progressive_decision(0.2, [], absolute_threshold=0.5, delta_threshold=0.1)
    assert not result["ready"]
    assert result["post_max"] is None


def test_frame_sampling_is_uniform_and_requires_16_frames():
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(32)]
    sampled = Swin3DVerifierService._sample(frames)
    assert len(sampled) == 16
    assert sampled[0][0, 0, 0] == 0
    assert sampled[-1][0, 0, 0] == 31
    with pytest.raises(ValueError, match="at least 16"):
        Swin3DVerifierService._sample(frames[:15])
