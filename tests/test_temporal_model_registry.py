import pytest
import torch

from temporal_model import (
    LEGACY_ARCHITECTURE,
    MODEL_ARCHITECTURES,
    FallTCN,
    architecture_from_checkpoint,
    build_temporal_model,
)


@pytest.mark.parametrize("architecture", MODEL_ARCHITECTURES)
def test_registered_model_output_shape(architecture):
    model = build_temporal_model(architecture, 109).eval()
    with torch.no_grad():
        output = model(torch.zeros((3, 30, 109), dtype=torch.float32))
    assert output.shape == (3,)
    assert torch.isfinite(output).all()


def test_old_checkpoint_defaults_to_exact_legacy_tcn():
    architecture = architecture_from_checkpoint({"feature_count": 109})
    assert architecture == LEGACY_ARCHITECTURE
    assert isinstance(build_temporal_model(architecture, 109), FallTCN)


def test_unknown_architecture_fails_closed():
    with pytest.raises(ValueError, match="unsupported"):
        build_temporal_model("mystery_model", 109)


def test_small_gru_has_lower_capacity_than_production_gru():
    regular = build_temporal_model("gru_v1", 109)
    small = build_temporal_model("gru_small_v1", 109)
    regular_parameters = sum(parameter.numel() for parameter in regular.parameters())
    small_parameters = sum(parameter.numel() for parameter in small.parameters())
    assert small_parameters < regular_parameters / 10


def test_transformer_is_causal_at_earlier_outputs_by_construction():
    model = build_temporal_model("temporal_transformer_v1", 109).eval()
    short = torch.randn((1, 10, 109), dtype=torch.float32)
    long = torch.cat((short, torch.randn((1, 5, 109), dtype=torch.float32)), dim=1)
    with torch.no_grad():
        short_result = model(short)
        prefix_result = model(long[:, :10])
    assert torch.allclose(short_result, prefix_result, atol=1e-6)
