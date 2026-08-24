import torch
from torch import nn

from scripts.export_temporal_onnx import NormalizedProbabilityModel


class SumModel(nn.Module):
    def forward(self, value):
        return value.sum(dim=(1, 2))


def test_wrapper_embeds_normalization_and_probability():
    wrapper = NormalizedProbabilityModel(SumModel(), mean=[1.0, 1.0], std=[2.0, 4.0])
    value = torch.tensor([[[3.0, 5.0]]])
    expected_logit = torch.tensor([2.0])
    assert torch.allclose(wrapper(value), torch.sigmoid(expected_logit))
