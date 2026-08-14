"""Small causal TCN architecture shared by training, evaluation, and serving."""

from __future__ import annotations

import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, 3, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, 3, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.network(x) + self.residual(x))


class FallTCN(nn.Module):
    def __init__(self, feature_count: int, channels: int = 48, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            TemporalBlock(feature_count, channels, 1, dropout),
            TemporalBlock(channels, channels, 2, dropout),
            TemporalBlock(channels, channels, 4, dropout),
        )
        self.output = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(x.transpose(1, 2))
        return self.output(encoded[:, :, -1]).squeeze(1)
