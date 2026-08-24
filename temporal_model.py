"""Temporal model registry shared by training, evaluation, and serving."""

from __future__ import annotations

import torch
from torch import nn


LEGACY_ARCHITECTURE = "causal_tcn_v1"


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


class FallGRU(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.recurrent = nn.GRU(
            feature_count,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.recurrent(x)
        return self.output(hidden[-1]).squeeze(1)


class FallBiLSTM(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.recurrent = nn.LSTM(
            feature_count,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.recurrent(x)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(encoded).squeeze(1)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, width: int, maximum_rows: int = 512):
        super().__init__()
        position = torch.arange(maximum_rows, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / width))
        encoding = torch.zeros(maximum_rows, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[0]:
            raise ValueError(f"sequence has {x.shape[1]} rows; maximum is {self.encoding.shape[0]}")
        return x + self.encoding[: x.shape[1]].to(dtype=x.dtype)


class FallTemporalTransformer(nn.Module):
    def __init__(
        self,
        feature_count: int,
        width: int = 128,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.projection = nn.Linear(feature_count, width)
        self.position = SinusoidalPositionEncoding(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.normalization = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.position(self.projection(x))
        rows = encoded.shape[1]
        causal_mask = torch.triu(
            torch.ones(rows, rows, device=encoded.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.encoder(encoded, mask=causal_mask)
        return self.output(self.normalization(encoded[:, -1])).squeeze(1)


MODEL_ARCHITECTURES = (
    LEGACY_ARCHITECTURE,
    "gru_v1",
    "bilstm_v1",
    "temporal_transformer_v1",
)


def build_temporal_model(architecture: str, feature_count: int) -> nn.Module:
    """Build an explicitly versioned temporal model.

    Old checkpoints did not store an architecture, so callers must default
    missing metadata to ``causal_tcn_v1`` before invoking this function.
    """
    normalized = str(architecture).strip().lower()
    builders = {
        LEGACY_ARCHITECTURE: FallTCN,
        "tcn": FallTCN,
        "gru_v1": FallGRU,
        "gru": FallGRU,
        "bilstm_v1": FallBiLSTM,
        "bilstm": FallBiLSTM,
        "temporal_transformer_v1": FallTemporalTransformer,
        "transformer": FallTemporalTransformer,
    }
    try:
        builder = builders[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported temporal architecture: {architecture}") from error
    return builder(int(feature_count))


def architecture_from_checkpoint(checkpoint: dict) -> str:
    return str(checkpoint.get("architecture", LEGACY_ARCHITECTURE))
