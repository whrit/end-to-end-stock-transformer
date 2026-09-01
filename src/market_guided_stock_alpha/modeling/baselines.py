"""Stock-local GRU and vanilla Transformer research baselines."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict

import torch
from torch import nn

from market_guided_stock_alpha.dataset.layout import (
    LOOKBACK_TRADING_DAYS,
    FeatureLayout,
)


def _positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))


def _dropout(value: float) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
        raise ValueError("dropout must be finite and in [0,1)")


@dataclass(frozen=True)
class GRUConfig:
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    lookback: int = LOOKBACK_TRADING_DAYS

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "num_layers", "lookback"):
            _positive_integer(name, getattr(self, name))
        if self.lookback != LOOKBACK_TRADING_DAYS:
            raise ValueError("canonical GRU lookback must equal eight")
        _dropout(self.dropout)

    @classmethod
    def from_layout(cls, layout: FeatureLayout, **overrides: Any) -> "GRUConfig":
        return cls(input_dim=layout.stock_dim, lookback=layout.lookback, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GRURegressor(nn.Module):
    """One shared GRU per stock; whole-date loss preserves cross-sectional math."""

    input_kind = "stock"

    def __init__(self, config: GRUConfig) -> None:
        super().__init__()
        self.config = config
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_dropout = nn.Dropout(config.dropout)
        self.prediction_head = nn.Linear(config.hidden_dim, 1)

    @classmethod
    def from_layout(cls, layout: FeatureLayout, **overrides: Any) -> "GRURegressor":
        return cls(GRUConfig.from_layout(layout, **overrides))

    def forward(self, stock_features: torch.Tensor) -> torch.Tensor:
        if tuple(stock_features.shape[1:]) != (
            self.config.lookback,
            self.config.input_dim,
        ):
            raise ValueError("GRU stock feature shape does not match config")
        sequence, _hidden = self.gru(stock_features)
        return self.prediction_head(
            self.output_dropout(sequence[:, -1, :])
        ).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, maximum_length: int) -> None:
        super().__init__()
        positions = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(d_model))
        )
        angles = positions * frequencies.unsqueeze(0)
        encoding = torch.zeros(maximum_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.shape[1], :]


@dataclass(frozen=True)
class TransformerConfig:
    input_dim: int
    d_model: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.5
    dim_feedforward: int = 512
    lookback: int = LOOKBACK_TRADING_DAYS

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "d_model",
            "num_layers",
            "heads",
            "dim_feedforward",
            "lookback",
        ):
            _positive_integer(name, getattr(self, name))
        if self.lookback != LOOKBACK_TRADING_DAYS:
            raise ValueError("canonical Transformer lookback must equal eight")
        if self.d_model % 2:
            raise ValueError("Transformer d_model must be even")
        if self.d_model % self.heads:
            raise ValueError("Transformer d_model must be divisible by heads")
        _dropout(self.dropout)

    @classmethod
    def from_layout(
        cls, layout: FeatureLayout, **overrides: Any
    ) -> "TransformerConfig":
        return cls(input_dim=layout.stock_dim, lookback=layout.lookback, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VanillaTransformerRegressor(nn.Module):
    input_kind = "stock"

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.lookback
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_dropout = nn.Dropout(config.dropout)
        self.prediction_head = nn.Linear(config.d_model, 1)

    @classmethod
    def from_layout(
        cls, layout: FeatureLayout, **overrides: Any
    ) -> "VanillaTransformerRegressor":
        return cls(TransformerConfig.from_layout(layout, **overrides))

    def forward(self, stock_features: torch.Tensor) -> torch.Tensor:
        if tuple(stock_features.shape[1:]) != (
            self.config.lookback,
            self.config.input_dim,
        ):
            raise ValueError("Transformer stock feature shape does not match config")
        projected = self.positional_encoding(self.input_projection(stock_features))
        encoded = self.encoder(projected)
        return self.prediction_head(
            self.output_dropout(encoded[:, -1, :])
        ).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()
