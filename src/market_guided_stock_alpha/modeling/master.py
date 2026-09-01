"""Canonical MASTER backbone and layout-aware public model interface.

The neural blocks intentionally stay close to the public SJTU-DMTai/MASTER
implementation used by the canonical stock-alpha research pipeline. The
wrapper keeps date-level market state separate from stock-level storage until
model execution.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from market_guided_stock_alpha.dataset.layout import (
    LOOKBACK_TRADING_DAYS,
    FeatureLayout,
)


@dataclass(frozen=True)
class MasterConfig:
    """Single source of truth for the canonical MASTER architecture."""

    stock_feature_dim: int
    market_feature_dim: int
    d_model: int = 256
    temporal_heads: int = 4
    spatial_heads: int = 2
    dropout: float = 0.5
    beta: float = 2.0
    lookback: int = LOOKBACK_TRADING_DAYS
    ablation: str = "none"

    def __post_init__(self) -> None:
        integer_fields = {
            "stock_feature_dim": self.stock_feature_dim,
            "market_feature_dim": self.market_feature_dim,
            "d_model": self.d_model,
            "temporal_heads": self.temporal_heads,
            "spatial_heads": self.spatial_heads,
            "lookback": self.lookback,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if self.lookback != LOOKBACK_TRADING_DAYS:
            raise ValueError("canonical MASTER lookback must equal eight")
        if self.d_model % 2:
            raise ValueError("d_model must be even for sinusoidal encoding")
        if self.d_model % self.temporal_heads:
            raise ValueError("d_model must be divisible by temporal_heads")
        if self.d_model % self.spatial_heads:
            raise ValueError("d_model must be divisible by spatial_heads")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(float(self.beta)) or self.beta <= 0.0:
            raise ValueError("beta must be finite and positive")
        from market_guided_stock_alpha.modeling.ablation import (
            validate_ablation_mode,
        )

        validate_ablation_mode(self.ablation)

    @classmethod
    def from_layout(cls, layout: FeatureLayout, **overrides: Any) -> "MasterConfig":
        return cls(
            stock_feature_dim=layout.stock_dim,
            market_feature_dim=layout.market_dim,
            lookback=layout.lookback,
            **overrides,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList(
            [Dropout(p=dropout) for _ in range(nhead)]
        )
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        width = self.d_model // self.nhead
        outputs = []
        for index in range(self.nhead):
            left = index * width
            right = None if index == self.nhead - 1 else (index + 1) * width
            qh, kh, vh = q[:, :, left:right], k[:, :, left:right], v[:, :, left:right]
            attention = torch.softmax(
                torch.matmul(qh, kh.transpose(1, 2)) / self.temperature,
                dim=-1,
            )
            attention = self.attn_dropout[index](attention)
            outputs.append(torch.matmul(attention, vh).transpose(0, 1))
        attention_output = torch.concat(outputs, dim=-1)
        hidden = self.norm2(x + attention_output)
        return hidden + self.ffn(hidden)


class TAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList(
            [Dropout(p=dropout) for _ in range(nhead)]
        )
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        q, k, v = self.qtrans(x), self.ktrans(x), self.vtrans(x)
        width = self.d_model // self.nhead
        outputs = []
        for index in range(self.nhead):
            left = index * width
            right = None if index == self.nhead - 1 else (index + 1) * width
            qh, kh, vh = q[:, :, left:right], k[:, :, left:right], v[:, :, left:right]
            attention = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            attention = self.attn_dropout[index](attention)
            outputs.append(torch.matmul(attention, vh))
        attention_output = torch.concat(outputs, dim=-1)
        hidden = self.norm2(x + attention_output)
        return hidden + self.ffn(hidden)


class Gate(nn.Module):
    def __init__(self, d_input: int, d_output: int, beta: float = 1.0) -> None:
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        return self.d_output * torch.softmax(self.trans(gate_input) / self.t, dim=-1)


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.trans(values)
        query = hidden[:, -1, :].unsqueeze(-1)
        weights = torch.softmax(torch.matmul(hidden, query).squeeze(-1), dim=1).unsqueeze(1)
        return torch.matmul(weights, values).squeeze(1)


class MASTER(nn.Module):
    """Upstream-shaped backbone consuming concatenated stock and market input."""

    def __init__(
        self,
        d_feat: int,
        d_model: int,
        t_nhead: int,
        s_nhead: int,
        T_dropout_rate: float,
        S_dropout_rate: float,
        gate_input_start_index: int,
        gate_input_end_index: int,
        beta: float,
    ) -> None:
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        source = values[:, :, : self.gate_input_start_index]
        gate_input = values[
            :, -1, self.gate_input_start_index : self.gate_input_end_index
        ]
        source = source * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        return self.layers(source).squeeze(-1)


def _build_backbone(config: MasterConfig) -> MASTER:
    architecture = {
        "d_feat": config.stock_feature_dim,
        "d_model": config.d_model,
        "t_nhead": config.temporal_heads,
        "s_nhead": config.spatial_heads,
        "T_dropout_rate": config.dropout,
        "S_dropout_rate": config.dropout,
        "gate_input_start_index": config.stock_feature_dim,
        "gate_input_end_index": config.stock_feature_dim + config.market_feature_dim,
        "beta": config.beta,
    }
    if config.ablation == "none":
        return MASTER(**architecture)
    from market_guided_stock_alpha.modeling.ablation import create_master_ablation

    return create_master_ablation(MASTER, architecture, config.ablation)


class MasterModel(nn.Module):
    """Layout-aware MASTER; market state is broadcast only inside forward."""

    input_kind = "stock_and_market"

    def __init__(self, config: MasterConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _build_backbone(config)

    @classmethod
    def from_layout(cls, layout: FeatureLayout, **overrides: Any) -> "MasterModel":
        return cls(MasterConfig.from_layout(layout, **overrides))

    def forward(
        self, stock_features: torch.Tensor, market_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if market_features is None:
            raise ValueError("MASTER requires separate date-level market features")
        if stock_features.ndim != 3:
            raise ValueError("stock_features must be [N,T,F]")
        if market_features.ndim != 2:
            raise ValueError("market_features must be [T,M]")
        expected_stock = (
            stock_features.shape[0], self.config.lookback, self.config.stock_feature_dim
        )
        if tuple(stock_features.shape) != expected_stock:
            raise ValueError("stock feature shape does not match MasterConfig")
        if tuple(market_features.shape) != (
            self.config.lookback, self.config.market_feature_dim
        ):
            raise ValueError("market feature shape does not match MasterConfig")
        repeated_market = market_features.unsqueeze(0).expand(
            stock_features.shape[0], -1, -1
        )
        combined = torch.cat((stock_features, repeated_market), dim=-1)
        return self.backbone(combined)

    def config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()
