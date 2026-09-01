"""Exploratory dual-frequency market gating for the canonical Stage 2 pipeline."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict

import torch
from torch import nn

from market_guided_stock_alpha.dataset.layout import FeatureLayout
from market_guided_stock_alpha.modeling.master import (
    PositionalEncoding,
    SAttention,
    TAttention,
    TemporalAttention,
)


@dataclass(frozen=True)
class DualFrequencyMasterConfig:
    stock_feature_dim: int
    alpha_feature_dim: int
    intraday_feature_dim: int
    daily_market_dim: int
    intraday_market_dim: int
    d_model: int = 256
    temporal_heads: int = 4
    spatial_heads: int = 2
    dropout: float = 0.5
    daily_gate_beta: float = 2.0
    minute_gate_beta: float = 2.0
    routing_mode: str = "learned_all_to_all"
    market_source_mode: str = "both"
    branch_output_mode: str = "both"
    fusion_mode: str = "learned_market"
    fusion_fixed_lambda: float = 1.0
    fusion_lambda_init: float = 1.0
    feature_gate_mode: str = "market_gate"
    lookback: int = 8

    def __post_init__(self) -> None:
        dimensions = (
            self.stock_feature_dim,
            self.alpha_feature_dim,
            self.intraday_feature_dim,
            self.daily_market_dim,
            self.intraday_market_dim,
            self.d_model,
            self.temporal_heads,
            self.spatial_heads,
            self.lookback,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("dual-frequency dimensions must be positive integers")
        if self.alpha_feature_dim + self.intraday_feature_dim != self.stock_feature_dim:
            raise ValueError("dual-frequency stock branch dimensions do not sum")
        if self.d_model % 2:
            raise ValueError("dual-frequency d_model must be even")
        if self.d_model % self.temporal_heads or self.d_model % self.spatial_heads:
            raise ValueError("dual-frequency d_model must be divisible by both head counts")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dual-frequency dropout must be in [0,1)")
        for name in ("daily_gate_beta", "minute_gate_beta"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and positive".format(name))
        domains = {
            "routing_mode": ("learned_all_to_all", "matched"),
            "market_source_mode": ("both", "daily_only", "intraday_only"),
            "branch_output_mode": ("both", "daily_only", "intraday_only"),
            "fusion_mode": ("learned_market", "fixed"),
            "feature_gate_mode": ("market_gate", "uniform"),
        }
        for name, allowed in domains.items():
            if getattr(self, name) not in allowed:
                raise ValueError("invalid dual-frequency {}".format(name))
        if not math.isfinite(self.fusion_fixed_lambda) or self.fusion_fixed_lambda < 0.0:
            raise ValueError("fusion_fixed_lambda must be finite and non-negative")
        if not math.isfinite(self.fusion_lambda_init) or self.fusion_lambda_init <= 0.0:
            raise ValueError("fusion_lambda_init must be finite and positive")

    @classmethod
    def from_layout(
        cls, layout: FeatureLayout, **overrides: Any
    ) -> "DualFrequencyMasterConfig":
        if not layout.intraday_market_family:
            raise ValueError("dual-frequency model requires the exploratory layout")
        alpha_left, alpha_right = layout.family_slices["alpha158"]
        intra_left, intra_right = layout.family_slices["intraday_price_volume"]
        return cls(
            stock_feature_dim=layout.stock_dim,
            alpha_feature_dim=alpha_right - alpha_left,
            intraday_feature_dim=intra_right - intra_left,
            daily_market_dim=layout.market_dim,
            intraday_market_dim=layout.intraday_market_dim,
            lookback=layout.lookback,
            **overrides,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DualFrequencyMasterModel(nn.Module):
    """Separate daily/minute gates followed by the canonical MASTER backbone."""

    input_kind = "dual_frequency"

    def __init__(self, config: DualFrequencyMasterConfig) -> None:
        super().__init__()
        self.config = config
        a, i = config.alpha_feature_dim, config.intraday_feature_dim
        d, m = config.daily_market_dim, config.intraday_market_dim
        self.daily_to_alpha = nn.Linear(d, a)
        self.intraday_to_alpha = nn.Linear(m, a)
        self.daily_to_intraday = nn.Linear(d, i)
        self.intraday_to_intraday = nn.Linear(m, i)
        self.alpha_routing_logits = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.intraday_routing_logits = nn.Parameter(torch.tensor([0.0, 1.0]))
        self.daily_projection = nn.Linear(a, config.d_model)
        self.intraday_projection = nn.Linear(i, config.d_model)
        if config.fusion_mode == "learned_market":
            self.fusion_controller = nn.Linear(d + m, 1)
            nn.init.zeros_(self.fusion_controller.weight)
            nn.init.constant_(
                self.fusion_controller.bias,
                math.log(math.expm1(config.fusion_lambda_init)),
            )
        else:
            self.fusion_controller = None
            self.register_buffer(
                "fixed_fusion_lambda", torch.tensor(config.fusion_fixed_lambda)
            )
        self.positional_encoding = PositionalEncoding(config.d_model)
        self.temporal_attention = TAttention(
            config.d_model, config.temporal_heads, config.dropout
        )
        self.spatial_attention = SAttention(
            config.d_model, config.spatial_heads, config.dropout
        )
        self.temporal_aggregation = TemporalAttention(config.d_model)
        self.decoder = nn.Linear(config.d_model, 1)

    def _routing(self):
        config = self.config
        if config.market_source_mode == "daily_only":
            return (
                self.alpha_routing_logits.new_tensor([1.0, 0.0]),
                self.intraday_routing_logits.new_tensor([1.0, 0.0]),
            )
        if config.market_source_mode == "intraday_only":
            return (
                self.alpha_routing_logits.new_tensor([0.0, 1.0]),
                self.intraday_routing_logits.new_tensor([0.0, 1.0]),
            )
        if config.routing_mode == "matched":
            return (
                self.alpha_routing_logits.new_tensor([1.0, 0.0]),
                self.intraday_routing_logits.new_tensor([0.0, 1.0]),
            )
        return (
            torch.softmax(self.alpha_routing_logits, dim=0),
            torch.softmax(self.intraday_routing_logits, dim=0),
        )

    def forward(
        self,
        stock_features: torch.Tensor,
        daily_market_features: torch.Tensor,
        intraday_market_features: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        if tuple(stock_features.shape[1:]) != (config.lookback, config.stock_feature_dim):
            raise ValueError("dual-frequency stock feature shape changed")
        if tuple(daily_market_features.shape) != (config.lookback, config.daily_market_dim):
            raise ValueError("dual-frequency daily market shape changed")
        if tuple(intraday_market_features.shape) != (
            config.lookback,
            config.intraday_market_dim,
        ):
            raise ValueError("dual-frequency intraday market shape changed")
        alpha = stock_features[:, :, : config.alpha_feature_dim]
        intraday = stock_features[:, :, config.alpha_feature_dim :]
        daily_market = daily_market_features[-1:].reshape(1, -1)
        minute_market = intraday_market_features[-1:].reshape(1, -1)
        route_alpha, route_intraday = self._routing()
        if config.feature_gate_mode == "uniform":
            alpha_gate = alpha.new_ones((stock_features.shape[0], config.alpha_feature_dim))
            intraday_gate = intraday.new_ones(
                (stock_features.shape[0], config.intraday_feature_dim)
            )
        else:
            alpha_logits = (
                route_alpha[0] * self.daily_to_alpha(daily_market)
                + route_alpha[1] * self.intraday_to_alpha(minute_market)
            )
            intraday_logits = (
                route_intraday[0] * self.daily_to_intraday(daily_market)
                + route_intraday[1] * self.intraday_to_intraday(minute_market)
            )
            alpha_gate = (
                config.alpha_feature_dim
                * torch.softmax(alpha_logits / config.daily_gate_beta, dim=-1)
            ).expand(stock_features.shape[0], -1)
            intraday_gate = (
                config.intraday_feature_dim
                * torch.softmax(intraday_logits / config.minute_gate_beta, dim=-1)
            ).expand(stock_features.shape[0], -1)
        daily_hidden = self.daily_projection(alpha * alpha_gate.unsqueeze(1))
        intraday_hidden = self.intraday_projection(
            intraday * intraday_gate.unsqueeze(1)
        )
        if config.fusion_mode == "learned_market":
            fusion_lambda = torch.nn.functional.softplus(
                self.fusion_controller(torch.cat((daily_market, minute_market), dim=-1))
            )
        else:
            fusion_lambda = self.fixed_fusion_lambda.reshape(1, 1)
        if config.branch_output_mode == "daily_only":
            hidden = daily_hidden
        elif config.branch_output_mode == "intraday_only":
            hidden = intraday_hidden
        else:
            hidden = daily_hidden + fusion_lambda.view(1, 1, 1) * intraday_hidden
        hidden = self.positional_encoding(hidden)
        hidden = self.temporal_attention(hidden)
        hidden = self.spatial_attention(hidden)
        hidden = self.temporal_aggregation(hidden)
        return self.decoder(hidden).squeeze(-1)

    def config_dict(self) -> Dict[str, Any]:
        return self.config.to_dict()
