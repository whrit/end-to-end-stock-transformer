"""Architecture-independent optimizer and training configuration."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Mapping

from market_guided_stock_alpha.training.losses import LossConfig


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    gradient_clip_value: float = 3.0

    def __post_init__(self) -> None:
        for name in ("learning_rate", "weight_decay", "gradient_clip_value"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError("{} must be finite".format(name))
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_value <= 0.0:
            raise ValueError("gradient_clip_value must be positive")


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 8
    minimum_epochs: int = 10

    def __post_init__(self) -> None:
        if type(self.patience) is not int or self.patience <= 0:
            raise ValueError("early-stopping patience must be positive")
        if type(self.minimum_epochs) is not int or self.minimum_epochs <= 0:
            raise ValueError("minimum_epochs must be positive")


@dataclass(frozen=True)
class LightGBMTrainingConfig:
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
        }
    )
    num_boost_round: int = 500
    early_stopping_rounds: int = 100
    minimum_valid_labels: int = 2

    def __post_init__(self) -> None:
        for name in ("num_boost_round", "early_stopping_rounds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if type(self.minimum_valid_labels) is not int or self.minimum_valid_labels < 2:
            raise ValueError("minimum_valid_labels must be at least two")
        if not self.parameters:
            raise ValueError("LightGBM parameters cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    seed: int = 42
    deterministic: bool = True
    minimum_valid_labels: int = 2
    checkpoint_every: int = 1
    primary_metric: str = "mean_rank_ic"

    def __post_init__(self) -> None:
        for name in ("epochs", "minimum_valid_labels", "checkpoint_every"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.minimum_valid_labels < 2:
            raise ValueError("minimum_valid_labels must be at least two")
        if self.early_stopping.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if self.primary_metric not in ("mean_rank_ic", "validation_loss"):
            raise ValueError("primary_metric must be mean_rank_ic or validation_loss")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_optimizer(
    model: Any, config: OptimizerConfig
) -> Any:
    import torch

    return torch.optim.Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
