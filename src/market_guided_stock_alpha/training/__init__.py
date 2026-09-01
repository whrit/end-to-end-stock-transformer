"""Canonical whole-date neural training core."""

from market_guided_stock_alpha.training.checkpoint import (
    CheckpointContract,
    CheckpointManager,
    feature_layout_identity,
    feature_order_hash,
)
from market_guided_stock_alpha.training.config import (
    EarlyStoppingConfig,
    LightGBMTrainingConfig,
    OptimizerConfig,
    TrainingConfig,
    create_optimizer,
)
from market_guided_stock_alpha.training.losses import (
    LOSS_NAMES,
    LossConfig,
    MaskedCCCLoss,
    MaskedListFoldLoss,
    MaskedMSELoss,
    MaskedRPLLoss,
    MaskedRSRLoss,
    build_loss,
)
from market_guided_stock_alpha.training.preprocessing import (
    LABEL_NORMALIZATION_CONTRACT,
    load_preprocessing,
    preprocessing_from_dict,
    preprocessing_to_dict,
    save_preprocessing,
)
from market_guided_stock_alpha.training.seed import resolve_device, seed_everything
from market_guided_stock_alpha.training.trainer import Trainer

__all__ = [
    "CheckpointContract",
    "CheckpointManager",
    "EarlyStoppingConfig",
    "LABEL_NORMALIZATION_CONTRACT",
    "LightGBMTrainingConfig",
    "LOSS_NAMES",
    "LossConfig",
    "MaskedCCCLoss",
    "MaskedListFoldLoss",
    "MaskedMSELoss",
    "MaskedRPLLoss",
    "MaskedRSRLoss",
    "OptimizerConfig",
    "Trainer",
    "TrainingConfig",
    "build_loss",
    "create_optimizer",
    "feature_layout_identity",
    "feature_order_hash",
    "load_preprocessing",
    "preprocessing_from_dict",
    "preprocessing_to_dict",
    "resolve_device",
    "save_preprocessing",
    "seed_everything",
]
