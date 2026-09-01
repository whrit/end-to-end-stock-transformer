"""Small warm/cold initialization contract."""

from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from market_guided_stock_alpha.training.checkpoint import CheckpointManager


class InitializationMode(str, Enum):
    WARM = "warm"
    COLD = "cold"


def initialize_target_model(
    model: torch.nn.Module,
    mode: InitializationMode,
    checkpoint_manager: CheckpointManager,
    previous_best_checkpoint: Optional[Path],
) -> str:
    if mode == InitializationMode.WARM and previous_best_checkpoint is not None:
        checkpoint_manager.load_weights_only(previous_best_checkpoint, model)
        return "previous_target_best_weights_only"
    if mode == InitializationMode.COLD or previous_best_checkpoint is None:
        return "new_model_initialization"
    raise ValueError("unsupported initialization mode")
