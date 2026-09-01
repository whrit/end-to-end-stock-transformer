"""Canonical rolling target, prediction, and experiment orchestration."""

from market_guided_stock_alpha.rolling.artifacts import (
    read_prediction_artifact,
    write_prediction_artifact,
)
from market_guided_stock_alpha.rolling.configuration import (
    load_experiment_config,
    run_from_json_config,
    validate_experiment_config,
)
from market_guided_stock_alpha.rolling.prediction import (
    predict_lightgbm_model,
    predict_neural_model,
)
from market_guided_stock_alpha.rolling.runner import (
    RollingDataBundle,
    RollingRunConfig,
    RollingRunner,
)
from market_guided_stock_alpha.rolling.schedule import (
    InitializationMode,
    initialize_target_model,
)
from market_guided_stock_alpha.rolling.targets import (
    ROLLING_FREQUENCIES,
    RollingTarget,
    generate_rolling_targets,
)

__all__ = [
    "InitializationMode",
    "ROLLING_FREQUENCIES",
    "RollingDataBundle",
    "RollingRunConfig",
    "RollingRunner",
    "RollingTarget",
    "generate_rolling_targets",
    "initialize_target_model",
    "predict_lightgbm_model",
    "predict_neural_model",
    "read_prediction_artifact",
    "load_experiment_config",
    "run_from_json_config",
    "validate_experiment_config",
    "write_prediction_artifact",
]
