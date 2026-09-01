"""Canonical neutral sequence dataset and model-input layout contracts."""

from market_guided_stock_alpha.dataset.alignment import (
    AlignmentError,
    ArtifactLookup,
)
from market_guided_stock_alpha.dataset.dataset import (
    SequenceDataset,
    SequenceSample,
    last_day_tabular_view,
)
from market_guided_stock_alpha.dataset.layout import (
    LOOKBACK_TRADING_DAYS,
    FeatureLayout,
    feature_names,
)
from market_guided_stock_alpha.dataset.normalization import (
    DatasetStatistics,
    FeatureStatistics,
    fit_dataset_statistics,
    fit_robust_statistics,
    normalize_cross_section_labels,
)
from market_guided_stock_alpha.dataset.splits import (
    DatasetSplits,
    build_split_contract,
)

__all__ = [
    "AlignmentError",
    "ArtifactLookup",
    "DatasetSplits",
    "DatasetStatistics",
    "FeatureLayout",
    "FeatureStatistics",
    "LOOKBACK_TRADING_DAYS",
    "SequenceDataset",
    "SequenceSample",
    "build_split_contract",
    "feature_names",
    "fit_dataset_statistics",
    "fit_robust_statistics",
    "last_day_tabular_view",
    "normalize_cross_section_labels",
]
