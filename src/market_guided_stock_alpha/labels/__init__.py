"""Label construction, loading, and normalization interfaces."""

from market_guided_stock_alpha.labels.derived_label import DerivedLabelReader
from market_guided_stock_alpha.labels.forward_return import ForwardReturnLabelGenerator
from market_guided_stock_alpha.labels.relative_excess_return import (
    DEFAULT_BENCHMARK_CODE,
    DEFAULT_VALUE_COLUMN,
    HOLDING_PERIOD_TRADING_DAYS,
    LABEL_END_OFFSET_TRADING_DAYS,
    LABEL_START_OFFSET_TRADING_DAYS,
    PURGE_DEPENDENCY_OFFSET_TRADING_DAYS,
    build_relative_excess_label,
    formula_metadata,
)

__all__ = [
    "DEFAULT_BENCHMARK_CODE",
    "DEFAULT_VALUE_COLUMN",
    "DerivedLabelReader",
    "ForwardReturnLabelGenerator",
    "HOLDING_PERIOD_TRADING_DAYS",
    "LABEL_END_OFFSET_TRADING_DAYS",
    "LABEL_START_OFFSET_TRADING_DAYS",
    "PURGE_DEPENDENCY_OFFSET_TRADING_DAYS",
    "build_relative_excess_label",
    "formula_metadata",
]
