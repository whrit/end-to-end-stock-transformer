"""Pure trading-calendar split and t+6 purge contracts."""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from market_guided_stock_alpha.data.calendar import TradingCalendar
from market_guided_stock_alpha.labels.relative_excess_return import (
    PURGE_DEPENDENCY_OFFSET_TRADING_DAYS,
)


@dataclass(frozen=True)
class DatasetSplits:
    train_dates: pd.DatetimeIndex
    valid_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purged_train_dates: pd.DatetimeIndex
    purged_valid_dates: pd.DatetimeIndex
    purge_dependency: int = PURGE_DEPENDENCY_OFFSET_TRADING_DAYS


def build_split_contract(
    calendar: TradingCalendar,
    train_start: Any,
    train_end: Any,
    valid_start: Any,
    valid_end: Any,
    test_start: Optional[Any] = None,
    test_end: Optional[Any] = None,
) -> DatasetSplits:
    """Resolve open-date segments and remove anchors whose t+6 crosses a boundary."""
    train_requested = calendar.load_open_dates(train_start, train_end)
    valid_requested = calendar.load_open_dates(valid_start, valid_end)
    test = (
        calendar.load_open_dates(test_start, test_end)
        if test_start is not None and test_end is not None
        else pd.DatetimeIndex([])
    )
    if train_requested.empty or valid_requested.empty:
        raise ValueError("train and valid splits must contain trading dates")
    if train_requested[-1] >= valid_requested[0]:
        raise ValueError("train and valid splits overlap or are unordered")
    if len(test) and valid_requested[-1] >= test[0]:
        raise ValueError("valid and test splits overlap or are unordered")

    dependency = PURGE_DEPENDENCY_OFFSET_TRADING_DAYS
    train_dependencies = pd.DatetimeIndex(
        [calendar.next_trading_date(day, dependency) for day in train_requested]
    )
    train_keep = train_dependencies < valid_requested[0]
    train = train_requested[train_keep]
    purged_train = train_requested[~train_keep]
    if train.empty:
        raise ValueError("t+6 purge removed every training date")

    valid = valid_requested
    purged_valid = pd.DatetimeIndex([])
    if len(test):
        valid_dependencies = pd.DatetimeIndex(
            [calendar.next_trading_date(day, dependency) for day in valid_requested]
        )
        valid_keep = valid_dependencies < test[0]
        valid = valid_requested[valid_keep]
        purged_valid = valid_requested[~valid_keep]
        if valid.empty:
            raise ValueError("t+6 purge removed every validation date")
    return DatasetSplits(
        train,
        valid,
        test,
        purged_train,
        purged_valid,
        purge_dependency=dependency,
    )
