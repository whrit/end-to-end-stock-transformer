"""Deterministic trading-calendar rolling target generation."""

from dataclasses import dataclass
from typing import Any, Tuple

import pandas as pd

from market_guided_stock_alpha.data.calendar import TradingCalendar
from market_guided_stock_alpha.dataset.splits import build_split_contract
from market_guided_stock_alpha.labels.relative_excess_return import (
    PURGE_DEPENDENCY_OFFSET_TRADING_DAYS,
)


ROLLING_FREQUENCIES = ("semiannual", "quarterly", "monthly")
VALIDATION_MONTHS = {"semiannual": 6, "quarterly": 3, "monthly": 3}


@dataclass(frozen=True)
class RollingTarget:
    target_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    frequency: str
    train_dates: pd.DatetimeIndex
    valid_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    purged_train_dates: pd.DatetimeIndex
    purged_valid_dates: pd.DatetimeIndex
    purge_dependency: int = PURGE_DEPENDENCY_OFFSET_TRADING_DAYS

    def to_dict(self, include_dates: bool = False) -> dict:
        result = {
            "target_id": self.target_id,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "valid_start": str(self.valid_start.date()),
            "valid_end": str(self.valid_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "frequency": self.frequency,
            "purge_dependency": self.purge_dependency,
            "train_trading_day_count": len(self.train_dates),
            "valid_trading_day_count": len(self.valid_dates),
            "test_trading_day_count": len(self.test_dates),
            "purged_train_day_count": len(self.purged_train_dates),
            "purged_valid_day_count": len(self.purged_valid_dates),
        }
        if include_dates:
            for name in ("train_dates", "valid_dates", "test_dates"):
                result[name] = [str(day.date()) for day in getattr(self, name)]
        return result


def _period_bounds(day: pd.Timestamp, frequency: str) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    if frequency == "monthly":
        period = day.to_period("M")
        return period.start_time.normalize(), period.end_time.normalize(), "{}M{:02d}".format(period.year, period.month)
    if frequency == "quarterly":
        period = day.to_period("Q-DEC")
        return period.start_time.normalize(), period.end_time.normalize(), "{}Q{}".format(period.year, period.quarter)
    half = 1 if day.month <= 6 else 2
    start = pd.Timestamp(day.year, 1 if half == 1 else 7, 1)
    end = pd.Timestamp(day.year, 6 if half == 1 else 12, 1) + pd.offsets.MonthEnd(0)
    return start, end.normalize(), "{}H{}".format(day.year, half)


def generate_rolling_targets(
    calendar: TradingCalendar,
    frequency: str,
    train_start: Any,
    test_start: Any,
    test_end: Any,
) -> Tuple[RollingTarget, ...]:
    """Generate expanding-window targets with actual open-date boundaries.

    Semiannual validation is the preceding six months.  Quarterly and monthly
    retain the historical formal implementation's preceding three-month
    validation window.  Every train/validation anchor whose t+6 dependency
    reaches the next segment is purged by ``build_split_contract``.
    """
    if frequency not in ROLLING_FREQUENCIES:
        raise ValueError("frequency must be semiannual, quarterly, or monthly")
    requested_start = pd.Timestamp(test_start).tz_localize(None).normalize()
    requested_end = pd.Timestamp(test_end).tz_localize(None).normalize()
    if requested_start > requested_end:
        raise ValueError("test_start must not follow test_end")
    test_open = calendar.load_open_dates(requested_start, requested_end)
    if test_open.empty:
        raise ValueError("requested test range contains no trading date")

    identities = []
    seen = set()
    for day in test_open:
        bounds = _period_bounds(pd.Timestamp(day), frequency)
        if bounds[2] not in seen:
            seen.add(bounds[2])
            identities.append(bounds)

    targets = []
    for natural_start, natural_end, base_id in identities:
        segment_start = max(natural_start, requested_start)
        segment_end = min(natural_end, requested_end)
        test_dates = calendar.load_open_dates(segment_start, segment_end)
        if test_dates.empty:
            continue
        validation_start = natural_start - pd.DateOffset(
            months=VALIDATION_MONTHS[frequency]
        )
        validation_end = natural_start - pd.Timedelta(days=1)
        requested_train_end = validation_start - pd.Timedelta(days=1)
        splits = build_split_contract(
            calendar,
            train_start,
            requested_train_end,
            validation_start,
            validation_end,
            test_dates[0],
            test_dates[-1],
        )
        partial = segment_start > natural_start or segment_end < natural_end
        target_id = base_id + ("_partial" if partial else "")
        targets.append(
            RollingTarget(
                target_id=target_id,
                train_start=pd.Timestamp(splits.train_dates[0]),
                train_end=pd.Timestamp(splits.train_dates[-1]),
                valid_start=pd.Timestamp(splits.valid_dates[0]),
                valid_end=pd.Timestamp(splits.valid_dates[-1]),
                test_start=pd.Timestamp(splits.test_dates[0]),
                test_end=pd.Timestamp(splits.test_dates[-1]),
                frequency=frequency,
                train_dates=splits.train_dates,
                valid_dates=splits.valid_dates,
                test_dates=splits.test_dates,
                purged_train_dates=splits.purged_train_dates,
                purged_valid_dates=splits.purged_valid_dates,
            )
        )
    if not targets:
        raise ValueError("rolling target generation produced no targets")
    return tuple(targets)
