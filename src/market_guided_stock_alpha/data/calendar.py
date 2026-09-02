"""DataFrame-backed unified trading-calendar operations."""

from typing import Any

import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    normalize_date_values,
    validate_calendar,
    validate_date_range,
)


class TradingCalendar:
    """Expose strict open-date operations over a canonical calendar DataFrame."""

    def __init__(self, calendar: pd.DataFrame) -> None:
        self.frame = validate_calendar(calendar)
        self.open_dates = pd.DatetimeIndex(
            self.frame.loc[self.frame["is_open"].eq(1), "calendar_date"]
        )
        if not self.open_dates.is_monotonic_increasing:
            raise DataQualityError("open trading calendar must be increasing")

    def load_open_dates(self, start: Any, end: Any) -> pd.DatetimeIndex:
        start_date, end_date = validate_date_range(start, end)
        return self.open_dates[
            (self.open_dates >= start_date) & (self.open_dates <= end_date)
        ]

    def is_trading_day(self, value: Any) -> bool:
        day = pd.Timestamp(normalize_date_values([value], "date").iloc[0])
        return bool(day in self.open_dates)

    def _offset(self, value: Any, n: int, forward: bool) -> pd.Timestamp:
        if type(n) is not int or n <= 0:
            raise ValueError("n must be a positive integer")
        day = pd.Timestamp(normalize_date_values([value], "date").iloc[0])
        candidates = (
            self.open_dates[self.open_dates > day]
            if forward
            else self.open_dates[self.open_dates < day][::-1]
        )
        if len(candidates) < n:
            raise ValueError("calendar does not cover the requested offset")
        return pd.Timestamp(candidates[n - 1])

    def next_trading_date(self, value: Any, n: int = 1) -> pd.Timestamp:
        return self._offset(value, n, True)

    def previous_trading_date(self, value: Any, n: int = 1) -> pd.Timestamp:
        return self._offset(value, n, False)
