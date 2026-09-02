"""Deterministic normalization into the canonical 120+120 minute session."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    MINUTE_FIELDS,
    normalize_security_code,
    validate_minute_index,
    validate_minute_stock,
)


MORNING_BARS = 120
AFTERNOON_BARS = 120
BARS_PER_DAY = 240
MIN_BAR_COVERAGE = 0.90


def expected_clock_times() -> Tuple[str, ...]:
    morning = pd.date_range("2000-01-01 09:31", "2000-01-01 11:30", freq="min")
    afternoon = pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="min")
    values = tuple(index.strftime("%H:%M") for index in morning.append(afternoon))
    if len(values) != BARS_PER_DAY:
        raise RuntimeError("canonical minute clock must contain 240 bars")
    return values


EXPECTED_CLOCK_TIMES = expected_clock_times()
EXPECTED_TIME_SET = set(EXPECTED_CLOCK_TIMES)


@dataclass(frozen=True)
class MinuteSession:
    trade_date: pd.Timestamp
    timestamps: pd.DatetimeIndex
    instruments: Tuple[str, ...]
    fields: Dict[str, np.ndarray]
    bar_coverage: np.ndarray

    @property
    def eligible(self) -> np.ndarray:
        return self.bar_coverage >= MIN_BAR_COVERAGE


def _expected_index(trade_date: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [trade_date + pd.to_timedelta(clock + ":00") for clock in EXPECTED_CLOCK_TIMES]
    )


def normalize_minute_session(
    frame: pd.DataFrame,
    code_column: str = "sec_code",
    trade_date: Optional[pd.Timestamp] = None,
    instruments: Optional[Sequence[str]] = None,
) -> MinuteSession:
    """Reindex observed rows to 240 slots with NaN only; never interpolate."""
    if code_column == "sec_code":
        canonical = validate_minute_stock(frame)
    elif code_column == "index_code":
        canonical = validate_minute_index(frame)
    else:
        raise ValueError("code_column must be sec_code or index_code")
    dates = pd.DatetimeIndex(canonical["timestamp"]).normalize().unique()
    if trade_date is None:
        if len(dates) != 1:
            raise DataQualityError("minute session input must contain exactly one date")
        day = pd.Timestamp(dates[0])
    else:
        day = pd.Timestamp(trade_date).tz_localize(None).normalize()
        canonical = canonical.loc[
            pd.DatetimeIndex(canonical["timestamp"]).normalize() == day
        ].copy()
    observed_times = set(canonical["timestamp"].dt.strftime("%H:%M"))
    unexpected = sorted(observed_times.difference(EXPECTED_TIME_SET))
    if unexpected:
        raise DataQualityError(
            "minute session contains out-of-session timestamps: {}".format(
                ",".join(unexpected[:5])
            )
        )
    if instruments is None:
        codes = tuple(sorted(canonical[code_column].unique()))
    else:
        codes = tuple(
            normalize_security_code(value, "minute session instrument")
            for value in instruments
        )
        if len(codes) != len(set(codes)):
            raise DataQualityError("minute session instruments contain duplicates")
    if not codes:
        raise DataQualityError("minute session requires at least one instrument")
    timestamps = _expected_index(day)
    full_index = pd.MultiIndex.from_product(
        [timestamps, codes], names=["timestamp", code_column]
    )
    keyed = canonical.set_index(["timestamp", code_column])
    fields = {
        field: keyed[field].reindex(full_index).to_numpy(dtype="float64").reshape(
            BARS_PER_DAY, len(codes)
        )
        for field in MINUTE_FIELDS
    }
    coverage = np.isfinite(fields["close"]).sum(axis=0) / float(BARS_PER_DAY)
    return MinuteSession(
        trade_date=day,
        timestamps=timestamps,
        instruments=codes,
        fields=fields,
        bar_coverage=coverage.astype("float64", copy=False),
    )


def normalize_wide_minute_day(
    frames: Mapping[str, pd.DataFrame],
    trade_date: pd.Timestamp,
    instruments: Optional[Sequence[str]] = None,
) -> MinuteSession:
    """Normalize six explicit wide timestamp-by-security frames for one day."""
    if set(frames) != set(MINUTE_FIELDS):
        raise ValueError("wide minute frames must contain exactly six fields")
    day = pd.Timestamp(trade_date).tz_localize(None).normalize()
    codes = (
        tuple(sorted(normalize_security_code(value) for value in frames["open"].columns))
        if instruments is None
        else tuple(normalize_security_code(value) for value in instruments)
    )
    rows = []
    for field, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("wide minute {} value must be DataFrame".format(field))
        if frame.index.has_duplicates or frame.columns.has_duplicates:
            raise DataQualityError("wide minute {} has duplicate axes".format(field))
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
        if index.hasnans:
            raise DataQualityError("wide minute {} has invalid timestamps".format(field))
        selected = frame.copy(deep=False)
        selected.index = index.tz_localize(None) if index.tz is not None else index
        selected.columns = [normalize_security_code(value) for value in selected.columns]
        selected = selected.loc[selected.index.normalize() == day]
        selected = selected.reindex(columns=codes)
        long_index = pd.MultiIndex.from_product(
            [selected.index, codes], names=["timestamp", "sec_code"]
        )
        stacked = pd.Series(
            selected.to_numpy().reshape(-1), index=long_index, name=field
        )
        rows.append(stacked)
    combined = pd.concat(rows, axis=1).reset_index()
    combined.columns = ["timestamp", "sec_code"] + list(MINUTE_FIELDS)
    return normalize_minute_session(
        combined, "sec_code", trade_date=day, instruments=codes
    )
