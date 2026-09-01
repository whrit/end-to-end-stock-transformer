"""F149-F160: minute extremes and normalized event positions."""

from typing import Tuple

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_nonnegative_field,
    safe_divide,
    safe_nanmax,
    safe_nanmin,
)


def event_value_and_position(
    values: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the extreme and first normalized row position, without stock loops."""
    array = np.asarray(values, dtype="float64")
    valid = np.isfinite(array)
    has_value = valid.any(axis=0)
    if mode == "max":
        filled = np.where(valid, array, -np.inf)
        extreme = safe_nanmax(array, axis=0)
    elif mode == "min":
        filled = np.where(valid, array, np.inf)
        extreme = safe_nanmin(array, axis=0)
    elif mode == "max_abs":
        filled = np.where(valid, np.abs(array), -np.inf)
        extreme = safe_nanmax(np.abs(array), axis=0)
    else:
        raise ValueError("event mode must be max, min, or max_abs")
    row = np.argmax(filled if mode != "min" else -filled, axis=0)
    denominator = max(array.shape[0] - 1, 1)
    position = row.astype("float64") / float(denominator)
    position[~has_value] = np.nan
    return extreme, position


def draw_event_positions(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(close) & (close > 0)
    values = np.where(valid, close, np.nan)
    running_max = np.fmax.accumulate(values, axis=0)
    running_min = np.fmin.accumulate(values, axis=0)
    drawdown = safe_divide(values, running_max) - 1.0
    drawup = safe_divide(values, running_min) - 1.0
    _minimum, drawdown_position = event_value_and_position(drawdown, "min")
    _maximum, drawup_position = event_value_and_position(drawup, "max")
    return drawdown_position, drawup_position


def compute(context: DayContext):
    active = ~context.structural_inactive
    returns = context.returns[active]
    volume = cached_nonnegative_field(context, "volume")[active]
    amt = cached_nonnegative_field(context, "amt")[active]
    high = context.fields["high"][active]
    low = context.fields["low"][active]
    close = context.fields["close"][active]
    max_return, max_return_position = event_value_and_position(returns, "max")
    min_return, min_return_position = event_value_and_position(returns, "min")
    max_abs_return, max_abs_position = event_value_and_position(returns, "max_abs")
    _max_volume, max_volume_position = event_value_and_position(volume, "max")
    _max_amount, max_amount_position = event_value_and_position(amt, "max")
    _high, high_position = event_value_and_position(high, "max")
    _low, low_position = event_value_and_position(low, "min")
    drawdown_position, drawup_position = draw_event_positions(close)
    return {
        "max_positive_minute_return": max_return,
        "max_negative_minute_return": min_return,
        "max_abs_minute_return": max_abs_return,
        "max_positive_return_position": max_return_position,
        "max_negative_return_position": min_return_position,
        "max_abs_return_position": max_abs_position,
        "max_volume_position": max_volume_position,
        "max_amount_position": max_amount_position,
        "intraday_high_position": high_position,
        "intraday_low_position": low_position,
        "max_drawdown_end_position": drawdown_position,
        "max_drawup_end_position": drawup_position,
    }
