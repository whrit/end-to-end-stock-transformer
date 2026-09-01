"""F053-F068 and F101-F148 position, block, and multiscale factors."""

from typing import Dict, Sequence, Tuple

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_field_total,
    cached_nonnegative_field,
    cached_session_returns,
    finite_sum,
    lag_corr,
    path_efficiency_from_returns,
    safe_divide,
    safe_log,
)
from market_guided_stock_alpha.features.intraday.geometry import cached_geometry


def _range(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    valid = np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0)
    maximum = np.where(valid, high, -np.inf).max(axis=0)
    minimum = np.where(valid, low, np.inf).min(axis=0)
    output = np.full(high.shape[1], np.nan, dtype="float64")
    usable = (valid.sum(axis=0) > 0) & (maximum > 0) & (minimum > 0)
    output[usable] = safe_log(safe_divide(maximum[usable], minimum[usable]))
    return output


def _window_families(
    context: DayContext, windows: Sequence[Tuple[str, slice]]
) -> Dict[str, np.ndarray]:
    returns = context.returns
    volume = cached_nonnegative_field(context, "volume")
    high = context.fields["high"]
    low = context.fields["low"]
    rv = np.where(np.isfinite(returns), returns ** 2, 0.0).sum(axis=0)
    rv[np.isfinite(returns).sum(axis=0) == 0] = np.nan
    volume_total = cached_field_total(context, "volume")
    result: Dict[str, np.ndarray] = {}
    for prefix, window in windows:
        window_returns = returns[window]
        window_volume = volume[window]
        result[prefix + "_ret"] = finite_sum(window_returns)
        result[prefix + "_rv_share"] = safe_divide(
            finite_sum(window_returns ** 2), rv
        )
        result[prefix + "_volume_share"] = safe_divide(
            finite_sum(window_volume), volume_total
        )
        result[prefix + "_range"] = _range(high[window], low[window])
    return result


def _block_factors(context: DayContext):
    returns = context.returns
    volume = cached_nonnegative_field(context, "volume")
    amt = cached_nonnegative_field(context, "amt")
    high = context.fields["high"]
    low = context.fields["low"]
    full_rv = finite_sum(returns ** 2)
    full_volume = cached_field_total(context, "volume")
    full_amt = cached_field_total(context, "amt")
    result: Dict[str, np.ndarray] = {}
    cache = []
    for block in range(4):
        window = slice(block * 60, (block + 1) * 60)
        block_returns = returns[window]
        block_rv = finite_sum(block_returns ** 2)
        block_volume = finite_sum(volume[window])
        geometry = cached_geometry(context, "block{}".format(block + 1), block_returns)
        prefix = "block{}".format(block + 1)
        values = {
            prefix + "_ret": finite_sum(block_returns),
            prefix + "_rv_share": safe_divide(block_rv, full_rv),
            prefix + "_volume_share": safe_divide(block_volume, full_volume),
            prefix + "_amount_share": safe_divide(finite_sum(amt[window]), full_amt),
            prefix + "_range": _range(high[window], low[window]),
            prefix + "_path_efficiency": geometry["path_efficiency"],
            prefix + "_abs_chord_area": geometry["abs_chord_area"],
            prefix + "_curvature_energy": geometry["curvature_energy"],
        }
        result.update(values)
        cache.append(
            {
                "ret": values[prefix + "_ret"],
                "rv": block_rv,
                "volume_share": values[prefix + "_volume_share"],
            }
        )
    return result, cache


def _multiscale(context: DayContext):
    returns = context.returns
    full_rv = finite_sum(returns ** 2)
    block_returns = {
        size: cached_session_returns(context, size)
        for size in (5, 15, 30)
    }
    result = {
        "rv_ratio_30m_1m": safe_divide(
            finite_sum(block_returns[30] ** 2), full_rv
        )
    }
    for size in (5, 15, 30):
        values = block_returns[size]
        result["ret_acf1_{}m".format(size)] = lag_corr(
            values, 1, context.config.min_multiscale_corr_obs
        )
        result["absret_acf1_{}m".format(size)] = lag_corr(
            np.abs(values), 1, context.config.min_multiscale_corr_obs
        )
        result["path_efficiency_{}m".format(size)] = path_efficiency_from_returns(
            values
        )
    return result


def _cross_session(context: DayContext, blocks):
    eps = context.config.eps
    first = blocks[0]
    second = blocks[1]
    third = blocks[2]
    last = blocks[3]
    first_last_logratio = np.full(context.stock_count, np.nan, dtype="float64")
    usable = np.isfinite(first["rv"]) & np.isfinite(last["rv"])
    nondegenerate = usable & ~((first["rv"] == 0) & (last["rv"] == 0))
    first_last_logratio[nondegenerate] = safe_log(safe_divide(
        first["rv"][nondegenerate] + eps, last["rv"][nondegenerate] + eps
    ))
    morning_rv = first["rv"] + second["rv"]
    afternoon_rv = third["rv"] + last["rv"]
    session_logratio = np.full(context.stock_count, np.nan, dtype="float64")
    session_usable = np.isfinite(morning_rv) & np.isfinite(afternoon_rv)
    session_nondegenerate = session_usable & ~(
        (morning_rv == 0) & (afternoon_rv == 0)
    )
    session_logratio[session_nondegenerate] = safe_log(safe_divide(
        morning_rv[session_nondegenerate] + eps,
        afternoon_rv[session_nondegenerate] + eps,
    ))
    return {
        "first60_last60_ret_diff": first["ret"] - last["ret"],
        "first60_last60_rv_logratio": first_last_logratio,
        "first60_last60_volume_share_diff": (
            first["volume_share"] - last["volume_share"]
        ),
        "morning_afternoon_ret_diff": (
            first["ret"] + second["ret"] - third["ret"] - last["ret"]
        ),
        "morning_afternoon_rv_logratio": session_logratio,
        "morning_afternoon_volume_share_diff": (
            first["volume_share"] + second["volume_share"]
            - third["volume_share"] - last["volume_share"]
        ),
    }


def compute(context: DayContext):
    morning = context.config.morning_bars
    afternoon = context.config.afternoon_bars
    if morning < 30 or afternoon < 30:
        raise ValueError("each session must contain at least 30 positions")
    windows = (
        ("open30", slice(0, 30)),
        ("midmorning", slice(30, morning)),
        ("midafternoon", slice(morning, morning + afternoon - 30)),
        ("close30", slice(morning + afternoon - 30, morning + afternoon)),
    )
    result = _window_families(context, windows)
    block_factors, blocks = _block_factors(context)
    result.update(block_factors)
    result.update(_multiscale(context))
    result.update(_cross_session(context, blocks))
    return result
