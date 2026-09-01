"""F001-F012: minute-return distribution and tails."""

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    nanmean,
    nanquantile,
    safe_divide,
)


def _corrected_skew(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0).astype("float64")
    mean = nanmean(values, axis=0)
    centered = np.where(valid, values - mean[None, :], 0.0)
    sum2 = (centered ** 2).sum(axis=0)
    sample_std = np.sqrt(safe_divide(sum2, count - 1.0))
    standardized3 = safe_divide((centered ** 3).sum(axis=0), sample_std ** 3)
    result = safe_divide(count * standardized3, (count - 1.0) * (count - 2.0))
    result[(count < 3) | (sample_std == 0)] = np.nan
    return result


corrected_skew = _corrected_skew


def _corrected_excess_kurtosis(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0).astype("float64")
    mean = nanmean(values, axis=0)
    centered = np.where(valid, values - mean[None, :], 0.0)
    moment2 = safe_divide((centered ** 2).sum(axis=0), count)
    moment4 = safe_divide((centered ** 4).sum(axis=0), count)
    biased_excess = safe_divide(moment4, moment2 ** 2) - 3.0
    result = safe_divide(
        (count - 1.0) * ((count + 1.0) * biased_excess + 6.0),
        (count - 2.0) * (count - 3.0),
    )
    result[(count < 4) | (moment2 == 0)] = np.nan
    return result


corrected_excess_kurtosis = _corrected_excess_kurtosis


def sign_change_ratio(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    adjacent = valid[1:] & valid[:-1]
    nonzero = adjacent & (values[1:] != 0) & (values[:-1] != 0)
    changes = nonzero & ((values[1:] * values[:-1]) < 0)
    return safe_divide(changes.sum(axis=0), nonzero.sum(axis=0))


def _longest_run_ratio(values: np.ndarray, positive: bool) -> np.ndarray:
    valid_count = np.isfinite(values).sum(axis=0)
    current = np.zeros(values.shape[1], dtype="int32")
    longest = np.zeros(values.shape[1], dtype="int32")
    condition = values > 0 if positive else values < 0
    condition &= np.isfinite(values)
    for row in condition:
        current = np.where(row, current + 1, 0)
        longest = np.maximum(longest, current)
    return safe_divide(longest, valid_count)


def _exact_top_squared_share(values: np.ndarray, fraction: float) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    squared = np.where(valid, values ** 2, -np.inf)
    ordered = np.sort(squared, axis=0)[::-1]
    take_count = np.ceil(fraction * count).astype("int64")
    maximum = int(take_count.max()) if take_count.size else 0
    selected = np.zeros(values.shape[1], dtype="float64")
    for rank in range(maximum):
        use = rank < take_count
        selected[use] += ordered[rank, use]
    total = np.where(valid, values ** 2, 0.0).sum(axis=0)
    return safe_divide(selected, total)


def compute(context: DayContext):
    returns = context.returns[~context.structural_inactive]
    absolute = context.abs_returns[~context.structural_inactive]
    valid = np.isfinite(returns)
    count = valid.sum(axis=0)
    positive_ratio = safe_divide(((returns > 0) & valid).sum(axis=0), count)
    zero_ratio = safe_divide(
        ((absolute <= context.config.zero_return_tolerance) & valid).sum(axis=0),
        count,
    )
    sign_change = sign_change_ratio(returns)
    upper = nanquantile(returns, 0.95)
    lower = np.abs(nanquantile(returns, 0.05))
    return {
        "ret_skew": _corrected_skew(returns),
        "ret_kurt": _corrected_excess_kurtosis(returns),
        "positive_ret_ratio": positive_ratio,
        "zero_ret_ratio": zero_ratio,
        "sign_change_ratio": sign_change,
        "longest_up_run_ratio": _longest_run_ratio(returns, True),
        "longest_down_run_ratio": _longest_run_ratio(returns, False),
        "absret_q90": nanquantile(absolute, 0.90),
        "absret_q95": nanquantile(absolute, 0.95),
        "absret_q99": nanquantile(absolute, 0.99),
        "tail_asymmetry": safe_divide(upper - lower, upper + lower),
        "extreme_rv_share": _exact_top_squared_share(returns, 0.05),
    }
