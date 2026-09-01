"""F177-F212 and F241-F248: cached multiscale and local volatility shape."""

from typing import Dict, Tuple

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_nonnegative_field,
    cached_session_returns,
    cached_session_sum,
    finite_sum,
    safe_divide,
    safe_entropy,
    safe_hhi,
    safe_log,
    safe_nanmax,
    safe_nanmean,
    safe_nanquantile,
    safe_std,
)
from market_guided_stock_alpha.features.intraday.distribution import (
    corrected_excess_kurtosis,
    corrected_skew,
    sign_change_ratio,
)


def trend_statistics(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS cumulative path on normalized valid positions, grouped by valid count."""
    stocks = returns.shape[1]
    slope = np.full(stocks, np.nan, dtype="float64")
    r2 = np.full(stocks, np.nan, dtype="float64")
    residual_std = np.full(stocks, np.nan, dtype="float64")
    counts = np.isfinite(returns).sum(axis=0)
    for count in np.unique(counts[counts >= 2]):
        columns = np.flatnonzero(counts == count)
        selected = returns[:, columns]
        packed = selected.T[np.isfinite(selected).T].reshape(len(columns), int(count)).T
        y = np.cumsum(packed, axis=0)
        x = np.linspace(0.0, 1.0, int(count), dtype="float64")
        x_centered = x - x.mean()
        y_centered = y - y.mean(axis=0, keepdims=True)
        denominator = float(np.sum(x_centered * x_centered))
        beta = safe_divide((x_centered[:, None] * y_centered).sum(axis=0), denominator)
        fitted = y.mean(axis=0, keepdims=True) + beta[None, :] * x_centered[:, None]
        residual = y - fitted
        total_ss = (y_centered * y_centered).sum(axis=0)
        residual_ss = (residual * residual).sum(axis=0)
        slope[columns] = beta
        r2[columns] = 1.0 - safe_divide(residual_ss, total_ss)
        residual_std[columns] = np.sqrt(
            safe_divide(residual_ss, float(int(count)))
        )
        r2[columns[total_ss <= 0]] = np.nan
    return slope, r2, residual_std


def _top_squared_share(values: np.ndarray, fraction: float) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    squared = np.where(valid, values * values, -np.inf)
    ordered = np.sort(squared, axis=0)[::-1]
    take_count = np.ceil(fraction * count).astype("int64")
    selected = np.zeros(values.shape[1], dtype="float64")
    for rank in range(int(take_count.max()) if take_count.size else 0):
        use = rank < take_count
        selected[use] += ordered[rank, use]
    total = np.where(valid, values * values, 0.0).sum(axis=0)
    return safe_divide(selected, total)


def _multiscale(context: DayContext) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for size in (5, 15, 30):
        returns = cached_session_returns(context, size)
        volume = cached_session_sum(context, "volume", size)
        rv_weights = returns * returns
        result["ret_skew_{}m".format(size)] = corrected_skew(returns)
        result["ret_kurt_{}m".format(size)] = corrected_excess_kurtosis(returns)
        result["sign_change_ratio_{}m".format(size)] = sign_change_ratio(returns)
        result["rv_hhi_{}m".format(size)] = safe_hhi(rv_weights)
        result["rv_entropy_{}m".format(size)] = safe_entropy(rv_weights)
        result["volume_hhi_{}m".format(size)] = safe_hhi(volume)
        if size in (5, 15):
            result["volume_entropy_{}m".format(size)] = safe_entropy(volume)
    return result


def _full_day(context: DayContext) -> Dict[str, np.ndarray]:
    returns = context.returns
    valid = np.isfinite(returns)
    squared = np.where(valid, returns * returns, np.nan)
    rv = np.where(valid, returns * returns, 0.0).sum(axis=0)
    absolute = context.abs_returns
    median_return = safe_nanquantile(returns, 0.50)
    quartiles = safe_nanquantile(returns, np.asarray([0.25, 0.75]))
    high = context.fields["high"]
    low = context.fields["low"]
    valid_range = (
        np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0)
    )
    log_range = safe_log(safe_divide(
        np.where(valid_range, high, np.nan),
        np.where(valid_range, low, np.nan),
    ))
    slope, r2, residual_std = trend_statistics(returns)
    block_returns = {size: cached_session_returns(context, size) for size in (5, 15, 30)}
    return {
        "rv_hhi_1m": safe_hhi(squared),
        "rv_entropy_1m": safe_entropy(squared),
        "max_minute_rv_share": safe_divide(safe_nanmax(squared), rv),
        "top1pct_rv_share": _top_squared_share(returns, 0.01),
        "median_absret": safe_nanquantile(absolute, 0.50),
        "ret_mad": safe_nanquantile(
            np.abs(returns - median_return[None, :]), 0.50
        ),
        "ret_iqr": quartiles[1] - quartiles[0],
        "mean_minute_log_range": safe_nanmean(log_range),
        "log_range_std": safe_std(log_range, ddof=1),
        "range_hhi": safe_hhi(log_range * log_range),
        "intraday_trend_slope": slope,
        "intraday_trend_r2": r2,
        "detrended_path_std": residual_std,
        "max_5m_abs_return": safe_nanmax(np.abs(block_returns[5])),
        "max_15m_abs_return": safe_nanmax(np.abs(block_returns[15])),
        "max_30m_abs_return": safe_nanmax(np.abs(block_returns[30])),
    }


def _position_blocks(context: DayContext) -> Dict[str, np.ndarray]:
    volume = cached_nonnegative_field(context, "volume")
    result: Dict[str, np.ndarray] = {}
    for block in range(4):
        window = slice(block * 60, (block + 1) * 60)
        returns = context.returns[window]
        block_volume = volume[window]
        valid_returns = np.isfinite(returns)
        rv = np.where(valid_returns, returns * returns, 0.0).sum(axis=0)
        downside = np.where(
            valid_returns & (returns < 0), returns * returns, 0.0
        ).sum(axis=0)
        valid_pair = valid_returns & np.isfinite(block_volume)
        signed = np.where(
            valid_pair, np.sign(returns) * block_volume, 0.0
        ).sum(axis=0)
        prefix = "block{}".format(block + 1)
        result[prefix + "_downside_rv_share"] = safe_divide(downside, rv)
        result[prefix + "_signed_volume_imbalance"] = safe_divide(
            signed, finite_sum(block_volume)
        )
    return result


def compute(context: DayContext):
    result = _multiscale(context)
    result.update(_full_day(context))
    result.update(_position_blocks(context))
    return result
