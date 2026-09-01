"""Small pandas/NumPy rolling operators with fixed Alpha158 semantics."""

from typing import Callable, Tuple

import numpy as np
import pandas as pd


def sanitize_non_finite(series: pd.Series) -> pd.Series:
    """Return a float copy with infinities replaced by NaN and an audit count."""
    result = pd.to_numeric(series, errors="coerce").astype("float64").copy()
    non_finite = result.notna() & ~np.isfinite(result.to_numpy())
    count = int(non_finite.sum())
    if count:
        result.loc[non_finite] = np.nan
    result.attrs["non_finite_to_nan_count"] = count
    return result


def _rolling(
    series: pd.Series, window: int, method: str, *args, **kwargs
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    roller = values.rolling(window=window, min_periods=1)
    result = getattr(roller, method)(*args, **kwargs)
    return sanitize_non_finite(result)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return _rolling(series, window, "mean")


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return _rolling(series, window, "sum")


def rolling_std(series: pd.Series, window: int, ddof: int = 1) -> pd.Series:
    return _rolling(series, window, "std", ddof=ddof)


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    return _rolling(series, window, "min")


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return _rolling(series, window, "max")


def rolling_quantile(
    series: pd.Series, window: int, quantile: float
) -> pd.Series:
    return _rolling(series, window, "quantile", quantile)


def _apply_last(
    series: pd.Series,
    window: int,
    function: Callable[[np.ndarray], float],
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    result = values.rolling(window=window, min_periods=1).apply(function, raw=True)
    return sanitize_non_finite(result)


def rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    def rank_last(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values[-1]):
            return np.nan
        ranks = pd.Series(values).rank(method="average", pct=True)
        return float(ranks.iloc[-1])

    return _apply_last(series, window, rank_last)


def rolling_idxmax(series: pd.Series, window: int) -> pd.Series:
    def idxmax(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values).all():
            return np.nan
        return float(np.nanargmax(values) + 1)

    return _apply_last(series, window, idxmax)


def rolling_idxmin(series: pd.Series, window: int) -> pd.Series:
    def idxmin(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values).all():
            return np.nan
        return float(np.nanargmin(values) + 1)

    return _apply_last(series, window, idxmin)


def rolling_corr(
    left: pd.Series, right: pd.Series, window: int
) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce").astype("float64")
    right_values = pd.to_numeric(right, errors="coerce").astype("float64")
    result = left_values.rolling(window=window, min_periods=1).corr(right_values)
    return sanitize_non_finite(result)


def _ols_stat(
    values: np.ndarray, statistic: str, constant_tolerance: float
) -> float:
    positions = np.arange(1, len(values) + 1, dtype="float64")
    valid = np.isfinite(values)
    if int(valid.sum()) < 2:
        return np.nan
    x = positions[valid]
    y = values[valid]
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        return np.nan
    slope = float(np.dot(x_centered, y_centered) / denominator)
    if statistic == "slope":
        return slope
    if statistic == "rsquare":
        if float(np.std(y, ddof=1)) <= constant_tolerance:
            return np.nan
        correlation = float(np.corrcoef(x, y)[0, 1])
        return correlation * correlation if np.isfinite(correlation) else np.nan
    if statistic == "residual":
        if not np.isfinite(values[-1]):
            return np.nan
        intercept = float(y.mean() - slope * x.mean())
        fitted_current = intercept + slope * float(len(values))
        return float(values[-1] - fitted_current)
    raise ValueError("Unknown rolling OLS statistic: {}".format(statistic))


def _rolling_ols_stat(
    series: pd.Series,
    window: int,
    statistic: str,
    constant_tolerance: float = 2.0e-5,
) -> pd.Series:
    return _apply_last(
        series,
        window,
        lambda values: _ols_stat(values, statistic, constant_tolerance),
    )


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    return _rolling_ols_stat(series, window, "slope")


def rolling_rsquare(series: pd.Series, window: int) -> pd.Series:
    return _rolling_ols_stat(series, window, "rsquare")


def rolling_residual(series: pd.Series, window: int) -> pd.Series:
    return _rolling_ols_stat(series, window, "residual")


def rolling_ols(
    series: pd.Series, window: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return slope, R-squared, and current residual with identical indices."""
    return (
        rolling_slope(series, window),
        rolling_rsquare(series, window),
        rolling_residual(series, window),
    )
