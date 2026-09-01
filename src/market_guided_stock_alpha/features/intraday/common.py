"""Shared contracts and warning-safe helpers for intraday price-volume features."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


CORE_FIELDS = ("open", "high", "low", "close", "volume", "amt")


@dataclass(frozen=True)
class IntradayPriceVolumeConfig:
    family: str
    factor_count: int
    expected_bars_per_day: int
    morning_bars: int
    afternoon_bars: int
    min_bar_coverage: float
    structural_zero_threshold: float
    eps: float
    zero_return_tolerance: float
    min_corr_obs: int
    min_multiscale_corr_obs: int
    min_fft_obs: int
    min_geometry_obs: int
    stock_chunk_size: int
    output_dtype: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IntradayPriceVolumeConfig":
        minimum = value.get("minimum_observations", {})
        session = value.get("session_contract", {})
        result = cls(
            family=str(value["name"]),
            factor_count=int(value["factor_count"]),
            expected_bars_per_day=int(session["bars_per_day"]),
            morning_bars=int(session["morning_bars"]),
            afternoon_bars=int(session["afternoon_bars"]),
            min_bar_coverage=float(value["min_bar_coverage"]),
            structural_zero_threshold=float(value["structural_zero_threshold"]),
            eps=float(value["epsilon"]),
            zero_return_tolerance=float(value["zero_return_tolerance"]),
            min_corr_obs=int(minimum["correlation"]),
            min_multiscale_corr_obs=int(minimum["multiscale_correlation"]),
            min_fft_obs=int(minimum["fft"]),
            min_geometry_obs=int(minimum["geometry"]),
            stock_chunk_size=int(value["stock_chunk_size"]),
            output_dtype=str(value["output_dtype"]),
        )
        if result.family != "intraday_price_volume":
            raise ValueError("intraday price-volume family name changed")
        if result.factor_count != 248:
            raise ValueError("intraday price-volume factor_count must be 248")
        if (
            result.expected_bars_per_day != 240
            or result.morning_bars != 120
            or result.afternoon_bars != 120
        ):
            raise ValueError("intraday price-volume features require a 120+120=240 bar session")
        if not 0 < result.min_bar_coverage <= 1:
            raise ValueError("min_bar_coverage must be in (0,1]")
        if not 0 <= result.structural_zero_threshold <= 1:
            raise ValueError("structural_zero_threshold must be in [0,1]")
        if result.eps <= 0 or result.stock_chunk_size <= 0:
            raise ValueError("epsilon and stock_chunk_size must be positive")
        if (
            result.min_corr_obs < 2
            or result.min_multiscale_corr_obs < 2
            or result.min_fft_obs < 3
            or result.min_geometry_obs < 3
        ):
            raise ValueError("minimum observation thresholds are too small")
        if result.output_dtype != "float32":
            raise ValueError("intraday price-volume output_dtype must be float32")
        return result


@dataclass
class DayContext:
    fields: Dict[str, np.ndarray]
    returns: np.ndarray
    abs_returns: np.ndarray
    structural_inactive: np.ndarray
    structural_zero_share: np.ndarray
    bar_coverage: np.ndarray
    eligible: np.ndarray
    config: IntradayPriceVolumeConfig
    cache: Dict[str, Any] = field(default_factory=dict)

    @property
    def stock_count(self) -> int:
        return self.returns.shape[1]

    @property
    def morning_slice(self) -> slice:
        return slice(0, self.config.morning_bars)

    @property
    def afternoon_slice(self) -> slice:
        return slice(self.config.morning_bars, self.config.expected_bars_per_day)


def as_float64_matrix(value: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype="float64")
    if array.ndim != 2:
        raise ValueError("{} day input must be a 2D matrix".format(field))
    return array


def prepare_day(
    raw_fields: Mapping[str, Any], config: IntradayPriceVolumeConfig
) -> DayContext:
    if set(raw_fields) != set(CORE_FIELDS):
        raise ValueError("day fields must equal {}".format(",".join(CORE_FIELDS)))
    raw = {field: as_float64_matrix(raw_fields[field], field) for field in CORE_FIELDS}
    shapes = {array.shape for array in raw.values()}
    if len(shapes) != 1:
        raise ValueError("six intraday day matrices are misaligned")
    bars, stocks = next(iter(shapes))
    if bars != config.expected_bars_per_day or stocks == 0:
        raise ValueError(
            "day matrix must have {} bars and at least one stock".format(
                config.expected_bars_per_day
            )
        )

    volume = raw["volume"]
    amt = raw["amt"]
    valid_va = np.isfinite(volume) & np.isfinite(amt)
    valid_count = valid_va.sum(axis=1)
    joint_zero = valid_va & (volume == 0.0) & (amt == 0.0)
    zero_share = safe_divide(joint_zero.sum(axis=1), valid_count)
    structural = np.isfinite(zero_share) & (
        zero_share >= config.structural_zero_threshold
    )
    active_rows = ~structural

    fields = {
        field: np.where(active_rows[:, None] & np.isfinite(array), array, np.nan)
        for field, array in raw.items()
    }
    close_coverage = np.isfinite(raw["close"]).sum(axis=0) / float(bars)
    eligible = close_coverage >= config.min_bar_coverage

    returns = np.full((bars, stocks), np.nan, dtype="float64")
    last_close = np.full(stocks, np.nan, dtype="float64")
    seen_close = np.zeros(stocks, dtype=bool)
    for position in range(bars):
        if structural[position]:
            continue
        current = raw["close"][position]
        opening = raw["open"][position]
        current_valid = np.isfinite(current) & (current > 0)
        first_valid = current_valid & ~seen_close & np.isfinite(opening) & (opening > 0)
        continuing = current_valid & seen_close & np.isfinite(last_close) & (last_close > 0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            returns[position, first_valid] = np.log(
                current[first_valid] / opening[first_valid]
            )
            returns[position, continuing] = np.log(
                current[continuing] / last_close[continuing]
            )
        last_close[current_valid] = current[current_valid]
        seen_close[current_valid] = True
    returns[~np.isfinite(returns)] = np.nan
    return DayContext(
        fields=fields,
        returns=returns,
        abs_returns=np.abs(returns),
        structural_inactive=structural,
        structural_zero_share=zero_share,
        bar_coverage=close_coverage.astype("float64", copy=False),
        eligible=eligible,
        config=config,
    )


def safe_divide(numerator: Any, denominator: Any) -> np.ndarray:
    left, right = np.broadcast_arrays(
        np.asarray(numerator, dtype="float64"),
        np.asarray(denominator, dtype="float64"),
    )
    output = np.full(left.shape, np.nan, dtype="float64")
    valid = np.isfinite(left) & np.isfinite(right) & (right != 0.0)
    np.divide(left, right, out=output, where=valid)
    output[~np.isfinite(output)] = np.nan
    return output


def finite_sum(values: np.ndarray, axis: int = 0) -> np.ndarray:
    valid = np.isfinite(values)
    result = np.where(valid, values, 0.0).sum(axis=axis, dtype="float64")
    result = np.asarray(result, dtype="float64")
    result[np.sum(valid, axis=axis) == 0] = np.nan
    return result


def safe_nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return safe_divide(
        np.where(np.isfinite(values), values, 0.0).sum(axis=axis),
        np.isfinite(values).sum(axis=axis),
    )


def safe_std(values: np.ndarray, axis: int = 0, ddof: int = 1) -> np.ndarray:
    count = np.isfinite(values).sum(axis=axis)
    mean = safe_nanmean(values, axis=axis)
    expanded = np.expand_dims(mean, axis=axis)
    squared = np.where(np.isfinite(values), (values - expanded) ** 2, 0.0)
    variance = safe_divide(squared.sum(axis=axis), count - ddof)
    variance = np.where(variance >= 0, variance, np.nan)
    with np.errstate(invalid="ignore"):
        return np.sqrt(variance)


def safe_nanmin(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    valid = np.isfinite(array)
    result = np.min(np.where(valid, array, np.inf), axis=axis)
    return np.where(valid.sum(axis=axis) > 0, result, np.nan).astype("float64")


def safe_nanmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    valid = np.isfinite(array)
    result = np.max(np.where(valid, array, -np.inf), axis=axis)
    return np.where(valid.sum(axis=axis) > 0, result, np.nan).astype("float64")


def safe_nanquantile(
    values: np.ndarray, quantile: Any, axis: int = 0
) -> np.ndarray:
    """Quantile without calling NumPy on an all-NaN slice."""
    array = np.asarray(values, dtype="float64")
    if array.ndim != 2 or axis != 0:
        raise ValueError("safe_nanquantile currently requires a 2D matrix and axis=0")
    q = np.asarray(quantile, dtype="float64")
    output_shape = (() if q.ndim == 0 else q.shape) + (array.shape[1],)
    output = np.full(output_shape, np.nan, dtype="float64")
    usable = np.isfinite(array).sum(axis=0) > 0
    if np.any(usable):
        output[..., usable] = np.nanquantile(array[:, usable], q, axis=0)
    return output


def safe_log(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    output = np.full(array.shape, np.nan, dtype="float64")
    usable = np.isfinite(array) & (array > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.log(array, out=output, where=usable)
    return np.where(np.isfinite(output), output, np.nan).astype("float64")


def safe_entropy(
    values: np.ndarray, axis: int = 0, normalized: bool = True
) -> np.ndarray:
    """Entropy of non-negative weights; degenerate support returns NaN."""
    array = np.asarray(values, dtype="float64")
    valid = np.isfinite(array) & (array >= 0)
    total = np.where(valid, array, 0.0).sum(axis=axis)
    expanded_total = np.expand_dims(total, axis=axis)
    shares = safe_divide(np.where(valid, array, 0.0), expanded_total)
    positive = np.isfinite(shares) & (shares > 0)
    logs = safe_log(np.where(positive, shares, np.nan))
    entropy = -np.where(positive, shares * logs, 0.0).sum(axis=axis)
    support = valid.sum(axis=axis)
    entropy = np.where((total > 0) & (support > 1), entropy, np.nan)
    if normalized:
        entropy = safe_divide(entropy, safe_log(support))
    return entropy


def safe_hhi(values: np.ndarray, axis: int = 0) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    valid = np.isfinite(array) & (array >= 0)
    total = np.where(valid, array, 0.0).sum(axis=axis)
    shares = safe_divide(
        np.where(valid, array, 0.0), np.expand_dims(total, axis=axis)
    )
    result = np.where(np.isfinite(shares), shares * shares, 0.0).sum(axis=axis)
    return np.where(total > 0, result, np.nan)


# Concise aliases used by the frozen base-factor modules.
nanmean = safe_nanmean
nanstd = safe_std
nanquantile = safe_nanquantile


def pairwise_corr(
    left: np.ndarray,
    right: np.ndarray,
    min_obs: int,
) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError("correlation matrices must have equal shape")
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=0).astype("float64")
    x = np.where(valid, left, 0.0)
    y = np.where(valid, right, 0.0)
    sum_x = x.sum(axis=0)
    sum_y = y.sum(axis=0)
    covariance = count * (x * y).sum(axis=0) - sum_x * sum_y
    variance_x = count * (x * x).sum(axis=0) - sum_x * sum_x
    variance_y = count * (y * y).sum(axis=0) - sum_y * sum_y
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    result = safe_divide(covariance, denominator)
    result[(count < min_obs) | (denominator == 0)] = np.nan
    return result


safe_corr = pairwise_corr


def lag_corr(values: np.ndarray, lag: int, min_obs: int) -> np.ndarray:
    if lag <= 0 or len(values) <= lag:
        return np.full(values.shape[1], np.nan, dtype="float64")
    return pairwise_corr(values[lag:], values[:-lag], min_obs)


def aggregate_session_returns(
    returns: np.ndarray, morning_bars: int, block_size: int
) -> np.ndarray:
    parts = []
    for session in (returns[:morning_bars], returns[morning_bars:]):
        if len(session) % block_size:
            raise ValueError("session bars must divide aggregation block size")
        blocks = session.reshape(
            len(session) // block_size, block_size, session.shape[1]
        )
        valid_count = np.isfinite(blocks).sum(axis=1)
        summed = np.where(np.isfinite(blocks), blocks, 0.0).sum(axis=1)
        summed[valid_count == 0] = np.nan
        parts.append(summed)
    return np.concatenate(parts, axis=0)


def aggregate_session_sum(
    values: np.ndarray, morning_bars: int, block_size: int
) -> np.ndarray:
    return aggregate_session_returns(values, morning_bars, block_size)


def cached_nonnegative_field(context: DayContext, name: str) -> np.ndarray:
    key = "nonnegative:{}".format(name)
    if key not in context.cache:
        values = context.fields[name]
        context.cache[key] = np.where(values >= 0, values, np.nan)
    return context.cache[key]


def cached_field_total(context: DayContext, name: str) -> np.ndarray:
    key = "field_total:{}".format(name)
    if key not in context.cache:
        context.cache[key] = finite_sum(cached_nonnegative_field(context, name))
    return context.cache[key]


def cached_field_share(context: DayContext, name: str) -> np.ndarray:
    key = "field_share:{}".format(name)
    if key not in context.cache:
        context.cache[key] = safe_divide(
            cached_nonnegative_field(context, name),
            cached_field_total(context, name)[None, :],
        )
    return context.cache[key]


def cached_vwap(context: DayContext) -> Dict[str, np.ndarray]:
    key = "vwap"
    if key not in context.cache:
        volume = cached_nonnegative_field(context, "volume")
        amt = cached_nonnegative_field(context, "amt")
        joint = np.isfinite(volume) & np.isfinite(amt)
        day = safe_divide(
            np.where(joint, amt, 0.0).sum(axis=0),
            np.where(joint, volume, 0.0).sum(axis=0),
        )
        minute = safe_divide(amt, volume)
        minute[~(
            joint & (volume > 0) & (amt > 0)
        )] = np.nan
        context.cache[key] = {"minute": minute, "day": day, "joint": joint}
    return context.cache[key]


def cached_session_returns(context: DayContext, block_size: int) -> np.ndarray:
    key = "session_returns:{}".format(int(block_size))
    if key not in context.cache:
        context.cache[key] = aggregate_session_returns(
            context.returns, context.config.morning_bars, int(block_size)
        )
    return context.cache[key]


def cached_session_sum(
    context: DayContext, field_name: str, block_size: int
) -> np.ndarray:
    key = "session_sum:{}:{}".format(field_name, int(block_size))
    if key not in context.cache:
        context.cache[key] = aggregate_session_sum(
            cached_nonnegative_field(context, field_name),
            context.config.morning_bars,
            int(block_size),
        )
    return context.cache[key]


def path_efficiency_from_returns(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    has_value = count > 0
    first_position = np.argmax(valid, axis=0)
    first_value = np.zeros(values.shape[1], dtype="float64")
    columns = np.arange(values.shape[1])
    first_value[has_value] = values[first_position[has_value], columns[has_value]]
    total_after_first = np.where(valid, values, 0.0).sum(axis=0) - first_value
    distance_after_first = (
        np.where(valid, np.abs(values), 0.0).sum(axis=0) - np.abs(first_value)
    )
    result = safe_divide(np.abs(total_after_first), distance_after_first)
    result[count < 2] = np.nan
    return result


def sanitize_factor_map(
    factors: Mapping[str, Any], stock_count: int
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for name, value in factors.items():
        array = np.asarray(value, dtype="float64")
        if array.shape != (stock_count,):
            raise ValueError("{} factor must have one value per stock".format(name))
        array = array.copy()
        array[~np.isfinite(array)] = np.nan
        result[name] = array
    return result
