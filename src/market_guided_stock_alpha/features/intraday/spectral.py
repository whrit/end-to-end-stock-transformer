"""F069-F083 plus cached warning-safe FFT intermediates."""

from typing import Dict, Tuple

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_nonnegative_field,
    safe_divide,
    safe_log,
)


FFT_KEYS = (
    "low_share", "high_share", "entropy", "centroid", "dominant_period",
    "low_high_logratio", "spectral_slope",
)


def fft_extended_summaries(
    values: np.ndarray, min_obs: int, chunk_size: int, eps: float
) -> Dict[str, np.ndarray]:
    stocks = values.shape[1]
    output = {key: np.full(stocks, np.nan, dtype="float64") for key in FFT_KEYS}
    for chunk_start in range(0, stocks, chunk_size):
        chunk_end = min(stocks, chunk_start + chunk_size)
        chunk = values[:, chunk_start:chunk_end]
        counts = np.isfinite(chunk).sum(axis=0)
        for count in np.unique(counts[counts >= min_obs]):
            local_columns = np.flatnonzero(counts == count)
            selected = chunk[:, local_columns]
            mask = np.isfinite(selected)
            packed = selected.T[mask.T].reshape(len(local_columns), int(count)).T
            centered = packed - packed.mean(axis=0, keepdims=True)
            power = np.abs(np.fft.rfft(centered, axis=0)[1:]) ** 2
            if power.shape[0] == 0:
                continue
            total_power = power.sum(axis=0)
            usable = np.isfinite(total_power) & (total_power > eps)
            normalized = safe_divide(power, total_power[None, :])
            frequency_index = np.arange(1, power.shape[0] + 1, dtype="float64")
            periods = float(count) / frequency_index
            frequencies = frequency_index / float(count)
            low_mask = periods >= 60.0
            high_mask = periods < 15.0
            low_power = power[low_mask].sum(axis=0)
            high_power = power[high_mask].sum(axis=0)
            low = np.where(np.isfinite(normalized[low_mask]), normalized[low_mask], 0.0).sum(axis=0)
            high = np.where(np.isfinite(normalized[high_mask]), normalized[high_mask], 0.0).sum(axis=0)
            positive = np.isfinite(normalized) & (normalized > 0)
            entropy_numerator = -np.where(
                positive,
                normalized * safe_log(np.where(positive, normalized, np.nan)),
                0.0,
            ).sum(axis=0)
            entropy = safe_divide(entropy_numerator, safe_log(power.shape[0]))
            centroid = np.where(
                np.isfinite(normalized), normalized * frequencies[:, None], 0.0
            ).sum(axis=0)
            dominant_period = periods[np.argmax(power, axis=0)]

            logratio = np.full(len(local_columns), np.nan, dtype="float64")
            if np.any(low_mask) and np.any(high_mask):
                ratio_usable = usable & np.isfinite(low_power) & np.isfinite(high_power)
                logratio[ratio_usable] = safe_log(
                    safe_divide(
                        low_power[ratio_usable] + eps,
                        high_power[ratio_usable] + eps,
                    )
                )

            slope = np.full(len(local_columns), np.nan, dtype="float64")
            if power.shape[0] >= 3:
                x = safe_log(frequencies)
                x_centered = x - x.mean()
                x_denominator = float(np.sum(x_centered * x_centered))
                y = safe_log(power + eps)
                y_centered = y - y.mean(axis=0, keepdims=True)
                slope_values = safe_divide(
                    (x_centered[:, None] * y_centered).sum(axis=0),
                    x_denominator,
                )
                slope[usable] = slope_values[usable]

            metrics = (low, high, entropy, centroid, dominant_period, logratio, slope)
            targets = chunk_start + local_columns
            for key, metric in zip(FFT_KEYS, metrics):
                cleaned = np.asarray(metric, dtype="float64").copy()
                cleaned[~usable] = np.nan
                output[key][targets] = cleaned
    return output


def fft_summaries(
    values: np.ndarray, min_obs: int, chunk_size: int, eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metrics = fft_extended_summaries(values, min_obs, chunk_size, eps)
    return tuple(metrics[key] for key in FFT_KEYS[:5])


def cached_fft(
    context: DayContext, cache_name: str, values: np.ndarray
) -> Dict[str, np.ndarray]:
    key = "fft:{}".format(cache_name)
    if key not in context.cache:
        context.cache[key] = fft_extended_summaries(
            values,
            context.config.min_fft_obs,
            context.config.stock_chunk_size,
            context.config.eps,
        )
    return context.cache[key]


def compute(context: DayContext):
    volume = cached_nonnegative_field(context, "volume")
    log_volume = np.where(np.isfinite(volume), np.log1p(volume), np.nan)
    ret = cached_fft(context, "ret", context.returns)
    absret = cached_fft(context, "absret", context.abs_returns)
    vol = cached_fft(context, "volume", log_volume)
    result = {}
    for prefix, metrics in (("ret", ret), ("absret", absret), ("volume", vol)):
        for suffix in FFT_KEYS[:5]:
            result["{}_fft_{}".format(prefix, suffix)] = metrics[suffix]
    return result
