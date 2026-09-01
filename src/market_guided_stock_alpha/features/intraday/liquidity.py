"""F029-F040: volume, amount, liquidity, and VWAP structure."""

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_field_share,
    cached_field_total,
    cached_nonnegative_field,
    cached_vwap,
    nanmean,
    nanquantile,
    nanstd,
    safe_divide,
    safe_entropy,
    safe_hhi,
    safe_log,
)


def _gini(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (values >= 0)
    count = valid.sum(axis=0).astype("float64")
    ordered = np.sort(np.where(valid, values, np.inf), axis=0)
    maximum_count = int(count.max()) if count.size else 0
    ranks = np.arange(1, maximum_count + 1, dtype="float64")[:, None]
    finite_ordered = ordered[:maximum_count]
    usable = np.arange(maximum_count)[:, None] < count[None, :]
    weighted = np.where(usable, finite_ordered * ranks, 0.0).sum(axis=0)
    total = np.where(valid, values, 0.0).sum(axis=0)
    result = safe_divide(2.0 * weighted, count * total) - safe_divide(count + 1.0, count)
    result[(count == 0) | (total == 0)] = np.nan
    return result


def _top_share(values: np.ndarray, fraction: float) -> np.ndarray:
    valid = np.isfinite(values) & (values >= 0)
    count = valid.sum(axis=0)
    ordered = np.sort(np.where(valid, values, -np.inf), axis=0)[::-1]
    take_count = np.ceil(fraction * count).astype("int64")
    maximum = int(take_count.max()) if take_count.size else 0
    selected = np.zeros(values.shape[1], dtype="float64")
    for rank in range(maximum):
        use = rank < take_count
        selected[use] += ordered[rank, use]
    total = np.where(valid, values, 0.0).sum(axis=0)
    return safe_divide(selected, total)


def compute(context: DayContext):
    volume = cached_nonnegative_field(context, "volume")
    amt = cached_nonnegative_field(context, "amt")
    returns = context.returns
    absolute = context.abs_returns
    valid_volume = np.isfinite(volume)
    valid_amt = np.isfinite(amt)
    volume_count = valid_volume.sum(axis=0)
    volume_total = cached_field_total(context, "volume")
    amt_total = cached_field_total(context, "amt")
    volume_share = cached_field_share(context, "volume")
    amount_share = cached_field_share(context, "amt")
    entropy = safe_entropy(volume, axis=0, normalized=True)

    illiq_valid = np.isfinite(absolute) & np.isfinite(amt) & (amt > 0)
    illiq = np.full(returns.shape, np.nan, dtype="float64")
    illiq[illiq_valid] = absolute[illiq_valid] / amt[illiq_valid]

    vwap = cached_vwap(context)
    minute_vwap = vwap["minute"]
    day_vwap = vwap["day"]
    vwap_valid = np.isfinite(minute_vwap)
    relative_vwap = np.full(volume.shape, np.nan, dtype="float64")
    valid_relative = vwap_valid & (minute_vwap > 0) & (day_vwap[None, :] > 0)
    relative_vwap[valid_relative] = safe_log(safe_divide(
        minute_vwap[valid_relative],
        np.broadcast_to(day_vwap, minute_vwap.shape)[valid_relative],
    ))

    return {
        "volume_hhi": safe_hhi(volume),
        "volume_entropy": entropy,
        "volume_gini": _gini(volume),
        "amount_hhi": safe_hhi(amt),
        "max_minute_volume_share": nanquantile(volume_share, 1.0),
        "top5pct_volume_share": _top_share(volume, 0.05),
        "zero_volume_ratio": safe_divide(
            (valid_volume & (volume == 0)).sum(axis=0), volume_count
        ),
        "volume_cv": safe_divide(nanstd(volume, ddof=1), nanmean(volume)),
        "illiq_mean": nanmean(illiq),
        "illiq_p90": nanquantile(illiq, 0.90),
        "vwap_path_absdev": nanmean(np.abs(relative_vwap)),
        "minute_vwap_dispersion": nanstd(relative_vwap, ddof=1),
    }
