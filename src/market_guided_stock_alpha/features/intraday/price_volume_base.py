"""F041-F052: price-volume correlation and lead-lag structure."""

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_field_total,
    cached_nonnegative_field,
    nanmean,
    nanquantile,
    pairwise_corr,
    safe_divide,
)


def compute(context: DayContext):
    active_rows = ~context.structural_inactive
    returns = context.returns[active_rows]
    absolute = context.abs_returns[active_rows]
    volume = cached_nonnegative_field(context, "volume")[active_rows]
    amt = cached_nonnegative_field(context, "amt")[active_rows]
    log_volume = np.where(np.isfinite(volume), np.log1p(volume), np.nan)
    delta_volume = np.full(log_volume.shape, np.nan, dtype="float64")
    valid_delta = np.isfinite(log_volume[1:]) & np.isfinite(log_volume[:-1])
    delta_volume[1:] = np.where(
        valid_delta, log_volume[1:] - log_volume[:-1], np.nan
    )
    min_obs = context.config.min_corr_obs

    volume_total = cached_field_total(context, "volume")
    amt_total = cached_field_total(context, "amt")
    signed_volume = np.where(
        np.isfinite(returns) & np.isfinite(volume), np.sign(returns) * volume, 0.0
    ).sum(axis=0)
    signed_amt = np.where(
        np.isfinite(returns) & np.isfinite(amt), np.sign(returns) * amt, 0.0
    ).sum(axis=0)

    thresholds_high = nanquantile(volume, 0.80)
    thresholds_low = nanquantile(volume, 0.20)
    valid_pair = np.isfinite(volume) & np.isfinite(returns)
    high = valid_pair & (volume >= thresholds_high[None, :])
    low = valid_pair & (volume <= thresholds_low[None, :])
    high_returns = np.where(high, returns, np.nan)
    low_returns = np.where(low, returns, np.nan)
    return {
        "corr_ret_volume": pairwise_corr(returns, log_volume, min_obs),
        "corr_absret_volume": pairwise_corr(absolute, log_volume, min_obs),
        "corr_ret_dvolume": pairwise_corr(returns, delta_volume, min_obs),
        "corr_absret_dvolume": pairwise_corr(absolute, delta_volume, min_obs),
        "volume_lead_ret_corr": pairwise_corr(log_volume[:-1], returns[1:], min_obs),
        "volume_lead_absret_corr": pairwise_corr(log_volume[:-1], absolute[1:], min_obs),
        "ret_lead_volume_corr": pairwise_corr(returns[:-1], log_volume[1:], min_obs),
        "absret_lead_volume_corr": pairwise_corr(absolute[:-1], log_volume[1:], min_obs),
        "signed_volume_imbalance": safe_divide(signed_volume, volume_total),
        "signed_amount_imbalance": safe_divide(signed_amt, amt_total),
        "high_volume_ret": nanmean(high_returns),
        "high_low_volume_ret_diff": nanmean(high_returns) - nanmean(low_returns),
    }
