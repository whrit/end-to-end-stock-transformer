"""F213-F228: amount lead/lag, daily VWAP, and signed impact."""

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_field_total,
    cached_nonnegative_field,
    cached_vwap,
    finite_sum,
    pairwise_corr,
    safe_divide,
    safe_log,
    safe_nanmax,
    safe_nanmin,
)


def first_last_valid(values: np.ndarray):
    array = np.asarray(values, dtype="float64")
    valid = np.isfinite(array) & (array > 0)
    has_value = valid.any(axis=0)
    columns = np.arange(array.shape[1])
    first_position = np.argmax(valid, axis=0)
    last_position = array.shape[0] - 1 - np.argmax(valid[::-1], axis=0)
    first = np.full(array.shape[1], np.nan, dtype="float64")
    last = np.full(array.shape[1], np.nan, dtype="float64")
    first[has_value] = array[first_position[has_value], columns[has_value]]
    last[has_value] = array[last_position[has_value], columns[has_value]]
    return first, last


def compute(context: DayContext):
    active = ~context.structural_inactive
    returns = context.returns[active]
    absolute = context.abs_returns[active]
    volume = cached_nonnegative_field(context, "volume")[active]
    amt = cached_nonnegative_field(context, "amt")[active]
    log_amt = np.where(np.isfinite(amt), np.log1p(amt), np.nan)
    min_obs = context.config.min_corr_obs

    vwap = cached_vwap(context)
    day_vwap = vwap["day"]
    minute_vwap = vwap["minute"][active]
    valid_vwap = (
        np.isfinite(minute_vwap) & (minute_vwap > 0)
        & np.isfinite(day_vwap)[None, :] & (day_vwap[None, :] > 0)
    )
    deviation = np.abs(safe_log(safe_divide(
        np.where(valid_vwap, minute_vwap, np.nan), day_vwap[None, :]
    )))

    first_open, _last_open = first_last_valid(context.fields["open"][active])
    _first_close, last_close = first_last_valid(context.fields["close"][active])
    day_high = safe_nanmax(context.fields["high"][active])
    day_low = safe_nanmin(context.fields["low"][active])

    valid_volume_return = np.isfinite(volume) & np.isfinite(returns)
    valid_amt_return = np.isfinite(amt) & np.isfinite(returns)
    volume_denominator = cached_field_total(context, "volume")
    amt_denominator = cached_field_total(context, "amt")
    volume_weighted = np.where(valid_volume_return, volume * returns, 0.0).sum(axis=0)
    amount_weighted = np.where(valid_amt_return, amt * returns, 0.0).sum(axis=0)
    signed_volume_impact = np.where(
        valid_volume_return,
        np.sign(returns) * volume * np.abs(returns),
        0.0,
    ).sum(axis=0)
    signed_amount_impact = np.where(
        valid_amt_return,
        np.sign(returns) * amt * np.abs(returns),
        0.0,
    ).sum(axis=0)
    return {
        "corr_ret_amount": pairwise_corr(returns, log_amt, min_obs),
        "corr_absret_amount": pairwise_corr(absolute, log_amt, min_obs),
        "amount_lead_ret_corr": pairwise_corr(log_amt[:-1], returns[1:], min_obs),
        "amount_lead_absret_corr": pairwise_corr(log_amt[:-1], absolute[1:], min_obs),
        "ret_lead_amount_corr": pairwise_corr(returns[:-1], log_amt[1:], min_obs),
        "absret_lead_amount_corr": pairwise_corr(absolute[:-1], log_amt[1:], min_obs),
        "close_vs_day_vwap": safe_log(safe_divide(last_close, day_vwap)),
        "open_vs_day_vwap": safe_log(safe_divide(first_open, day_vwap)),
        "day_vwap_location": safe_divide(day_vwap - day_low, day_high - day_low),
        "volume_weighted_return": safe_divide(volume_weighted, volume_denominator),
        "amount_weighted_return": safe_divide(amount_weighted, amt_denominator),
        "corr_absret_vwap_deviation": pairwise_corr(absolute, deviation, min_obs),
        "signed_volume_impact": safe_divide(signed_volume_impact, volume_denominator),
        "signed_amount_impact": safe_divide(signed_amount_impact, amt_denominator),
        "positive_return_volume_share": safe_divide(
            np.where(valid_volume_return & (returns > 0), volume, 0.0).sum(axis=0),
            volume_denominator,
        ),
        "negative_return_volume_share": safe_divide(
            np.where(valid_volume_return & (returns < 0), volume, 0.0).sum(axis=0),
            volume_denominator,
        ),
    }
