"""F013-F028: realized volatility, range estimators, and dependence."""

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_session_returns,
    finite_sum,
    lag_corr,
    safe_divide,
    safe_log,
    safe_nanmax,
    safe_nanmin,
)


def _draw_measures(close: np.ndarray):
    valid = np.isfinite(close) & (close > 0)
    values = np.where(valid, close, np.nan)
    running_max = np.fmax.accumulate(values, axis=0)
    running_min = np.fmin.accumulate(values, axis=0)
    drawdown = safe_divide(values, running_max) - 1.0
    drawup = safe_divide(values, running_min) - 1.0
    max_drawdown = -safe_nanmin(drawdown, axis=0)
    max_drawup = safe_nanmax(drawup, axis=0)
    return max_drawdown, max_drawup


def compute(context: DayContext):
    returns = context.returns[~context.structural_inactive]
    absolute = context.abs_returns[~context.structural_inactive]
    valid_returns = np.isfinite(returns)
    count = valid_returns.sum(axis=0)
    rv = np.where(valid_returns, returns ** 2, 0.0).sum(axis=0)
    adjacent = valid_returns[1:] & valid_returns[:-1]
    bv = (np.pi / 2.0) * np.where(
        adjacent, absolute[1:] * absolute[:-1], 0.0
    ).sum(axis=0)
    bv[adjacent.sum(axis=0) == 0] = np.nan
    downside = np.where(valid_returns & (returns < 0), returns ** 2, 0.0).sum(axis=0)
    downside[count == 0] = np.nan
    quarticity = safe_divide(count, 3.0) * np.where(
        valid_returns, returns ** 4, 0.0
    ).sum(axis=0)
    quarticity[count == 0] = np.nan

    high = context.fields["high"]
    low = context.fields["low"]
    opening = context.fields["open"]
    close = context.fields["close"]
    valid_hl = np.isfinite(high) & np.isfinite(low) & (high > 0) & (low > 0)
    log_hl = safe_log(safe_divide(high, low))
    parkinson = np.where(
        valid_hl, (log_hl ** 2) / (4.0 * float(safe_log(2.0))), 0.0
    ).sum(axis=0)
    parkinson[valid_hl.sum(axis=0) == 0] = np.nan
    valid_ohlc = (
        valid_hl & np.isfinite(opening) & np.isfinite(close)
        & (opening > 0) & (close > 0)
    )
    safe_h = np.where(valid_ohlc, high, 1.0)
    safe_l = np.where(valid_ohlc, low, 1.0)
    safe_o = np.where(valid_ohlc, opening, 1.0)
    safe_c = np.where(valid_ohlc, close, 1.0)
    rs_rows = safe_log(safe_divide(safe_h, safe_o)) * safe_log(
        safe_divide(safe_h, safe_c)
    )
    rs_rows += safe_log(safe_divide(safe_l, safe_o)) * safe_log(
        safe_divide(safe_l, safe_c)
    )
    rs = np.where(valid_ohlc, rs_rows, 0.0).sum(axis=0)
    rs[valid_ohlc.sum(axis=0) == 0] = np.nan
    max_drawdown, max_drawup = _draw_measures(close)

    ret5 = cached_session_returns(context, 5)
    ret15 = cached_session_returns(context, 15)
    rv5 = finite_sum(ret5 ** 2, axis=0)
    rv15 = finite_sum(ret15 ** 2, axis=0)
    return {
        "realized_variance": np.where(count > 0, rv, np.nan),
        "bipower_variation": bv,
        "jump_share": safe_divide(np.maximum(rv - bv, 0.0), rv),
        "downside_rv": downside,
        "downside_rv_share": safe_divide(downside, rv),
        "realized_quarticity": quarticity,
        "intramin_parkinson_var": parkinson,
        "intramin_rs_var": rs,
        "intraday_max_drawdown": max_drawdown,
        "intraday_max_drawup": max_drawup,
        "ret_acf1": lag_corr(returns, 1, context.config.min_corr_obs),
        "ret_acf5": lag_corr(returns, 5, context.config.min_corr_obs),
        "absret_acf1": lag_corr(absolute, 1, context.config.min_corr_obs),
        "absret_acf5": lag_corr(absolute, 5, context.config.min_corr_obs),
        "rv_ratio_5m_1m": safe_divide(rv5, rv),
        "rv_ratio_15m_1m": safe_divide(rv15, rv),
    }
