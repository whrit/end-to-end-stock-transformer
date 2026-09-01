"""F161-F176: morning/afternoon distribution, geometry, and flow shape."""

import numpy as np

from market_guided_stock_alpha.features.intraday.common import (
    DayContext,
    cached_nonnegative_field,
    finite_sum,
    safe_divide,
    safe_hhi,
)
from market_guided_stock_alpha.features.intraday.distribution import (
    corrected_excess_kurtosis,
    corrected_skew,
)
from market_guided_stock_alpha.features.intraday.geometry import cached_geometry


def _downside_share(returns: np.ndarray) -> np.ndarray:
    valid = np.isfinite(returns)
    rv = np.where(valid, returns * returns, 0.0).sum(axis=0)
    downside = np.where(valid & (returns < 0), returns * returns, 0.0).sum(axis=0)
    return safe_divide(downside, rv)


def _signed_volume(returns: np.ndarray, volume: np.ndarray) -> np.ndarray:
    valid = np.isfinite(returns) & np.isfinite(volume)
    numerator = np.where(valid, np.sign(returns) * volume, 0.0).sum(axis=0)
    return safe_divide(numerator, finite_sum(volume))


def compute(context: DayContext):
    morning = context.morning_slice
    afternoon = context.afternoon_slice
    morning_returns = context.returns[morning]
    afternoon_returns = context.returns[afternoon]
    volume = cached_nonnegative_field(context, "volume")
    morning_geometry = cached_geometry(context, "morning", morning_returns)
    afternoon_geometry = cached_geometry(context, "afternoon", afternoon_returns)
    return {
        "morning_ret_skew": corrected_skew(morning_returns),
        "afternoon_ret_skew": corrected_skew(afternoon_returns),
        "morning_ret_kurt": corrected_excess_kurtosis(morning_returns),
        "afternoon_ret_kurt": corrected_excess_kurtosis(afternoon_returns),
        "morning_downside_rv_share": _downside_share(morning_returns),
        "afternoon_downside_rv_share": _downside_share(afternoon_returns),
        "morning_path_efficiency": morning_geometry["path_efficiency"],
        "afternoon_path_efficiency": afternoon_geometry["path_efficiency"],
        "morning_abs_chord_area": morning_geometry["abs_chord_area"],
        "afternoon_abs_chord_area": afternoon_geometry["abs_chord_area"],
        "morning_curvature_energy": morning_geometry["curvature_energy"],
        "afternoon_curvature_energy": afternoon_geometry["curvature_energy"],
        "morning_signed_volume_imbalance": _signed_volume(
            morning_returns, volume[morning]
        ),
        "afternoon_signed_volume_imbalance": _signed_volume(
            afternoon_returns, volume[afternoon]
        ),
        "morning_volume_hhi": safe_hhi(volume[morning]),
        "afternoon_volume_hhi": safe_hhi(volume[afternoon]),
    }
