"""Immutable F001-F248 intraday price-volume feature registry."""

from typing import Any, Dict, List, Tuple

from market_guided_stock_alpha.features.intraday.registry_base import (
    FACTOR_NAMES as BASE_FACTOR_NAMES,
)


EXTENDED_FACTOR_NAMES: Tuple[str, ...] = (
    "max_positive_minute_return", "max_negative_minute_return",
    "max_abs_minute_return", "max_positive_return_position",
    "max_negative_return_position", "max_abs_return_position",
    "max_volume_position", "max_amount_position", "intraday_high_position",
    "intraday_low_position", "max_drawdown_end_position",
    "max_drawup_end_position",
    "morning_ret_skew", "afternoon_ret_skew", "morning_ret_kurt",
    "afternoon_ret_kurt", "morning_downside_rv_share",
    "afternoon_downside_rv_share", "morning_path_efficiency",
    "afternoon_path_efficiency", "morning_abs_chord_area",
    "afternoon_abs_chord_area", "morning_curvature_energy",
    "afternoon_curvature_energy", "morning_signed_volume_imbalance",
    "afternoon_signed_volume_imbalance", "morning_volume_hhi",
    "afternoon_volume_hhi",
    "ret_skew_5m", "ret_skew_15m", "ret_skew_30m", "ret_kurt_5m",
    "ret_kurt_15m", "ret_kurt_30m", "sign_change_ratio_5m",
    "sign_change_ratio_15m", "sign_change_ratio_30m", "rv_hhi_5m",
    "rv_hhi_15m", "rv_hhi_30m", "rv_entropy_5m", "rv_entropy_15m",
    "rv_entropy_30m", "volume_hhi_5m", "volume_hhi_15m",
    "volume_hhi_30m", "volume_entropy_5m", "volume_entropy_15m",
    "rv_hhi_1m", "rv_entropy_1m", "max_minute_rv_share",
    "top1pct_rv_share", "median_absret", "ret_mad", "ret_iqr",
    "mean_minute_log_range", "log_range_std", "range_hhi",
    "intraday_trend_slope", "intraday_trend_r2", "detrended_path_std",
    "max_5m_abs_return", "max_15m_abs_return", "max_30m_abs_return",
    "corr_ret_amount", "corr_absret_amount", "amount_lead_ret_corr",
    "amount_lead_absret_corr", "ret_lead_amount_corr",
    "absret_lead_amount_corr", "close_vs_day_vwap", "open_vs_day_vwap",
    "day_vwap_location", "volume_weighted_return",
    "amount_weighted_return", "corr_absret_vwap_deviation",
    "signed_volume_impact", "signed_amount_impact",
    "positive_return_volume_share", "negative_return_volume_share",
    "ret_fft_low_high_logratio", "absret_fft_low_high_logratio",
    "volume_fft_low_high_logratio", "ret_spectral_slope",
    "absret_spectral_slope", "volume_spectral_slope",
    "mean_abs_curvature_5m", "mean_abs_curvature_15m",
    "mean_abs_curvature_30m", "curvature_energy_5m",
    "curvature_energy_15m", "curvature_energy_30m",
    "block1_downside_rv_share", "block2_downside_rv_share",
    "block3_downside_rv_share", "block4_downside_rv_share",
    "block1_signed_volume_imbalance", "block2_signed_volume_imbalance",
    "block3_signed_volume_imbalance", "block4_signed_volume_imbalance",
)

FACTOR_NAMES: Tuple[str, ...] = tuple(BASE_FACTOR_NAMES) + EXTENDED_FACTOR_NAMES
FACTOR_IDS: Tuple[str, ...] = tuple(
    "F{:03d}".format(position) for position in range(1, 249)
)

_ADDITIONAL_CATEGORY_RANGES = (
    (149, 160, "event_timing"),
    (161, 176, "cross_session"),
    (177, 196, "multiscale"),
    (197, 212, "volatility"),
    (213, 228, "price_volume"),
    (229, 234, "spectral"),
    (235, 240, "geometry"),
    (241, 248, "position_blocks"),
)


def _base_categories() -> Dict[str, str]:
    from market_guided_stock_alpha.features.intraday.registry_base import FACTOR_REGISTRY
    return {factor_id: category for factor_id, _name, category in FACTOR_REGISTRY}


def _category(position: int) -> str:
    if position <= 148:
        return _base_categories()["F{:03d}".format(position)]
    for start, end, category in _ADDITIONAL_CATEGORY_RANGES:
        if start <= position <= end:
            return category
    raise ValueError("intraday factor position has no category: {}".format(position))


FACTOR_REGISTRY: Tuple[Tuple[str, str, str], ...] = tuple(
    (factor_id, name, _category(position))
    for position, (factor_id, name) in enumerate(
        zip(FACTOR_IDS, FACTOR_NAMES), start=1
    )
)

if FACTOR_IDS != tuple("F{:03d}".format(position) for position in range(1, 249)):
    raise RuntimeError("intraday factor IDs must equal F001..F248")
if len(FACTOR_REGISTRY) != 248:
    raise RuntimeError("intraday factor registry must contain 248 factors")
if len(set(FACTOR_IDS)) != 248 or len(set(FACTOR_NAMES)) != 248:
    raise RuntimeError("intraday factor IDs and names must be unique")


def factor_manifest() -> List[Dict[str, Any]]:
    return [
        {
            "factor_index": index,
            "factor_id": factor_id,
            "factor_name": name,
            "category": category,
        }
        for index, (factor_id, name, category) in enumerate(FACTOR_REGISTRY)
    ]
