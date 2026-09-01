"""Immutable F001-F148 base registry used by finalized Intraday Price-Volume Features."""

from typing import Any, Dict, List, Tuple


_BASE_FACTOR_NAMES = (
    "ret_skew", "ret_kurt", "positive_ret_ratio", "zero_ret_ratio",
    "sign_change_ratio", "longest_up_run_ratio", "longest_down_run_ratio",
    "absret_q90", "absret_q95", "absret_q99", "tail_asymmetry",
    "extreme_rv_share", "realized_variance", "bipower_variation",
    "jump_share", "downside_rv", "downside_rv_share",
    "realized_quarticity", "intramin_parkinson_var", "intramin_rs_var",
    "intraday_max_drawdown", "intraday_max_drawup", "ret_acf1",
    "ret_acf5", "absret_acf1", "absret_acf5", "rv_ratio_5m_1m",
    "rv_ratio_15m_1m", "volume_hhi", "volume_entropy", "volume_gini",
    "amount_hhi", "max_minute_volume_share", "top5pct_volume_share",
    "zero_volume_ratio", "volume_cv", "illiq_mean", "illiq_p90",
    "vwap_path_absdev", "minute_vwap_dispersion", "corr_ret_volume",
    "corr_absret_volume", "corr_ret_dvolume", "corr_absret_dvolume",
    "volume_lead_ret_corr", "volume_lead_absret_corr",
    "ret_lead_volume_corr", "absret_lead_volume_corr",
    "signed_volume_imbalance", "signed_amount_imbalance",
    "high_volume_ret", "high_low_volume_ret_diff", "open30_ret",
    "midmorning_ret", "midafternoon_ret", "close30_ret",
    "open30_rv_share", "midmorning_rv_share", "midafternoon_rv_share",
    "close30_rv_share", "open30_volume_share",
    "midmorning_volume_share", "midafternoon_volume_share",
    "close30_volume_share", "open30_range", "midmorning_range",
    "midafternoon_range", "close30_range", "ret_fft_low_share",
    "ret_fft_high_share", "ret_fft_entropy", "ret_fft_centroid",
    "ret_fft_dominant_period", "absret_fft_low_share",
    "absret_fft_high_share", "absret_fft_entropy",
    "absret_fft_centroid", "absret_fft_dominant_period",
    "volume_fft_low_share", "volume_fft_high_share",
    "volume_fft_entropy", "volume_fft_centroid",
    "volume_fft_dominant_period", "path_efficiency",
    "arc_chord_ratio", "signed_chord_area", "abs_chord_area",
    "max_chord_deviation", "rms_chord_deviation", "turning_rate_1m",
    "turning_rate_5m", "turning_rate_15m", "turning_rate_30m",
    "mean_turn_angle", "p90_turn_angle", "mean_abs_curvature",
    "p90_abs_curvature", "curvature_energy", "roughness_5m_1m",
    "roughness_15m_1m",
)

_BLOCK_NAMES = tuple(
    "block{}_{}".format(block, metric)
    for block in range(1, 5)
    for metric in (
        "ret", "rv_share", "volume_share", "amount_share", "range",
        "path_efficiency", "abs_chord_area", "curvature_energy",
    )
)

_MULTISCALE_NAMES = (
    "rv_ratio_30m_1m",
    "ret_acf1_5m", "ret_acf1_15m", "ret_acf1_30m",
    "absret_acf1_5m", "absret_acf1_15m", "absret_acf1_30m",
    "path_efficiency_5m", "path_efficiency_15m", "path_efficiency_30m",
)

_CROSS_SESSION_NAMES = (
    "first60_last60_ret_diff",
    "first60_last60_rv_logratio",
    "first60_last60_volume_share_diff",
    "morning_afternoon_ret_diff",
    "morning_afternoon_rv_logratio",
    "morning_afternoon_volume_share_diff",
)

FACTOR_NAMES: Tuple[str, ...] = (
    _BASE_FACTOR_NAMES + _BLOCK_NAMES + _MULTISCALE_NAMES + _CROSS_SESSION_NAMES
)
FACTOR_IDS: Tuple[str, ...] = tuple(
    "F{:03d}".format(position) for position in range(1, len(FACTOR_NAMES) + 1)
)

_CATEGORY_RANGES = (
    (1, 12, "distribution"),
    (13, 28, "volatility"),
    (29, 40, "liquidity"),
    (41, 52, "price_volume"),
    (53, 68, "time_of_day"),
    (69, 83, "spectral"),
    (84, 100, "geometry"),
    (101, 132, "position_blocks"),
    (133, 142, "multiscale"),
    (143, 148, "cross_session"),
)


def _category(position: int) -> str:
    for start, end, category in _CATEGORY_RANGES:
        if start <= position <= end:
            return category
    raise ValueError("factor position has no category: {}".format(position))


FACTOR_REGISTRY: Tuple[Tuple[str, str, str], ...] = tuple(
    (factor_id, name, _category(position))
    for position, (factor_id, name) in enumerate(
        zip(FACTOR_IDS, FACTOR_NAMES), start=1
    )
)

if len(FACTOR_REGISTRY) != 148:
    raise RuntimeError("Intraday Price-Volume Features base registry must contain 148 factors")
if len(set(FACTOR_IDS)) != 148 or len(set(FACTOR_NAMES)) != 148:
    raise RuntimeError("Intraday Price-Volume Features base IDs and names must be unique")


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


def factor_id_to_name() -> Dict[str, str]:
    return {factor_id: name for factor_id, name, _category_name in FACTOR_REGISTRY}
