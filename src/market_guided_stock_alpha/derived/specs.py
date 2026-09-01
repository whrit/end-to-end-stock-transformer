"""Fixed public feature-definition contracts, separate from model configs."""

from typing import Any, Dict

from market_guided_stock_alpha.features.alpha158 import (
    KLINE_FEATURES,
    PRICE_FEATURES,
    ROLLING_FAMILIES,
)
from market_guided_stock_alpha.features.intraday_market import UNIVERSES
from market_guided_stock_alpha.features.market63 import (
    MARKET63_INDEX_CODES,
    MARKET_UNIVERSES,
)
from market_guided_stock_alpha.features.market84 import (
    MARKET84_INDEX_CODES,
    MARKET84_UNIVERSES,
)


def alpha158_spec() -> Dict[str, Any]:
    return {
        "name": "alpha158",
        "version": 1,
        "source": {
            "repository": "microsoft/qlib",
            "file": "qlib/contrib/data/loader.py",
            "class": "Alpha158DL",
            "blob_sha": "4d11f3a34cc1d12728723956c8e9324556eee8e3",
        },
        "epsilon": 1e-12,
        "rolling_windows": [5, 10, 20, 30, 60],
        "rolling_min_periods": 1,
        "rolling_std_ddof": 1,
        "sanitize_non_finite_to_nan": True,
        "feature_count": 158,
        "feature_group_order": ["kbar", "price", "rolling"],
        "kbar_features": list(KLINE_FEATURES),
        "price_features": list(PRICE_FEATURES),
        "rolling_family_order": list(ROLLING_FAMILIES),
    }


def market63_spec() -> Dict[str, Any]:
    return {
        "name": "market63",
        "version": 1,
        "universes": list(MARKET_UNIVERSES),
        "index_codes": list(MARKET63_INDEX_CODES),
        "windows": [5, 10, 20, 30, 60],
        "rolling_min_periods": 1,
        "rolling_std_ddof": 1,
        "epsilon": 1e-12,
        "feature_count_per_index": 21,
        "total_feature_count": 63,
    }


def market84_spec() -> Dict[str, Any]:
    return {
        "name": "market84",
        "version": 1,
        "universes": list(MARKET84_UNIVERSES),
        "index_codes": list(MARKET84_INDEX_CODES),
        "windows": [5, 10, 20, 30, 60],
        "rolling_min_periods": 1,
        "rolling_std_ddof": 1,
        "epsilon": 1e-12,
        "feature_count_per_index": 21,
        "total_feature_count": 84,
    }


def intraday_price_volume_spec() -> Dict[str, Any]:
    return {
        "name": "intraday_price_volume",
        "factor_count": 248,
        "min_bar_coverage": 0.90,
        "structural_zero_threshold": 0.80,
        "epsilon": 1e-12,
        "zero_return_tolerance": 1e-12,
        "minimum_observations": {
            "correlation": 10,
            "multiscale_correlation": 4,
            "fft": 20,
            "geometry": 5,
        },
        "session_contract": {
            "bars_per_day": 240,
            "morning_bars": 120,
            "afternoon_bars": 120,
        },
        "stock_chunk_size": 256,
        "output_dtype": "float32",
    }


def intraday_market_spec() -> Dict[str, Any]:
    return {
        "universes": list(UNIVERSES),
        "bars_per_day": 240,
        "feature_count_per_index": 48,
        "total_feature_count": 192,
        "epsilon": 1e-12,
        "minute_volume_semantics": "increment_not_cumulative",
        "minute_amount_semantics": "increment_not_cumulative",
        "volume_positive_scale_invariant": True,
        "amount_positive_scale_invariant": True,
        "ohlc_repair": (
            "high=max(high,open,close);low=min(low,open,close);in_memory_only"
        ),
        "block_returns_cross_lunch": False,
        "one_minute_return_preserves_lunch_jump": True,
        "return_shape_moments": (
            "z=(r-mean(r))/std_pop(r);if_std_le_epsilon_skew_and_"
            "excess_kurt_are_zero"
        ),
        "return_excess_kurtosis": "mean(z^4)-3",
    }
