"""Independent 84-dimensional market-information feature generator."""

from typing import Any, List, Mapping

import pandas as pd

from market_guided_stock_alpha.features.market_state import (
    MARKET_WINDOWS,
    build_market_manifest,
    transform_market_history,
)


MARKET84_UNIVERSES = ("csi300", "csi500", "csi1000", "csi_all")
MARKET84_INDEX_CODES = ("000300.SH", "000905.SH", "000852.SH", "000985.SH")
MARKET84_WINDOWS = MARKET_WINDOWS


class Market84SpecError(ValueError):
    pass


def validate_market84_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise Market84SpecError("Market84 spec must be an object")
    if tuple(spec.get("universes", [])) != MARKET84_UNIVERSES:
        raise Market84SpecError(
            "Market84 universe order must be csi300,csi500,csi1000,csi_all"
        )
    if tuple(spec.get("index_codes", [])) != MARKET84_INDEX_CODES:
        raise Market84SpecError("Market84 index-code order changed")
    if tuple(spec.get("windows", [])) != MARKET84_WINDOWS:
        raise Market84SpecError("Market84 windows must equal 5,10,20,30,60")
    if spec.get("rolling_min_periods") != 1 or spec.get("rolling_std_ddof") != 1:
        raise Market84SpecError(
            "Market84 rolling min_periods and std ddof must equal 1"
        )
    if (
        spec.get("feature_count_per_index") != 21
        or spec.get("total_feature_count") != 84
    ):
        raise Market84SpecError("Market84 feature counts must equal 21 and 84")
    epsilon = spec.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or epsilon <= 0
    ):
        raise Market84SpecError("Market84 epsilon must be positive")


class Market84Generator:
    """Generate 21 features per index in the fixed four-index order."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_market84_spec(spec)
        self.spec = dict(spec)
        self.epsilon = float(spec["epsilon"])
        self.last_non_finite_to_nan_count = 0
        self._manifest = build_market_manifest(MARKET84_UNIVERSES, 84)

    def get_feature_names(self) -> List[str]:
        return self._manifest["feature_name"].tolist()

    def get_feature_manifest(self) -> pd.DataFrame:
        return self._manifest.copy(deep=True)

    def transform(self, index_market_history: pd.DataFrame) -> pd.DataFrame:
        output, count = transform_market_history(
            index_market_history,
            MARKET84_UNIVERSES,
            self.get_feature_names(),
            "Market84",
        )
        self.last_non_finite_to_nan_count = count
        return output
