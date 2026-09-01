"""Fixed 63-dimensional MASTER market-information generator."""

from typing import Any, List, Mapping

import pandas as pd

from market_guided_stock_alpha.features.market_state import (
    MARKET_WINDOWS,
    build_market_manifest,
    transform_market_history,
)


MARKET_UNIVERSES = ("csi300", "csi500", "csi800")
MARKET63_INDEX_CODES = ("000300.SH", "000905.SH", "000906.SH")


class Market63SpecError(ValueError):
    pass


def validate_market63_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise Market63SpecError("Market63 spec must be an object")
    if tuple(spec.get("universes", [])) != MARKET_UNIVERSES:
        raise Market63SpecError(
            "Market63 universe order must be csi300,csi500,csi800"
        )
    if tuple(spec.get("windows", [])) != MARKET_WINDOWS:
        raise Market63SpecError("Market63 windows must equal 5,10,20,30,60")
    if spec.get("rolling_min_periods") != 1 or spec.get("rolling_std_ddof") != 1:
        raise Market63SpecError(
            "Market63 rolling min_periods and std ddof must equal 1"
        )
    if (
        spec.get("feature_count_per_index") != 21
        or spec.get("total_feature_count") != 63
    ):
        raise Market63SpecError("Market63 feature counts must equal 21 and 63")
    epsilon = spec.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or epsilon <= 0
    ):
        raise Market63SpecError("Market63 epsilon must be positive")


class Market63Generator:
    """Generate 21 features per index in fixed CSI300/500/800 order."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_market63_spec(spec)
        self.spec = dict(spec)
        self.epsilon = float(spec["epsilon"])
        self.last_non_finite_to_nan_count = 0
        self._manifest = build_market_manifest(MARKET_UNIVERSES, 63)

    def get_feature_names(self) -> List[str]:
        return self._manifest["feature_name"].tolist()

    def get_feature_manifest(self) -> pd.DataFrame:
        return self._manifest.copy(deep=True)

    def transform(self, index_market_history: pd.DataFrame) -> pd.DataFrame:
        output, count = transform_market_history(
            index_market_history,
            MARKET_UNIVERSES,
            self.get_feature_names(),
            "Market63",
        )
        self.last_non_finite_to_nan_count = count
        return output
