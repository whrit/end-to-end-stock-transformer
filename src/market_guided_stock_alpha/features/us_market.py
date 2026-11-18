"""US ETF market-state families reusing the shared 21-feature formulas."""

from typing import Any, List, Mapping

import pandas as pd

from market_guided_stock_alpha.features.market_state import (
    MARKET_WINDOWS,
    build_market_manifest,
    transform_market_history,
)


US_MARKET63_UNIVERSES = ("spy", "ijh", "ijr")
US_MARKET63_INDEX_CODES = ("SPY.US", "IJH.US", "IJR.US")
US_MARKET84_UNIVERSES = US_MARKET63_UNIVERSES + ("iwm",)
US_MARKET84_INDEX_CODES = US_MARKET63_INDEX_CODES + ("IWM.US",)
US_MARKET_FAMILIES = {
    "us_market63": (US_MARKET63_UNIVERSES, US_MARKET63_INDEX_CODES),
    "us_market84": (US_MARKET84_UNIVERSES, US_MARKET84_INDEX_CODES),
}


class USMarketSpecError(ValueError):
    pass


def validate_us_market_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise USMarketSpecError("US market spec must be an object")
    name = spec.get("name")
    if name not in US_MARKET_FAMILIES:
        raise USMarketSpecError("US market spec name must be us_market63 or us_market84")
    universes, codes = US_MARKET_FAMILIES[name]
    if tuple(spec.get("universes", [])) != universes:
        raise USMarketSpecError(
            "{} universe order must be {}".format(name, ",".join(universes))
        )
    if tuple(spec.get("index_codes", [])) != codes:
        raise USMarketSpecError("{} index-code order changed".format(name))
    if tuple(spec.get("windows", [])) != MARKET_WINDOWS:
        raise USMarketSpecError("{} windows must equal 5,10,20,30,60".format(name))
    if spec.get("rolling_min_periods") != 1 or spec.get("rolling_std_ddof") != 1:
        raise USMarketSpecError(
            "{} rolling min_periods and std ddof must equal 1".format(name)
        )
    total = 21 * len(universes)
    if (
        spec.get("feature_count_per_index") != 21
        or spec.get("total_feature_count") != total
    ):
        raise USMarketSpecError(
            "{} feature counts must equal 21 and {}".format(name, total)
        )
    epsilon = spec.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or epsilon <= 0
    ):
        raise USMarketSpecError("{} epsilon must be positive".format(name))


class USMarketGenerator:
    """Generate 21 features per ETF in the fixed family order."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_us_market_spec(spec)
        self.spec = dict(spec)
        self.family = str(spec["name"])
        self.universes, self.index_codes = US_MARKET_FAMILIES[self.family]
        self.epsilon = float(spec["epsilon"])
        self.last_non_finite_to_nan_count = 0
        self._manifest = build_market_manifest(
            self.universes, int(spec["total_feature_count"])
        )

    def get_feature_names(self) -> List[str]:
        return self._manifest["feature_name"].tolist()

    def get_feature_manifest(self) -> pd.DataFrame:
        return self._manifest.copy(deep=True)

    def transform(self, index_market_history: pd.DataFrame) -> pd.DataFrame:
        output, count = transform_market_history(
            index_market_history,
            self.universes,
            self.get_feature_names(),
            self.family,
        )
        self.last_non_finite_to_nan_count = count
        return output
