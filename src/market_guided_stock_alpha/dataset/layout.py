"""Canonical feature layouts derived from fixed feature definitions."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

from market_guided_stock_alpha.derived.specs import (
    alpha158_spec,
    intraday_market_spec,
    intraday_price_volume_spec,
    market63_spec,
    market84_spec,
    us_market63_spec,
    us_market84_spec,
)
from market_guided_stock_alpha.features.alpha158 import Alpha158Generator
from market_guided_stock_alpha.features.intraday import IntradayPriceVolumeGenerator
from market_guided_stock_alpha.features.intraday_market import (
    IntradayMarketGenerator,
)
from market_guided_stock_alpha.features.market63 import Market63Generator
from market_guided_stock_alpha.features.market84 import Market84Generator
from market_guided_stock_alpha.features.us_market import USMarketGenerator


MARKET_FAMILIES = ("market63", "market84", "us_market63", "us_market84")


LOOKBACK_TRADING_DAYS = 8


@lru_cache(maxsize=None)
def feature_names(family: str) -> Tuple[str, ...]:
    """Resolve feature order from its canonical formula implementation."""
    if family == "alpha158":
        return tuple(Alpha158Generator(alpha158_spec()).get_feature_names())
    if family == "intraday_price_volume":
        return tuple(IntradayPriceVolumeGenerator(intraday_price_volume_spec()).get_factor_names())
    if family == "market63":
        return tuple(Market63Generator(market63_spec()).get_feature_names())
    if family == "market84":
        return tuple(Market84Generator(market84_spec()).get_feature_names())
    if family == "us_market63":
        return tuple(USMarketGenerator(us_market63_spec()).get_feature_names())
    if family == "us_market84":
        return tuple(USMarketGenerator(us_market84_spec()).get_feature_names())
    if family == "intraday_market":
        return tuple(
            IntradayMarketGenerator(
                intraday_market_spec()
            ).get_feature_names()
        )
    raise ValueError("unknown canonical feature family: {}".format(family))


@dataclass(frozen=True)
class FeatureLayout:
    """Neutral stock/market component layout; market data stays date-level."""

    stock_families: Tuple[str, ...]
    market_family: str
    intraday_market_family: str = ""
    position: str = "core"
    lookback: int = LOOKBACK_TRADING_DAYS

    def __post_init__(self) -> None:
        if self.stock_families not in (
            ("alpha158",),
            ("alpha158", "intraday_price_volume"),
        ):
            raise ValueError("unsupported stock feature layout")
        if self.market_family not in MARKET_FAMILIES:
            raise ValueError(
                "market family must be one of {}".format(",".join(MARKET_FAMILIES))
            )
        if self.intraday_market_family not in ("", "intraday_market"):
            raise ValueError("unknown exploratory intraday market family")
        if self.intraday_market_family and self.position != "exploratory":
            raise ValueError("intraday market layout must be exploratory")
        if self.lookback != LOOKBACK_TRADING_DAYS:
            raise ValueError("canonical lookback must equal eight trading days")

    @property
    def stock_feature_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for family in self.stock_families
            for name in feature_names(family)
        )

    @property
    def market_feature_names(self) -> Tuple[str, ...]:
        return feature_names(self.market_family)

    @property
    def intraday_market_feature_names(self) -> Tuple[str, ...]:
        return (
            feature_names(self.intraday_market_family)
            if self.intraday_market_family
            else ()
        )

    @property
    def stock_dim(self) -> int:
        return len(self.stock_feature_names)

    @property
    def market_dim(self) -> int:
        return len(self.market_feature_names)

    @property
    def intraday_market_dim(self) -> int:
        return len(self.intraday_market_feature_names)

    @property
    def total_market_dim(self) -> int:
        return self.market_dim + self.intraday_market_dim

    @property
    def conceptual_input_dim(self) -> int:
        return self.stock_dim + self.total_market_dim

    @property
    def family_slices(self) -> Dict[str, Tuple[int, int]]:
        result = {}
        start = 0
        for family in self.stock_families:
            end = start + len(feature_names(family))
            result[family] = (start, end)
            start = end
        return result

    @classmethod
    def stage1(cls, market_family: str = "market63") -> "FeatureLayout":
        return cls(("alpha158",), market_family, position="core_stage1")

    @classmethod
    def stage2(cls, market_family: str = "market63") -> "FeatureLayout":
        position = "core_stage2" if market_family == "market63" else "secondary_stage2"
        return cls(
            ("alpha158", "intraday_price_volume"), market_family, position=position
        )

    @classmethod
    def dual_frequency(cls) -> "FeatureLayout":
        return cls(
            ("alpha158", "intraday_price_volume"),
            "market84",
            intraday_market_family="intraday_market",
            position="exploratory",
        )
