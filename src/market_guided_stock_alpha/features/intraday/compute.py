"""Orchestrator for the frozen 248 intraday price-volume features."""

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import market_guided_stock_alpha.features.intraday.distribution as distribution
import market_guided_stock_alpha.features.intraday.event_timing as event_timing
import market_guided_stock_alpha.features.intraday.geometry as geometry
import market_guided_stock_alpha.features.intraday.liquidity as liquidity
import market_guided_stock_alpha.features.intraday.multiscale as multiscale
import market_guided_stock_alpha.features.intraday.price_volume_base as price_volume_base
import market_guided_stock_alpha.features.intraday.price_volume_extended as price_volume_extended
import market_guided_stock_alpha.features.intraday.session_shape as session_shape
import market_guided_stock_alpha.features.intraday.spectral as spectral
import market_guided_stock_alpha.features.intraday.spectral_geometry as spectral_geometry
import market_guided_stock_alpha.features.intraday.time_of_day as time_of_day
import market_guided_stock_alpha.features.intraday.volatility as volatility
from market_guided_stock_alpha.features.intraday.common import (
    IntradayPriceVolumeConfig,
    prepare_day,
    sanitize_factor_map,
)
from market_guided_stock_alpha.features.intraday.registry import (
    FACTOR_NAMES,
    factor_manifest,
)


FACTOR_MODULES = (
    distribution,
    volatility,
    liquidity,
    price_volume_base,
    time_of_day,
    spectral,
    geometry,
    event_timing,
    session_shape,
    multiscale,
    price_volume_extended,
    spectral_geometry,
)


class IntradayPriceVolumeGenerator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = IntradayPriceVolumeConfig.from_mapping(config)
        if self.config.factor_count != 248 or len(FACTOR_NAMES) != 248:
            raise ValueError("intraday price-volume features require exactly 248 factors")

    def get_factor_names(self):
        return list(FACTOR_NAMES)

    def get_factor_manifest(self):
        return factor_manifest()

    def transform_day_arrays(
        self, raw_fields: Mapping[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        context = prepare_day(raw_fields, self.config)
        factors: Dict[str, np.ndarray] = {}
        for module in FACTOR_MODULES:
            produced = sanitize_factor_map(module.compute(context), context.stock_count)
            duplicate = set(factors).intersection(produced)
            if duplicate:
                raise ValueError(
                    "duplicate intraday factor implementations: {}".format(sorted(duplicate))
                )
            factors.update(produced)
        missing = set(FACTOR_NAMES).difference(factors)
        unexpected = set(factors).difference(FACTOR_NAMES)
        if missing or unexpected or len(factors) != 248:
            raise ValueError(
                "intraday factor implementation/registry mismatch missing={} unexpected={}".format(
                    sorted(missing), sorted(unexpected)
                )
            )
        matrix = np.column_stack([factors[name] for name in FACTOR_NAMES])
        matrix[~np.isfinite(matrix)] = np.nan
        matrix[~context.eligible] = np.nan
        return (
            matrix.astype(self.config.output_dtype, copy=False),
            context.bar_coverage.astype("float32", copy=False),
            context.eligible.astype("bool", copy=False),
        )

    def transform_day(
        self,
        trade_date: Any,
        instruments: Sequence[Any],
        raw_fields: Mapping[str, Any],
    ) -> pd.DataFrame:
        matrix, coverage, eligible = self.transform_day_arrays(raw_fields)
        if len(instruments) != matrix.shape[0]:
            raise ValueError("instrument count does not match day matrices")
        positions = np.flatnonzero(eligible)
        output = pd.DataFrame(matrix[positions], columns=list(FACTOR_NAMES))
        output.insert(0, "bar_coverage", coverage[positions])
        output.insert(0, "instrument", np.asarray(instruments, dtype=object)[positions])
        output.insert(
            0,
            "datetime",
            pd.DatetimeIndex([pd.Timestamp(trade_date).normalize()] * len(output)),
        )
        if output.duplicated(["datetime", "instrument"]).any():
            raise ValueError("daily intraday output has duplicate logical keys")
        return output
