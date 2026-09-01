"""Direct canonical-DataFrame to derived-feature artifact builders."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    validate_daily_index,
    validate_daily_stock,
    validate_minute_index,
    validate_minute_stock,
)
from market_guided_stock_alpha.data.minute_session import normalize_minute_session
from market_guided_stock_alpha.derived.artifact import write_feature_artifact
from market_guided_stock_alpha.derived.specs import (
    alpha158_spec,
    intraday_market_spec,
    intraday_price_volume_spec,
    market63_spec,
    market84_spec,
)
from market_guided_stock_alpha.features.alpha158 import Alpha158Generator
from market_guided_stock_alpha.features.intraday import IntradayPriceVolumeGenerator
from market_guided_stock_alpha.features.intraday_market import (
    IntradayMarketGenerator,
)
from market_guided_stock_alpha.features.market63 import (
    MARKET63_INDEX_CODES,
    Market63Generator,
)
from market_guided_stock_alpha.features.market84 import (
    MARKET84_INDEX_CODES,
    Market84Generator,
)


def _provenance(
    spec: Mapping[str, Any], extra: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    result = {"feature_spec": dict(spec)}
    result.update(dict(extra or {}))
    return result


def _dense_cross_section(
    features: pd.DataFrame,
    source_keys: pd.DataFrame,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, pd.DatetimeIndex, Tuple[str, ...], np.ndarray]:
    dates = pd.DatetimeIndex(sorted(features["trade_date"].unique()))
    codes = tuple(sorted(features["sec_code"].unique()))
    full = pd.MultiIndex.from_product(
        [dates, codes], names=["trade_date", "sec_code"]
    )
    dense = features.set_index(["trade_date", "sec_code"]).reindex(full)
    values = dense.loc[:, list(feature_names)].to_numpy().reshape(
        len(dates), len(codes), len(feature_names)
    )
    observed_series = pd.Series(
        True,
        index=pd.MultiIndex.from_frame(
            source_keys.loc[:, ["trade_date", "sec_code"]]
        ),
    )
    observed = observed_series.reindex(full, fill_value=False).to_numpy().reshape(
        len(dates), len(codes)
    )
    return values, dates, codes, observed


def build_alpha158_artifact(
    daily_stock: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    source = validate_daily_stock(daily_stock)
    spec = alpha158_spec()
    generator = Alpha158Generator(spec)
    features = generator.transform_market_history(source)
    values, dates, codes, observed = _dense_cross_section(
        features, source, generator.get_feature_names()
    )
    return write_feature_artifact(
        output_path,
        "alpha158",
        values.astype("float64", copy=False),
        dates,
        generator.get_feature_names(),
        security_codes=codes,
        source_observed=observed,
        provenance=_provenance(spec, provenance),
    )


def _market_observed(
    source: pd.DataFrame, dates: pd.DatetimeIndex, universes: Sequence[str]
) -> np.ndarray:
    usable = source.assign(
        _usable=np.isfinite(source["close"]) & np.isfinite(source["amount"])
    )
    counts = usable.loc[usable["_usable"]].groupby("trade_date")[
        "universe"
    ].nunique()
    return np.asarray(
        [int(counts.get(date, 0)) == len(universes) for date in dates],
        dtype="bool",
    )


def _build_market_artifact(
    daily_index: pd.DataFrame,
    output_path: Path,
    family: str,
    generator: Any,
    universes: Sequence[str],
    index_codes: Sequence[str],
    spec: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    source = validate_daily_index(daily_index)
    mapping = dict(zip(index_codes, universes))
    unknown = sorted(set(source["index_code"]).difference(mapping))
    if unknown:
        raise ValueError(
            "{} input contains unexpected index codes: {}".format(
                family, ",".join(unknown)
            )
        )
    source["universe"] = source["index_code"].map(mapping)
    features = generator.transform(source)
    dates = pd.DatetimeIndex(features["trade_date"])
    values = features.loc[:, generator.get_feature_names()].to_numpy(
        dtype="float64"
    )
    return write_feature_artifact(
        output_path,
        family,
        values,
        dates,
        generator.get_feature_names(),
        source_observed=_market_observed(source, dates, universes),
        provenance=_provenance(spec, provenance),
    )


def build_market63_artifact(
    daily_index: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    spec = market63_spec()
    return _build_market_artifact(
        daily_index,
        output_path,
        "market63",
        Market63Generator(spec),
        spec["universes"],
        MARKET63_INDEX_CODES,
        spec,
        provenance,
    )


def build_market84_artifact(
    daily_index: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    spec = market84_spec()
    return _build_market_artifact(
        daily_index,
        output_path,
        "market84",
        Market84Generator(spec),
        spec["universes"],
        MARKET84_INDEX_CODES,
        spec,
        provenance,
    )


def build_intraday_price_volume_artifact(
    minute_stock: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    source = validate_minute_stock(minute_stock)
    spec = intraday_price_volume_spec()
    generator = IntradayPriceVolumeGenerator(spec)
    dates = pd.DatetimeIndex(source["timestamp"]).normalize().unique().sort_values()
    codes = tuple(sorted(source["sec_code"].unique()))
    values = np.full((len(dates), len(codes), 248), np.nan, dtype="float32")
    observed = np.zeros((len(dates), len(codes)), dtype="bool")
    coverage = np.zeros((len(dates), len(codes)), dtype="float32")
    for position, day in enumerate(dates):
        frame = source.loc[
            pd.DatetimeIndex(source["timestamp"]).normalize() == day
        ]
        session = normalize_minute_session(
            frame, "sec_code", trade_date=day, instruments=codes
        )
        matrix, day_coverage, eligible = generator.transform_day_arrays(
            session.fields
        )
        values[position] = matrix
        coverage[position] = day_coverage
        observed[position] = eligible
    return write_feature_artifact(
        output_path,
        "intraday_price_volume",
        values,
        dates,
        generator.get_factor_names(),
        security_codes=codes,
        source_observed=observed,
        bar_coverage=coverage,
        provenance=_provenance(
            spec,
            {
                **dict(provenance or {}),
                "minute_session": "09:31-11:30,13:01-15:00",
                "interpolation": "none",
            },
        ),
    )


def build_intraday_market_artifact(
    minute_index: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    source = validate_minute_index(minute_index)
    spec = intraday_market_spec()
    generator = IntradayMarketGenerator(spec)
    dates = pd.DatetimeIndex(source["timestamp"]).normalize().unique().sort_values()
    codes = tuple(MARKET84_INDEX_CODES)
    values = np.full((len(dates), 192), np.nan, dtype="float64")
    observed = np.zeros(len(dates), dtype="bool")
    total_ohlc_repair_count = 0
    for position, day in enumerate(dates):
        frame = source.loc[
            pd.DatetimeIndex(source["timestamp"]).normalize() == day
        ]
        session = normalize_minute_session(
            frame, "index_code", trade_date=day, instruments=codes
        )
        day_fields = {
            field: array.T for field, array in session.fields.items()
        }
        result, diagnostic = generator.compute_day(day_fields)
        values[position] = result
        observed[position] = bool(diagnostic["source_observed"])
        total_ohlc_repair_count += int(diagnostic["ohlc_repair_count"])
    return write_feature_artifact(
        output_path,
        "intraday_market",
        values,
        dates,
        generator.get_feature_names(),
        source_observed=observed,
        provenance=_provenance(
            spec,
            {
                **dict(provenance or {}),
                "position": "exploratory_market_state",
                "minute_session": "09:31-11:30,13:01-15:00",
                "interpolation": "none",
                "source_observed_day_count": int(observed.sum()),
                "total_ohlc_repair_count": total_ohlc_repair_count,
            },
        ),
    )
