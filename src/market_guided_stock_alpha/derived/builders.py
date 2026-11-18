"""Direct canonical-DataFrame to derived-feature artifact builders."""

import os
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    normalize_security_code,
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
    us_market63_spec,
    us_market84_spec,
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
from market_guided_stock_alpha.features.us_market import USMarketGenerator


def _provenance(
    spec: Mapping[str, Any], extra: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    result = {"feature_spec": dict(spec)}
    result.update(dict(extra or {}))
    return result


ALPHA158_DTYPES = ("float64", "float32")
# ~64 securities x 10y of pandas temporaries stays well under 1 GB per worker,
# so ~10 parallel workers fit an 18 GB machine alongside the parent's input.
ALPHA158_BATCH_SECURITIES = 64
# Worker processes run at reduced OS priority so a long build only uses CPU the
# rest of the machine is not using (interactive apps stay responsive).
ALPHA158_WORKER_NICENESS = 10


def _lower_worker_priority() -> None:
    os.nice(ALPHA158_WORKER_NICENESS)


def _write_alpha158_batch(
    values_path: str,
    batch: pd.DataFrame,
    dates: np.ndarray,
    codes: Tuple[str, ...],
) -> None:
    """Compute one security batch and write it into the shared .npy memmap.

    Batches own disjoint security columns, so concurrent writers never touch
    the same bytes. Module-level so a spawned worker process can import it.
    """
    generator = Alpha158Generator(alpha158_spec())
    features = generator.transform_market_history(batch)
    values = np.lib.format.open_memmap(values_path, mode="r+")
    values[
        pd.DatetimeIndex(dates).get_indexer(features["trade_date"]),
        pd.Index(codes).get_indexer(features["sec_code"]),
    ] = features.loc[:, generator.get_feature_names()].to_numpy(
        dtype="float64"
    ).astype(values.dtype, copy=False)
    values.flush()
    del values


def build_alpha158_artifact(
    daily_stock: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
    dtype: str = "float64",
    security_codes: Optional[Iterable[str]] = None,
    batch_securities: int = ALPHA158_BATCH_SECURITIES,
    workers: int = 1,
) -> Dict[str, Any]:
    """Build Alpha158 in per-security batches into a disk-backed [D,S,F] array.

    Alpha158 is per-security time-series math, so batching is exact and the
    output is identical for any ``batch_securities`` or ``workers``. With
    ``workers > 1`` batches run in a process pool, each worker writing its own
    security columns into the shared ``.npy`` memmap next to the output; the
    file is moved (not copied) into the artifact. ``dtype="float32"`` halves
    storage (training consumes float32 anyway); ``security_codes`` drops
    securities outside that set (e.g. never members).
    """
    if dtype not in ALPHA158_DTYPES:
        raise ValueError("alpha158 dtype must be one of {}".format(ALPHA158_DTYPES))
    if type(batch_securities) is not int or batch_securities <= 0:
        raise ValueError("batch_securities must be a positive integer")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    source = validate_daily_stock(daily_stock)
    storage: Dict[str, Any] = {}
    if security_codes is not None:
        keep = {
            normalize_security_code(value, "alpha158 security filter")
            for value in security_codes
        }
        before = int(source["sec_code"].nunique())
        source = source.loc[source["sec_code"].isin(keep)].reset_index(drop=True)
        if source.empty:
            raise ValueError("alpha158 security filter removed every input row")
        storage["security_filter"] = {
            "requested_count": len(keep),
            "input_security_count": before,
            "kept_security_count": int(source["sec_code"].nunique()),
        }
    if dtype != "float64":
        storage["dtype"] = dtype
    spec = alpha158_spec()
    generator = Alpha158Generator(spec)
    names = generator.get_feature_names()
    dates = pd.DatetimeIndex(sorted(source["trade_date"].unique()))
    codes = tuple(sorted(source["sec_code"].unique()))
    code_index = pd.Index(codes)
    observed = np.zeros((len(dates), len(codes)), dtype="bool")
    observed[
        dates.get_indexer(source["trade_date"]), code_index.get_indexer(source["sec_code"])
    ] = True
    rows_by_code = source.groupby("sec_code", sort=True).indices

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=target.name + ".scratch-", dir=target.parent))
    try:
        values = np.lib.format.open_memmap(
            scratch / "values.npy",
            mode="w+",
            dtype=np.dtype(dtype),
            shape=(len(dates), len(codes), len(names)),
        )
        for start in range(0, len(dates), 64):
            values[start : start + 64] = np.nan
        values.flush()
        values_path = str(scratch / "values.npy")
        date_values = dates.to_numpy()
        batches = (
            source.iloc[
                np.sort(
                    np.concatenate(
                        [rows_by_code[code] for code in codes[start : start + batch_securities]]
                    )
                )
            ]
            for start in range(0, len(codes), batch_securities)
        )
        if workers == 1:
            for batch in batches:
                _write_alpha158_batch(values_path, batch, date_values, codes)
        else:
            # Bound queued batches so the parent never holds a second full
            # copy of the input as pending task arguments.
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_lower_worker_priority
            ) as pool:
                in_flight = set()
                try:
                    for batch in batches:
                        if len(in_flight) >= 2 * workers:
                            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                            for future in done:
                                future.result()
                        in_flight.add(
                            pool.submit(
                                _write_alpha158_batch, values_path, batch, date_values, codes
                            )
                        )
                    for future in in_flight:
                        future.result()
                except BaseException:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
        extra = dict(provenance or {})
        if storage:
            extra["alpha158_storage"] = storage
        return write_feature_artifact(
            target,
            "alpha158",
            values,
            dates,
            names,
            security_codes=codes,
            source_observed=observed,
            provenance=_provenance(spec, extra),
            adopt_values_file=True,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


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


def _build_us_market_artifact(
    daily_index: pd.DataFrame,
    output_path: Path,
    spec: Mapping[str, Any],
    provenance: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    generator = USMarketGenerator(spec)
    return _build_market_artifact(
        daily_index,
        output_path,
        generator.family,
        generator,
        generator.universes,
        generator.index_codes,
        spec,
        provenance,
    )


def build_us_market63_artifact(
    daily_index: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return _build_us_market_artifact(
        daily_index, output_path, us_market63_spec(), provenance
    )


def build_us_market84_artifact(
    daily_index: pd.DataFrame,
    output_path: Path,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return _build_us_market_artifact(
        daily_index, output_path, us_market84_spec(), provenance
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
