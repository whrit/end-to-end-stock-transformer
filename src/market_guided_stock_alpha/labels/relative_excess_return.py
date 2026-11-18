"""Pure calendar-aligned relative-excess return label construction.

The module accepts canonical pandas objects only. File formats, database
schemas, command-line parsing, and training are deliberately outside its
scope.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    normalize_date_values,
    normalize_security_code,
)


LABEL_START_OFFSET_TRADING_DAYS = 1
LABEL_END_OFFSET_TRADING_DAYS = 6
HOLDING_PERIOD_TRADING_DAYS = 5
PURGE_DEPENDENCY_OFFSET_TRADING_DAYS = 6
DEFAULT_BENCHMARK_CODE = "000905.SH"
DEFAULT_VALUE_COLUMN = "relative_excess_return_5d"
FORMULA = (
    "(stock_adj_open_t6 / stock_adj_open_t1) / "
    "(benchmark_open_t6 / benchmark_open_t1) - 1"
)


def formula_metadata(
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    value_column: str = DEFAULT_VALUE_COLUMN,
) -> Dict[str, Any]:
    """Return the stable research definition recorded with derived artifacts."""
    return {
        "name": "relative_excess_return_t1_t6",
        "formula": FORMULA,
        "benchmark": (
            "CSI500"
            if str(benchmark_code) == DEFAULT_BENCHMARK_CODE
            else str(benchmark_code)
        ),
        "benchmark_code": str(benchmark_code),
        "value_column": str(value_column),
        "holding_period_trading_days": HOLDING_PERIOD_TRADING_DAYS,
        "label_start_offset_trading_days": LABEL_START_OFFSET_TRADING_DAYS,
        "label_end_offset_trading_days": LABEL_END_OFFSET_TRADING_DAYS,
        "purge_dependency_offset_trading_days": (
            PURGE_DEPENDENCY_OFFSET_TRADING_DAYS
        ),
        "calendar_axis": "unified_open_trading_calendar",
        "stock_price_field": "adj_open",
        "benchmark_price_field": "open",
    }


def _normalized_dates(values: Any, label: str) -> pd.Series:
    return normalize_date_values(values, label)


def _open_calendar(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    required = {"calendar_date", "is_open"}
    if not required.issubset(calendar.columns):
        raise ValueError("calendar requires calendar_date and is_open")
    frame = calendar.loc[:, ["calendar_date", "is_open"]].copy()
    frame["calendar_date"] = _normalized_dates(
        frame["calendar_date"], "calendar_date"
    ).to_numpy()
    open_mask = pd.to_numeric(frame["is_open"], errors="coerce").eq(1)
    dates = pd.DatetimeIndex(
        frame.loc[open_mask, "calendar_date"].sort_values(kind="mergesort")
    )
    if len(dates) < LABEL_END_OFFSET_TRADING_DAYS + 1:
        raise ValueError("calendar has fewer than seven open trading dates")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("open trading calendar must be unique and increasing")
    return dates


def build_calendar_mapping(calendar: pd.DataFrame) -> pd.DataFrame:
    """Map every eligible anchor to t+1 and t+6 on one unified calendar."""
    dates = _open_calendar(calendar)
    end = LABEL_END_OFFSET_TRADING_DAYS
    start = LABEL_START_OFFSET_TRADING_DAYS
    return pd.DataFrame(
        {
            "trade_date": dates[:-end].to_numpy(),
            "label_date_t1": dates[start : -(end - start)].to_numpy(),
            "label_date_t6": dates[end:].to_numpy(),
        }
    )


def _stock_prices(stock_daily: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "sec_code", "adj_open"}
    if not required.issubset(stock_daily.columns):
        raise ValueError("stock_daily requires trade_date, sec_code, and adj_open")
    frame = stock_daily.loc[:, ["trade_date", "sec_code", "adj_open"]].copy()
    frame["trade_date"] = _normalized_dates(
        frame["trade_date"], "stock trade_date"
    ).to_numpy()
    frame["sec_code"] = [
        normalize_security_code(value, "stock_daily sec_code")
        for value in frame["sec_code"]
    ]
    if frame.duplicated(["trade_date", "sec_code"]).any():
        raise ValueError("stock_daily contains duplicate trade_date/sec_code keys")
    frame["adj_open"] = pd.to_numeric(frame["adj_open"], errors="coerce")
    valid = (
        np.isfinite(frame["adj_open"].to_numpy(dtype="float64"))
        & frame["adj_open"].gt(0.0).to_numpy()
    )
    return frame.loc[valid].reset_index(drop=True)


def _benchmark_prices(
    benchmark_daily: pd.DataFrame, benchmark_code: str
) -> pd.DataFrame:
    required = {"trade_date", "index_code", "open"}
    if not required.issubset(benchmark_daily.columns):
        raise ValueError("benchmark_daily requires trade_date, index_code, and open")
    frame = benchmark_daily.loc[:, ["trade_date", "index_code", "open"]].copy()
    frame["trade_date"] = _normalized_dates(
        frame["trade_date"], "benchmark trade_date"
    ).to_numpy()
    frame["index_code"] = [
        normalize_security_code(value, "benchmark_daily index_code")
        for value in frame["index_code"]
    ]
    canonical_benchmark = normalize_security_code(benchmark_code, "benchmark code")
    frame = frame.loc[frame["index_code"].eq(canonical_benchmark)].copy()
    if frame.empty:
        raise ValueError("benchmark code {} has no rows".format(benchmark_code))
    if frame.duplicated(["trade_date"]).any():
        raise ValueError("benchmark_daily contains duplicate benchmark dates")
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    valid = (
        np.isfinite(frame["open"].to_numpy(dtype="float64"))
        & frame["open"].gt(0.0).to_numpy()
    )
    return frame.loc[valid, ["trade_date", "open"]].reset_index(drop=True)


def _delisting_exits(stock_daily: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """sec_code, last_trade_date, kind, exit_price (last adjusted close x exit_factor)."""
    required = {"sec_code", "last_trade_date", "kind", "exit_factor"}
    if not required.issubset(delistings.columns):
        raise ValueError("delistings require sec_code, last_trade_date, kind, exit_factor")
    if "close" not in stock_daily.columns:
        raise ValueError("delisting exits need the adjusted close column in stock_daily")
    exits = delistings.loc[:, ["sec_code", "last_trade_date", "kind", "exit_factor"]].copy()
    exits["last_trade_date"] = _normalized_dates(exits["last_trade_date"], "delisting last_trade_date").to_numpy()
    if exits["sec_code"].duplicated().any():
        raise ValueError("delistings list a security more than once")
    factor = pd.to_numeric(exits["exit_factor"], errors="coerce")
    if not (np.isfinite(factor) & (factor > 0) & (factor <= 1)).all():
        raise ValueError("delisting exit_factor must be in (0, 1]")
    closes = stock_daily.loc[:, ["trade_date", "sec_code", "close"]].copy()
    closes["trade_date"] = _normalized_dates(closes["trade_date"], "stock trade_date").to_numpy()
    exits = exits.merge(
        closes.rename(columns={"trade_date": "last_trade_date", "close": "last_close"}),
        on=["sec_code", "last_trade_date"],
        how="left",
        validate="one_to_one",
    )
    if exits["last_close"].isna().any():
        raise ValueError(
            "no close on the last trade date for {}".format(
                ",".join(exits.loc[exits["last_close"].isna(), "sec_code"].head(5))
            )
        )
    exits["exit_price"] = exits["last_close"].astype("float64") * factor.astype("float64")
    return exits.drop(columns=["last_close", "exit_factor"])


def validate_output(
    output: pd.DataFrame, calendar: pd.DataFrame, value_column: str
) -> None:
    """Validate the public artifact schema and the t+6 future boundary."""
    if not isinstance(output.index, pd.MultiIndex) or output.index.nlevels != 2:
        raise ValueError("label output requires a two-level MultiIndex")
    if list(output.index.names) != ["trade_date", "sec_code"]:
        raise ValueError("label output index names changed")
    if list(output.columns) != [value_column]:
        raise ValueError("label output must contain exactly the value column")
    if output.index.has_duplicates or not output.index.is_monotonic_increasing:
        raise ValueError("label output index must be unique and sorted")
    values = output[value_column].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("label output contains NaN or infinite values")
    if len(output):
        open_dates = _open_calendar(calendar)
        maximum_anchor = open_dates[-(LABEL_END_OFFSET_TRADING_DAYS + 1)]
        output_dates = pd.DatetimeIndex(
            output.index.get_level_values("trade_date")
        ).normalize()
        if output_dates.max() > maximum_anchor:
            raise ValueError("label output exceeds the t+6 calendar boundary")
        eligible = set(open_dates[:-LABEL_END_OFFSET_TRADING_DAYS])
        if not set(output_dates).issubset(eligible):
            raise ValueError("label output contains an ineligible anchor date")


def build_relative_excess_label(
    stock_daily: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    calendar: pd.DataFrame,
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    value_column: str = DEFAULT_VALUE_COLUMN,
    delistings: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the finalized t+1-to-t+6 relative-excess label artifact.

    ``delistings`` (``sec_code, last_trade_date, kind, exit_factor``) adds the
    anchors whose stock stops trading between t+1 and t+6: the missing t+6 open
    becomes the last adjusted close times ``exit_factor`` (1 for mergers,
    1 - delta for performance delistings). Without it nothing changes.
    """
    if not isinstance(value_column, str) or not value_column.strip():
        raise ValueError("value_column must be a non-empty string")
    mapping = build_calendar_mapping(calendar)
    stock = _stock_prices(stock_daily)
    benchmark = _benchmark_prices(benchmark_daily, benchmark_code)

    benchmark_t1 = mapping.loc[:, ["trade_date", "label_date_t1"]].merge(
        benchmark.rename(
            columns={"trade_date": "label_date_t1", "open": "benchmark_open_t1"}
        ),
        on="label_date_t1",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    benchmark_pair = benchmark_t1.merge(
        mapping.loc[:, ["trade_date", "label_date_t6"]],
        on="trade_date",
        how="inner",
        validate="one_to_one",
        sort=False,
    ).merge(
        benchmark.rename(
            columns={"trade_date": "label_date_t6", "open": "benchmark_open_t6"}
        ),
        on="label_date_t6",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    benchmark_pair["benchmark_gross"] = (
        benchmark_pair["benchmark_open_t6"]
        / benchmark_pair["benchmark_open_t1"]
    )
    valid_benchmark_gross = (
        np.isfinite(benchmark_pair["benchmark_gross"].to_numpy(dtype="float64"))
        & benchmark_pair["benchmark_gross"].gt(0.0).to_numpy()
    )
    benchmark_pair = benchmark_pair.loc[valid_benchmark_gross].copy()

    stock_t1 = mapping.merge(
        stock.rename(
            columns={
                "trade_date": "label_date_t1",
                "adj_open": "stock_adj_open_t1",
            }
        ),
        on="label_date_t1",
        how="inner",
        validate="one_to_many",
        sort=False,
    )
    stock_pair = stock_t1.merge(
        stock.rename(
            columns={
                "trade_date": "label_date_t6",
                "adj_open": "stock_adj_open_t6",
            }
        ).loc[:, ["label_date_t6", "sec_code", "stock_adj_open_t6"]],
        on=["label_date_t6", "sec_code"],
        how="inner",
        validate="many_to_one",
        sort=False,
    ).merge(
        benchmark_pair.loc[:, ["trade_date", "benchmark_gross"]],
        on="trade_date",
        how="inner",
        validate="many_to_one",
        sort=False,
    )
    values = (
        stock_pair["stock_adj_open_t6"] / stock_pair["stock_adj_open_t1"]
    ) / stock_pair["benchmark_gross"] - 1.0
    finite = np.isfinite(values.to_numpy(dtype="float64"))
    result = stock_pair.loc[finite, ["trade_date", "sec_code"]].copy()
    result[value_column] = values.loc[finite].to_numpy(dtype="float64")
    delisting_counts = None
    if delistings is not None:
        exits = _delisting_exits(stock_daily, delistings)
        affected = stock_t1.merge(exits, on="sec_code", how="inner", sort=False)
        affected = affected.loc[
            (affected["label_date_t1"] <= affected["last_trade_date"])
            & (affected["last_trade_date"] < affected["label_date_t6"])
        ].merge(
            benchmark_pair.loc[:, ["trade_date", "benchmark_gross"]],
            on="trade_date",
            how="inner",
            validate="many_to_one",
            sort=False,
        )
        exit_values = (
            affected["exit_price"] / affected["stock_adj_open_t1"]
        ) / affected["benchmark_gross"] - 1.0
        keep = np.isfinite(exit_values.to_numpy(dtype="float64"))
        added = affected.loc[keep, ["trade_date", "sec_code"]].copy()
        added[value_column] = exit_values.loc[keep].to_numpy(dtype="float64")
        if len(added.merge(result, on=["trade_date", "sec_code"])):
            raise ValueError("delisting exit overlaps an anchor that has a t+6 open")
        result = pd.concat([result, added], ignore_index=True)
        delisting_counts = {
            "delisting_adjusted_count": int(len(added)),
            "delisting_adjusted_by_kind": {
                str(kind): int(count)
                for kind, count in affected.loc[keep, "kind"].value_counts().sort_index().items()
            },
            "delistings_supplied": int(len(exits)),
        }
    output = result.set_index(["trade_date", "sec_code"])[
        [value_column]
    ].sort_index()
    output[value_column] = output[value_column].astype("float64")
    validate_output(output, calendar, value_column)
    diagnostics = {
        "theoretical_anchor_count": int(len(mapping)),
        "benchmark_valid_anchor_count": int(len(benchmark_pair)),
        "output_row_count": int(len(output)),
        "output_date_count": (
            int(output.index.get_level_values("trade_date").nunique())
            if len(output)
            else 0
        ),
        "formula": formula_metadata(benchmark_code, value_column),
    }
    if delisting_counts is not None:
        diagnostics.update(delisting_counts)
    return output, diagnostics
