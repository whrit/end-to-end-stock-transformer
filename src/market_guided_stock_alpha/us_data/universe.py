"""Pure pandas assembly of the canonical US raw tables from Alpaca and OpenFIGI frames."""

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    normalize_security_code,
    validate_calendar,
    validate_daily_index,
    validate_daily_stock,
    validate_membership,
)
from market_guided_stock_alpha.us_data.openfigi import openfigi_ticker


COMMON_STOCK_TYPE = "Common Stock"
EQUITY_SECTOR = "Equity"
MARKET63_ETFS = ("SPY", "IJH", "IJR")
MARKET84_ETFS = MARKET63_ETFS + ("IWM",)
BENCHMARK_CODE = "SPY.US"
DOLLAR_VOLUME_WINDOW = 20
DEFAULT_MIN_PRICE = 5.0
DEFAULT_MIN_DOLLAR_VOLUME = 5000000.0
DAILY_STOCK_OUTPUT_COLUMNS = (
    "trade_date",
    "sec_code",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "adj_open",
    "raw_close",
    "raw_volume",
)
SECURITIES_COLUMNS = (
    "sec_code",
    "ticker",
    "alpaca_symbol",
    "name",
    "status",
    "exchange",
    "security_type2",
)
TABLE_FILES = {
    "daily_stock": "daily_stock.parquet",
    "daily_index_us_market63": "daily_index_us_market63.parquet",
    "daily_index_us_market84": "daily_index_us_market84.parquet",
    "membership": "membership.parquet",
    "calendar": "calendar.parquet",
    "securities": "securities.parquet",
}
_BAR_KEYS = ["alpaca_symbol", "trade_date"]
_BAR_VALUES = ("open", "high", "low", "close", "volume", "vwap")


def etf_code(symbol: str) -> str:
    return normalize_security_code("{}.US".format(symbol), "ETF index_code")


def build_securities(
    assets: pd.DataFrame, figi: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Join Alpaca assets with OpenFIGI rows and keep mapped US common stocks.

    A symbol listed both active and inactive keeps its active row; the dropped
    rows are counted. Every asset symbol must have an OpenFIGI row.
    """
    status_rank = assets["status"].map({"active": 0}).fillna(1)
    ordered = assets.assign(_rank=status_rank).sort_values(
        ["alpaca_symbol", "_rank"], kind="mergesort"
    )
    unique_assets = ordered.drop_duplicates("alpaca_symbol").drop(columns="_rank")
    if figi["ticker"].duplicated().any():
        raise DataQualityError(
            "OpenFIGI frame has duplicate tickers: {}".format(
                ",".join(sorted(set(figi.loc[figi["ticker"].duplicated(), "ticker"])))
            )
        )
    merged = unique_assets.merge(
        figi.rename(columns={"ticker": "alpaca_symbol", "name": "figi_name"}),
        on="alpaca_symbol",
        how="left",
        indicator=True,
    )
    unrequested = merged["_merge"] != "both"
    if unrequested.any():
        raise DataQualityError(
            "OpenFIGI frame lacks rows for Alpaca symbols: {}".format(
                ",".join(merged.loc[unrequested, "alpaca_symbol"].head(20))
            )
        )
    mapped = merged["share_class_figi"].notna()
    common = (
        mapped
        & (merged["security_type2"] == COMMON_STOCK_TYPE)
        & (merged["market_sector"] == EQUITY_SECTOR)
    )
    kept = merged.loc[common].copy()
    kept["sec_code"] = [
        normalize_security_code(value, "OpenFIGI share_class_figi")
        for value in kept["share_class_figi"]
    ]
    kept["ticker"] = [openfigi_ticker(value) for value in kept["alpaca_symbol"]]
    securities = (
        kept.loc[:, list(SECURITIES_COLUMNS)]
        .sort_values(["sec_code", "alpaca_symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    counts = {
        "assets_seen": int(len(assets)),
        "assets_active": int((assets["status"] == "active").sum()),
        "assets_inactive": int((assets["status"] != "active").sum()),
        "duplicate_symbol_rows": int(len(assets) - len(unique_assets)),
        "symbols": int(len(unique_assets)),
        "mapped": int(mapped.sum()),
        "unmapped": int((~mapped).sum()),
        "ambiguous_figi": int(
            merged["mapping_note"].fillna("").str.startswith("ambiguous").sum()
        ),
        "excluded_by_type": int((mapped & ~common).sum()),
        "common_stocks": int(common.sum()),
    }
    return securities, counts


def _bars(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [c for c in _BAR_KEYS + list(_BAR_VALUES) if c not in frame.columns]
    if missing:
        raise DataQualityError(
            "{} bars missing columns: {}".format(name, ",".join(missing))
        )
    output = frame.loc[:, _BAR_KEYS + list(_BAR_VALUES)].copy()
    output["alpaca_symbol"] = output["alpaca_symbol"].astype(str)
    output["trade_date"] = pd.to_datetime(output["trade_date"]).astype("datetime64[ns]")
    for column in _BAR_VALUES:
        output[column] = pd.to_numeric(output[column], errors="raise").astype("float64")
    return output


def merge_adjusted_and_raw(adjusted: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """One row per symbol/date with adjusted OHLCV plus raw close/volume/vwap."""
    raw_values = _bars(raw, "raw").loc[:, _BAR_KEYS + ["close", "volume", "vwap"]]
    raw_values = raw_values.rename(
        columns={"close": "raw_close", "volume": "raw_volume", "vwap": "raw_vwap"}
    )
    merged = _bars(adjusted, "adjusted").merge(
        raw_values, on=_BAR_KEYS, how="outer", indicator=True
    )
    partial = merged["_merge"] != "both"
    if partial.any():
        first = merged.loc[partial].iloc[0]
        raise DataQualityError(
            "adjusted and raw Alpaca bars disagree: {} on {} exists only in {} bars".format(
                first["alpaca_symbol"],
                first["trade_date"].date(),
                "adjusted" if first["_merge"] == "left_only" else "raw",
            )
        )
    return merged.drop(columns="_merge")


def _require_open_days(bars: pd.DataFrame, trading_days: pd.DatetimeIndex) -> None:
    closed = ~bars["trade_date"].isin(trading_days)
    if closed.any():
        first = bars.loc[closed].iloc[0]
        raise DataQualityError(
            "Alpaca bar for {} on {} is not an open trading day in range".format(
                first["alpaca_symbol"], first["trade_date"].date()
            )
        )


def build_daily_stock(bars: pd.DataFrame, securities: pd.DataFrame) -> pd.DataFrame:
    """Canonical daily_stock keyed by share-class FIGI, adjusted plus raw columns."""
    symbol_to_figi = dict(zip(securities["alpaca_symbol"], securities["sec_code"]))
    stock = bars.loc[bars["alpaca_symbol"].isin(symbol_to_figi)].copy()
    stock["sec_code"] = stock["alpaca_symbol"].map(symbol_to_figi)
    clash = stock.duplicated(["trade_date", "sec_code"], keep=False)
    if clash.any():
        first = stock.loc[clash].iloc[0]
        symbols = stock.loc[
            clash
            & (stock["trade_date"] == first["trade_date"])
            & (stock["sec_code"] == first["sec_code"]),
            "alpaca_symbol",
        ]
        raise DataQualityError(
            "share-class FIGI {} maps to several Alpaca symbols on {}: {}".format(
                first["sec_code"],
                first["trade_date"].date(),
                ",".join(sorted(symbols)),
            )
        )
    stock["adj_open"] = stock["open"]
    frame = (
        stock.loc[:, list(DAILY_STOCK_OUTPUT_COLUMNS)]
        .sort_values(["trade_date", "sec_code"], kind="mergesort")
        .reset_index(drop=True)
    )
    validated = validate_daily_stock(frame, require_adj_open=True)
    if not (validated["sec_code"].to_numpy() == frame["sec_code"].to_numpy()).all():
        raise AssertionError("validated daily stock order diverged from input order")
    for column in ("raw_close", "raw_volume"):
        validated[column] = frame[column].to_numpy(dtype="float64")
    return validated


def build_daily_index(
    bars: pd.DataFrame, etfs: Sequence[str], trading_days: pd.DatetimeIndex
) -> pd.DataFrame:
    """ETF market-state rows: adjusted open/close, amount = raw volume x raw vwap."""
    pieces = []
    for symbol in etfs:
        rows = bars.loc[bars["alpaca_symbol"] == symbol]
        missing = trading_days.difference(pd.DatetimeIndex(rows["trade_date"]))
        if len(missing):
            raise DataQualityError(
                "ETF {} lacks Alpaca bars on {} open trading days, first {}".format(
                    symbol, len(missing), missing[0].date()
                )
            )
        pieces.append(
            pd.DataFrame(
                {
                    "trade_date": rows["trade_date"].to_numpy(),
                    "index_code": etf_code(symbol),
                    "open": rows["open"].to_numpy(),
                    "close": rows["close"].to_numpy(),
                    "amount": (rows["raw_volume"] * rows["raw_vwap"]).to_numpy(),
                }
            )
        )
    return validate_daily_index(pd.concat(pieces, ignore_index=True))


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{} must be a number".format(name))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("{} must be finite and positive".format(name))
    return float(value)


def build_membership(
    daily_stock: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    min_price: float = DEFAULT_MIN_PRICE,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
    window: int = DOLLAR_VOLUME_WINDOW,
) -> pd.DataFrame:
    """Members: raw close >= min_price and mean raw dollar volume over the last
    ``window`` open trading days (day t included, every day present) >= threshold."""
    min_price = _positive(min_price, "min_price")
    min_dollar_volume = _positive(min_dollar_volume, "min_dollar_volume")
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive int")
    frame = daily_stock.assign(
        dollar_volume=daily_stock["raw_close"] * daily_stock["raw_volume"]
    )
    close = frame.pivot(index="trade_date", columns="sec_code", values="raw_close")
    dollar = frame.pivot(index="trade_date", columns="sec_code", values="dollar_volume")
    close = close.reindex(trading_days)
    dollar = dollar.reindex(trading_days).rolling(window, min_periods=window).mean()
    member = (close >= min_price) & (dollar >= min_dollar_volume)
    rows, columns = np.nonzero(member.to_numpy())
    return validate_membership(
        pd.DataFrame(
            {
                "trade_date": member.index[rows],
                "sec_code": member.columns[columns],
            }
        )
    )


def build_calendar(start: object, end: object, trading_days: Iterable) -> pd.DataFrame:
    """Every calendar day from start..end with is_open from the trading calendar."""
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    open_days = pd.DatetimeIndex(list(trading_days)).normalize()
    return validate_calendar(
        pd.DataFrame({"calendar_date": days, "is_open": days.isin(open_days).astype("int8")})
    )


def build_us_tables(
    securities: pd.DataFrame,
    adjusted_bars: pd.DataFrame,
    raw_bars: pd.DataFrame,
    trading_days: Iterable,
    start: object,
    end: object,
    min_price: float = DEFAULT_MIN_PRICE,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOLUME,
    membership: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Assemble and validate every canonical raw table keyed by ``TABLE_FILES``.

    ``membership`` replaces the price/dollar-volume filter (the point-in-time
    S&P 500 universe); it is kept only on days the stock has a bar.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start > end:
        raise DataQualityError("start date is after end date")
    calendar = build_calendar(start, end, trading_days)
    open_days = pd.DatetimeIndex(
        calendar.loc[calendar["is_open"] == 1, "calendar_date"]
    )
    bars = merge_adjusted_and_raw(adjusted_bars, raw_bars)
    _require_open_days(bars, open_days)
    daily_stock = build_daily_stock(bars, securities)
    if membership is None:
        members = build_membership(daily_stock, open_days, min_price, min_dollar_volume)
    else:
        members = validate_membership(
            membership.merge(daily_stock.loc[:, ["trade_date", "sec_code"]], on=["trade_date", "sec_code"])
        )
    return {
        "daily_stock": daily_stock,
        "daily_index_us_market63": build_daily_index(bars, MARKET63_ETFS, open_days),
        "daily_index_us_market84": build_daily_index(bars, MARKET84_ETFS, open_days),
        "membership": members,
        "calendar": calendar,
        "securities": securities.loc[:, list(SECURITIES_COLUMNS)].reset_index(drop=True),
    }
