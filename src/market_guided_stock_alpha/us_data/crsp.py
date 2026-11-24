"""Point-in-time US universe from a CRSP CIZ daily stock file (WRDS dsf_v2 export).

Identity is the PERMNO (never reused), written ``NNNNN.CRSP``. Each trading day
the universe is the ``top_n`` eligible common stocks by market cap with a raw
close of at least ``min_price``. Eligibility uses that day's CRSP attributes
(EQTY/COM/NS, US-incorporated, NYSE/AMEX/Nasdaq primary listing, a trade price),
so nothing known only later decides who is in.

Prices: raw OHLC scaled by a total-return index built from ``DlyRet``, so splits
and dividends are both adjusted. A missing ``DlyRet`` counts as 0, because CRSP's
next return spans the gap from the last valid price. Volume is split-adjusted
(``DlyVol x DlyCumFacShr``). CRSP has no VWAP; ``vwap`` is the typical price
(high + low + close) / 3.

Delistings: the ``DlyDelFlg = Y`` row carries CRSP's delisting return, so
``exit_factor`` is the total-return growth from the last trade close to the
delisting value. A missing delisting return falls back to Shumway's delta for
performance delistings (``GDR``) and to no change otherwise.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import DataQualityError, validate_daily_stock, validate_membership
from market_guided_stock_alpha.us_data.sp500 import (
    DELISTING_COLUMNS,
    SHUMWAY_DELTA_NASDAQ,
    SHUMWAY_DELTA_NYSE_AMEX,
)
from market_guided_stock_alpha.us_data.universe import (
    DAILY_STOCK_OUTPUT_COLUMNS,
    MARKET63_ETFS,
    MARKET84_ETFS,
    build_calendar,
    build_daily_index,
)


DEFAULT_TOP_N = 1500
DEFAULT_MIN_PRICE = 5.0
# Fixed PERMNOs, verified in the 2000-2025 export: a ticker can be reused, a PERMNO cannot.
ETF_PERMNOS = {84398: "SPY", 88218: "IJH", 88229: "IJR", 88222: "IWM"}
ELIGIBLE = (("SecurityType", "EQTY"), ("SecuritySubType", "COM"), ("ShareType", "NS"), ("USIncFlg", "Y"))
EXCHANGES = ("N", "A", "Q")  # NYSE, NYSE American (AMEX), Nasdaq
NYSE_AMEX = ("N", "A")
DELISTING_KINDS = {"MER": "merger", "GDR": "performance", "GLI": "liquidation", "GEX": "exchange"}
COLUMNS = (
    "PERMNO", "DlyCalDt", "Ticker", "IssuerNm", "PrimaryExch", "SecurityType", "SecuritySubType", "ShareType",
    "USIncFlg", "DelActionType", "DlyDelFlg", "DlyPrcFlg", "DlyRet", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose",
    "DlyVol", "DlyCap", "DlyCumFacShr",
)
SECURITIES_COLUMNS = ("sec_code", "permno", "ticker", "name", "exchange", "first_trade_date", "last_trade_date")


def crsp_code(permno: int) -> str:
    return "{:05d}.CRSP".format(int(permno))


def _eligible(rows: pd.DataFrame) -> pd.Series:
    mask = rows["PrimaryExch"].isin(EXCHANGES)
    for column, value in ELIGIBLE:
        mask &= rows[column] == value
    return mask


def _traded(rows: pd.DataFrame) -> pd.Series:
    prices = rows[["DlyOpen", "DlyHigh", "DlyLow", "DlyClose"]]
    return (rows["DlyPrcFlg"] == "TR") & (rows["DlyDelFlg"] == "N") & (prices > 0).all(axis=1)


def _top(pool: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if type(top_n) is not int or top_n <= 0:
        raise ValueError("top_n must be a positive int")
    pool = pool.sort_values(["DlyCalDt", "DlyCap", "PERMNO"], ascending=[True, False, True], kind="mergesort")
    return pool.loc[pool.groupby("DlyCalDt").cumcount() < top_n, ["PERMNO", "DlyCalDt"]].reset_index(drop=True)


def rank_members(rows: pd.DataFrame, top_n: int, min_price: float) -> pd.DataFrame:
    """PERMNO and DlyCalDt of each day's top_n eligible stocks by market cap."""
    pool = rows.loc[_eligible(rows) & _traded(rows) & (rows["DlyClose"] >= min_price) & (rows["DlyCap"] > 0)]
    return _top(pool.loc[:, ["PERMNO", "DlyCalDt", "DlyCap"]], top_n)


def _delistings(rows: pd.DataFrame, traded: pd.Series, growth: pd.Series) -> pd.DataFrame:
    exits = []
    delisted = rows.loc[rows["DlyDelFlg"] == "Y"]
    last_trade = rows.loc[traded].groupby("PERMNO").tail(1).set_index("PERMNO")
    action = rows.groupby("PERMNO")["DelActionType"].last()  # absent on the delisting row itself
    for index, row in delisted.iterrows():
        permno = row["PERMNO"]
        if permno not in last_trade.index:
            continue  # never traded in the window: not in daily_stock
        base = growth.loc[last_trade.loc[permno, "_row"]]
        kind = DELISTING_KINDS.get(action.loc[permno], "other")
        exchange = last_trade.loc[permno, "PrimaryExch"]  # the delisting row reads X
        # Growth from the last trade close: returns on any no-trade days, then the delisting return.
        factor = growth.loc[index] / base
        delta = 0.0
        if pd.isna(row["DlyRet"]) and kind == "performance":
            delta = SHUMWAY_DELTA_NYSE_AMEX if exchange in NYSE_AMEX else SHUMWAY_DELTA_NASDAQ
            factor *= 1.0 - delta
        exits.append(
            {
                "sec_code": crsp_code(permno),
                "last_trade_date": last_trade.loc[permno, "DlyCalDt"],
                "kind": kind,
                "exchange": exchange,
                "delta": delta,
                "exit_factor": float(factor),
                "crsp_return": pd.notna(row["DlyRet"]),
            }
        )
    return pd.DataFrame(exits, columns=list(DELISTING_COLUMNS) + ["crsp_return"])


def build_crsp_tables(
    rows: pd.DataFrame,
    start: Any,
    end: Any,
    top_n: int = DEFAULT_TOP_N,
    min_price: float = DEFAULT_MIN_PRICE,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int]]:
    """Canonical raw tables (``TABLE_FILES`` keys plus ``delistings``) and counts from CRSP rows.

    ``rows`` holds the ``COLUMNS`` of every row of the stocks to keep and of the
    ETFs in ``ETF_PERMNOS``; repeated rows (one per same-day distribution) collapse.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    rows = rows.loc[rows["DlyCalDt"].between(start, end)]
    rows = rows.sort_values(["PERMNO", "DlyCalDt"], kind="mergesort").drop_duplicates(["PERMNO", "DlyCalDt"])
    rows = rows.reset_index(drop=True)
    rows["_row"] = rows.index
    traded = _traded(rows)
    # A missing DlyRet is 0: CRSP's next return already spans the gap from the last valid price.
    growth = (1.0 + rows["DlyRet"].fillna(0.0)).groupby(rows["PERMNO"]).cumprod()
    trading_days = pd.DatetimeIndex(np.unique(rows.loc[rows["DlyDelFlg"] == "N", "DlyCalDt"]))

    bars = rows.loc[traded].copy()
    factor = growth.loc[bars.index] / bars["DlyClose"]
    bars = bars.assign(
        trade_date=bars["DlyCalDt"],
        open=bars["DlyOpen"] * factor,
        high=bars["DlyHigh"] * factor,
        low=bars["DlyLow"] * factor,
        close=growth.loc[bars.index],
        vwap=(bars["DlyHigh"] + bars["DlyLow"] + bars["DlyClose"]) / 3.0 * factor,
        volume=bars["DlyVol"] * bars["DlyCumFacShr"],
        raw_close=bars["DlyClose"],
        raw_volume=bars["DlyVol"],
        raw_vwap=bars["DlyClose"],  # index amount = raw close x raw volume
    )
    bars["adj_open"] = bars["open"]
    is_etf = bars["PERMNO"].isin(ETF_PERMNOS)
    etf_bars = bars.loc[is_etf].assign(alpaca_symbol=bars.loc[is_etf, "PERMNO"].map(ETF_PERMNOS))

    stock = bars.loc[~is_etf].assign(sec_code=bars.loc[~is_etf, "PERMNO"].map(crsp_code))
    frame = stock.loc[:, list(DAILY_STOCK_OUTPUT_COLUMNS)].sort_values(["trade_date", "sec_code"], kind="mergesort")
    daily_stock = validate_daily_stock(frame.reset_index(drop=True), require_adj_open=True)
    for column in ("raw_close", "raw_volume"):
        daily_stock[column] = frame[column].to_numpy(dtype="float64")

    members = rank_members(rows.loc[~rows["PERMNO"].isin(ETF_PERMNOS)], top_n, min_price)
    membership = validate_membership(
        pd.DataFrame({"trade_date": members["DlyCalDt"], "sec_code": members["PERMNO"].map(crsp_code)})
    )
    stock_rows = rows.loc[~rows["PERMNO"].isin(ETF_PERMNOS)]
    delistings = _delistings(stock_rows, traded.loc[stock_rows.index], growth)
    span = stock.groupby("PERMNO")["trade_date"].agg(["min", "max"])
    last = stock_rows.groupby("PERMNO")[["Ticker", "IssuerNm", "PrimaryExch"]].last().loc[span.index]
    securities = pd.DataFrame(
        {
            "sec_code": [crsp_code(p) for p in span.index],
            "permno": span.index.to_numpy(dtype="int64"),
            "ticker": last["Ticker"].to_numpy(),
            "name": last["IssuerNm"].to_numpy(),
            "exchange": last["PrimaryExch"].to_numpy(),
            "first_trade_date": span["min"].to_numpy(),
            "last_trade_date": span["max"].to_numpy(),
        }
    )
    tables = {
        "daily_stock": daily_stock,
        "daily_index_us_market63": build_daily_index(etf_bars, MARKET63_ETFS, trading_days),
        "daily_index_us_market84": build_daily_index(etf_bars, MARKET84_ETFS, trading_days),
        "membership": membership,
        "calendar": build_calendar(start, end, trading_days),
        "securities": securities,
        "delistings": delistings.drop(columns="crsp_return"),
    }
    counts = {
        "trading_days": len(trading_days),
        "daily_stock_securities": int(len(span)),
        "membership_securities": int(membership["sec_code"].nunique()),
        "members_per_day_min": int(membership.groupby("trade_date").size().min()),
        "delistings": int(len(delistings)),
        "delistings_with_crsp_return": int(delistings["crsp_return"].sum()),
    }
    for kind, count in delistings["kind"].value_counts().items():
        counts["delisted_{}".format(kind)] = int(count)
    return tables, counts


def read_crsp_rows(path: Path, start: Any, end: Any, top_n: int, min_price: float) -> pd.DataFrame:
    """Rows of every stock ever in the universe plus the ETFs, read in two passes
    so only those securities (not the whole 50M-row file) reach memory."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="parquet")
    window = (pc.field("DlyCalDt") >= pd.Timestamp(start).date()) & (pc.field("DlyCalDt") <= pd.Timestamp(end).date())
    eligible = pc.field("PrimaryExch").isin(list(EXCHANGES)) & (pc.field("DlyPrcFlg") == "TR") & (pc.field("DlyDelFlg") == "N")
    for column, value in ELIGIBLE:
        eligible &= pc.field(column) == value
    eligible &= (pc.field("DlyClose") >= min_price) & (pc.field("DlyCap") > 0)
    pool = dataset.to_table(columns=["PERMNO", "DlyCalDt", "DlyCap"], filter=window & eligible).to_pandas()
    keep = set(_top(pool, top_n)["PERMNO"].tolist()) | set(ETF_PERMNOS)
    del pool
    table = dataset.to_table(columns=list(COLUMNS), filter=window & pc.field("PERMNO").isin(sorted(keep)))
    rows = table.to_pandas(strings_to_categorical=True)
    rows["DlyCalDt"] = pd.to_datetime(rows["DlyCalDt"])
    return rows


def convert_export(source: Path, destination: Path, columns: Iterable[str] = COLUMNS + ("DisSeqNbr",)) -> int:
    """Stream the WRDS CSV (``.csv`` or the ``.zip`` WRDS returns) into Parquet with only ``columns``."""
    import zipfile

    import pyarrow as pa
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq

    numeric = {"DlyRet", "DlyOpen", "DlyHigh", "DlyLow", "DlyClose", "DlyVol", "DlyCap", "DlyCumFacShr"}
    types = {
        name: pa.int64() if name == "PERMNO" else pa.date32() if name == "DlyCalDt" else pa.float64() if name in numeric else pa.string()
        for name in columns
    }
    handle: Optional[Any] = None
    if Path(source).suffix == ".zip":
        archive = zipfile.ZipFile(str(source))
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise DataQualityError("expected one CSV in {}, found {}".format(source, names))
        handle = archive.open(names[0])
    reader = pcsv.open_csv(
        handle if handle is not None else str(source),
        read_options=pcsv.ReadOptions(block_size=64 << 20),
        convert_options=pcsv.ConvertOptions(include_columns=list(types), column_types=types, strings_can_be_null=True),
    )
    writer, count = None, 0
    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])
            writer = writer or pq.ParquetWriter(str(destination), table.schema, compression="zstd")
            writer.write_table(table)
            count += len(table)
    finally:
        if writer is not None:
            writer.close()
    return count
