"""Point-in-time S&P 500 universe (US v2): who was a member on each day, which
company each ticker meant at the time, and how delisted members exited.

Identity. The membership history lists tickers as they were on each date, and
tickers get renamed (FB -> META) and reused (FB is now an ETF). Each membership
span ("episode") is therefore priced with Alpaca bars requested ``asof`` its last
member day, which follows that company through later renames and never returns
a later holder of the ticker. Episodes with identical raw price histories are
the same company.

FIGIs. A company still in the index on the last day maps through its current
ticker. A removed company keeps a FIGI only when its last ticker is unused today
(OpenFIGI then still knows the delisted security) or still trades as the same
company. Anything else gets ``TICKER.US.YYYYMMDD`` (last ticker, last bar date)
instead of a guessed FIGI.

Delistings (companies whose bars stop before the last day) exit at their last
adjusted close: unchanged for mergers, times (1 - delta) for performance
delistings (Shumway 1997; Shumway & Warther 1999).
"""

import csv
import hashlib
import io
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    normalize_security_code,
    validate_membership,
)
from market_guided_stock_alpha.progress import progress


SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
ACCEPTED_SECURITY_TYPES = ("Common Stock", "REIT")
SHUMWAY_DELTA_NYSE_AMEX = 0.30
SHUMWAY_DELTA_NASDAQ = 0.55
NYSE_AMEX_EXCHANGES = ("NYSE", "AMEX", "ARCA", "BATS")  # Alpaca exchange names
PERFORMANCE_MAX_RAW_CLOSE = 1.0
PERFORMANCE_DRAWDOWN_SESSIONS = 20
PERFORMANCE_DRAWDOWN = 0.5
MIN_IDENTITY_OVERLAP_DAYS = 5  # ticker still prices the same company (FIGI check)
MIN_SAME_COMPANY_DAYS = 20  # two membership spans are one company: shared days...
SAME_COMPANY_AGREEMENT = 0.99  # ...and the share of them with identical raw closes
DELISTING_COLUMNS = ("sec_code", "last_trade_date", "kind", "exchange", "delta", "exit_factor")

FetchBars = Callable[[Sequence[str], Optional[pd.Timestamp], str], pd.DataFrame]
MapFigi = Callable[[Sequence[str], bool], pd.DataFrame]


def load_history(source: Any) -> Tuple[pd.DataFrame, str]:
    """Read the fja05680/sp500 history (URL or path); returns (frame, sha256)."""
    text = str(source)
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text, timeout=60) as response:
            payload = response.read()
    else:
        payload = Path(text).read_bytes()
    return parse_history(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def parse_history(text: str) -> pd.DataFrame:
    """``date,tickers`` rows (full member list per change date) -> sorted frame."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or [cell.strip().lower() for cell in rows[0][:2]] != ["date", "tickers"]:
        raise DataQualityError("S&P 500 history must start with a date,tickers header")
    dates, members = [], []
    for row in rows[1:]:
        if not row:
            continue
        tickers = tuple(sorted({t.strip().upper() for t in row[1].split(",") if t.strip()}))
        if not tickers:
            raise DataQualityError("S&P 500 history row {} lists no tickers".format(row[0]))
        dates.append(pd.Timestamp(row[0]).normalize())
        members.append(tickers)
    frame = pd.DataFrame({"date": dates, "tickers": members})
    if frame.empty or not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise DataQualityError("S&P 500 history dates must be unique and increasing")
    return frame


def daily_members(history: pd.DataFrame, trading_days: Sequence[Any]) -> pd.DataFrame:
    """(trade_date, ticker): the latest history row on or before each trading day."""
    days = pd.DatetimeIndex(trading_days).sort_values()
    if len(days) == 0:
        raise DataQualityError("no trading days")
    if days[0] < history["date"].iloc[0]:
        raise DataQualityError("trading days start before the S&P 500 history")
    row = pd.DatetimeIndex(history["date"]).searchsorted(days, side="right") - 1
    exploded = (
        history.reset_index(drop=True)
        .rename_axis("row")
        .reset_index()
        .explode("tickers")
        .rename(columns={"tickers": "ticker"})
    )
    frame = pd.DataFrame({"trade_date": days, "row": row}).merge(
        exploded.loc[:, ["row", "ticker"]], on="row"
    )
    return frame.loc[:, ["trade_date", "ticker"]].sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def episodes(members: pd.DataFrame, trading_days: Sequence[Any]) -> pd.DataFrame:
    """Runs of consecutive member days per ticker, with an ``episode`` id per
    member row: (trade_date, ticker, episode, first_date, last_date)."""
    days = pd.DatetimeIndex(trading_days).sort_values()
    position = days.get_indexer(pd.DatetimeIndex(members["trade_date"]))
    frame = members.assign(_pos=position).sort_values(["ticker", "_pos"]).reset_index(drop=True)
    new_run = (frame["ticker"] != frame["ticker"].shift()) | (frame["_pos"] != frame["_pos"].shift() + 1)
    frame["episode"] = new_run.cumsum().astype("int64") - 1
    spans = frame.groupby("episode")["trade_date"].agg(first_date="min", last_date="max")
    return frame.drop(columns="_pos").merge(spans, left_on="episode", right_index=True)


def _companies(raw: pd.DataFrame) -> pd.Series:
    """episode -> company id: episodes whose raw closes agree on at least
    ``SAME_COMPANY_AGREEMENT`` of their shared days (and share at least
    ``MIN_IDENTITY_OVERLAP_DAYS``) are the same company, transitively.

    Not a hash of the whole history: the same company read as of different
    dates can cover different ranges (Dr Pepper Snapple listed as KDP).
    """
    rows = raw.loc[:, ["episode", "trade_date", "close"]].copy()
    rows["price"] = np.round(rows["close"].to_numpy(dtype="float64"), 4)
    # Candidate pairs: episodes printing the same price on the same day.
    hits = rows.merge(rows, on=["trade_date", "price"])
    hits = hits.loc[hits["episode_x"] < hits["episode_y"]]
    pairs = hits.groupby(["episode_x", "episode_y"]).size()
    parent = {int(e): int(e) for e in rows["episode"].unique()}

    def root(e: int) -> int:
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    days = rows.groupby("episode")["trade_date"].apply(set)
    for (a, b), matched in pairs.items():
        if matched < MIN_SAME_COMPANY_DAYS:
            continue
        shared = len(days[a] & days[b])
        if matched >= SAME_COMPANY_AGREEMENT * shared:
            parent[root(int(a))] = root(int(b))
    return pd.Series({e: root(e) for e in parent}, name="company")


def _same_prices(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    joined = left.loc[:, ["trade_date", "close"]].merge(
        right.loc[:, ["trade_date", "close"]], on="trade_date"
    )
    return len(joined) >= MIN_IDENTITY_OVERLAP_DAYS and bool(
        np.allclose(joined["close_x"], joined["close_y"], rtol=0.0, atol=1e-4)
    )


def _accepted(row: Optional[pd.Series]) -> Optional[str]:
    if row is None or pd.isna(row.get("share_class_figi")):
        return None
    if row.get("security_type2") not in ACCEPTED_SECURITY_TYPES:
        return None
    return str(row["share_class_figi"])


def _by_ticker(frame: pd.DataFrame) -> Dict[str, pd.Series]:
    return {str(row["ticker"]): row for _, row in frame.iterrows()}


def _fetch_by_asof(
    fetch_bars: FetchBars, spans: pd.DataFrame, key: str, adjustment: str, desc: str
) -> pd.DataFrame:
    """Bars for each span's ticker as of its last_date, labelled with ``key``."""
    parts = []
    groups = list(spans.groupby("last_date"))
    for asof, group in progress(groups, total=len(groups), desc=desc, unit="date", leave=False):
        bars = fetch_bars(sorted(set(group["ticker"])), pd.Timestamp(asof), adjustment)
        labels = group.loc[:, ["ticker", key]].rename(columns={"ticker": "alpaca_symbol"})
        parts.append(bars.merge(labels, on="alpaca_symbol"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_sp500_universe(
    history: pd.DataFrame,
    trading_days: Sequence[Any],
    fetch_bars: FetchBars,
    map_figi: MapFigi,
    assets: pd.DataFrame,
) -> Dict[str, Any]:
    """Assemble the v2 stock universe.

    ``fetch_bars(symbols, asof, adjustment)`` returns Alpaca bar rows
    (``alpaca_symbol, trade_date, open, high, low, close, volume, vwap``);
    ``asof=None`` means the literal symbol: every bar any holder printed under
    it. ``map_figi(tickers, include_unlisted)``
    returns OpenFIGI rows (``map_tickers`` columns). ``assets`` is the Alpaca
    asset list, used only for the exchange of delisted companies.

    Returns ``adjusted``/``raw`` stock bars keyed by ``sec_code`` (in
    ``alpaca_symbol``), ``membership``, ``securities``, ``delistings`` and
    ``counts``.
    """
    days = pd.DatetimeIndex(trading_days).sort_values()
    last_day = days[-1]
    member_rows = episodes(daily_members(history, days), days)
    spans = member_rows.drop_duplicates("episode").loc[:, ["episode", "ticker", "first_date", "last_date"]]

    # 1. Company identity: raw bars per episode, read as of its last member day.
    raw_by_episode = _fetch_by_asof(fetch_bars, spans, "episode", "raw", "S&P members (raw, as of)")
    companies = _companies(raw_by_episode) if len(raw_by_episode) else pd.Series(dtype="int64")
    priced = spans.loc[spans["episode"].isin(companies.index)].copy()
    priced["company"] = priced["episode"].map(companies)
    # The latest span names the company (its ticker decides the FIGI); the span
    # with the longest history prices it.
    representative = (
        priced.sort_values(["company", "last_date", "episode"]).groupby("company").tail(1).set_index("company")
    )
    lengths = raw_by_episode.groupby("episode").size()
    widest = (
        priced.assign(bars=priced["episode"].map(lengths))
        .sort_values(["company", "bars", "last_date", "episode"])
        .groupby("company").tail(1).set_index("company")
    )
    company_raw = raw_by_episode.loc[raw_by_episode["episode"].isin(widest["episode"])].copy()
    company_raw["company"] = company_raw["episode"].map(pd.Series(widest.index, index=widest["episode"]))

    # 2. FIGIs: current members by ticker; removed ones only when verifiable.
    #    ``literal`` = every bar ever printed under the ticker (any holder).
    last_bar = company_raw.groupby("company")["trade_date"].max()
    current = representative["last_date"] == last_day
    removed_tickers = sorted(set(representative.loc[~current, "ticker"]))
    listed = _by_ticker(map_figi(sorted(set(representative["ticker"])), False))
    literal = (
        fetch_bars(removed_tickers, None, "raw") if removed_tickers else pd.DataFrame(columns=["alpaca_symbol", "trade_date"])
    )
    decisions: Dict[str, str] = {}
    for company, rep in representative.loc[~current].iterrows():
        ticker = str(rep["ticker"])
        printed = literal.loc[literal["alpaca_symbol"] == ticker]
        if last_bar[company] < last_day:
            # Delisted: its FIGI is trustworthy only if nobody reused the ticker since.
            decisions[company] = (
                "fallback_ticker_reused" if (printed["trade_date"] > last_bar[company]).any() else "unlisted"
            )
        elif (printed["trade_date"] == last_day).any() and _same_prices(
            printed, company_raw.loc[company_raw["company"] == company]
        ):
            decisions[company] = "listed"  # left the index, still trades under this ticker
        else:
            decisions[company] = "fallback_renamed"  # trades on under a ticker we do not know
    unlisted_tickers = sorted({str(representative.at[c, "ticker"]) for c, d in decisions.items() if d == "unlisted"})
    unlisted = _by_ticker(map_figi(unlisted_tickers, True)) if unlisted_tickers else {}
    # company -> (FIGI or None, how it was decided, the OpenFIGI row it came from)
    identity: Dict[str, Tuple[Optional[str], str, Optional[pd.Series]]] = {}
    for company, rep in representative.iterrows():
        ticker = str(rep["ticker"])
        decision = "current" if current[company] else decisions[company]
        row = {"current": listed, "listed": listed, "unlisted": unlisted}.get(decision, {}).get(ticker)
        figi = _accepted(row)
        if decision.startswith("fallback"):
            identity[company] = (None, decision, None)
        elif figi is None:
            identity[company] = (None, "fallback_unmapped", None)
        else:
            kind = {"current": "figi_current", "listed": "figi_verified", "unlisted": "figi_delisted"}[decision]
            identity[company] = (figi, kind, row)
    figis = pd.Series({c: value[0] for c, value in identity.items() if value[0]})
    for company in figis[figis.duplicated(keep=False)].index:
        identity[company] = (None, "fallback_duplicate_figi", None)
    codes = {}
    for company, (figi, _kind, _row) in identity.items():
        ticker = str(representative.at[company, "ticker"])
        codes[company] = normalize_security_code(
            figi if figi else "{}.US.{:%Y%m%d}".format(ticker, last_bar[company]),
            "S&P 500 company {}".format(ticker),
        )
    clash = pd.Series(codes)
    clash = clash[clash.duplicated(keep=False)]
    if len(clash):
        raise DataQualityError(
            "several S&P 500 companies share code {}: spans {}".format(
                clash.iloc[0],
                ", ".join(
                    "{}..{} {}".format(r.first_date.date(), r.last_date.date(), r.ticker)
                    for r in priced.loc[priced["company"].isin(clash[clash == clash.iloc[0]].index)].itertuples()
                ),
            )
        )

    # 3. Adjusted bars per company, as of its widest-history span.
    adjusted = _fetch_by_asof(
        fetch_bars, widest.reset_index(), "company", "all", "S&P members (adjusted, as of)"
    )
    adjusted["alpaca_symbol"] = adjusted.pop("company").map(codes)
    raw = company_raw.drop(columns=["episode"]).copy()
    raw["alpaca_symbol"] = raw.pop("company").map(codes)

    # 4. Membership: member that day and the company is priced.
    episode_code = priced.set_index("episode")["company"].map(codes)
    membership = member_rows.loc[member_rows["episode"].isin(episode_code.index)].copy()
    membership["sec_code"] = membership["episode"].map(episode_code)
    membership = validate_membership(
        membership.loc[:, ["trade_date", "sec_code"]].drop_duplicates()
    )

    # 5. Securities.
    tickers = priced.sort_values("first_date").groupby("company")["ticker"].agg(lambda t: "|".join(dict.fromkeys(t)))
    spans_by_company = priced.groupby("company").agg(first_member=("first_date", "min"), last_member=("last_date", "max"))
    source = [identity[c][2] for c in representative.index]
    securities = pd.DataFrame(
        {
            "sec_code": [codes[c] for c in representative.index],
            "ticker": representative["ticker"].to_numpy(),
            "alpaca_symbol": representative["ticker"].to_numpy(),
            "name": [None if row is None else row.get("name") for row in source],
            "status": ["active" if last_bar[c] == last_day else "delisted" for c in representative.index],
            "exchange": None,
            "security_type2": [None if row is None else row.get("security_type2") for row in source],
            "identity": [identity[c][1] for c in representative.index],
            "asof": representative["last_date"].to_numpy(),
            "sp500_tickers": [tickers[c] for c in representative.index],
            "first_member_date": [spans_by_company.at[c, "first_member"] for c in representative.index],
            "last_member_date": [spans_by_company.at[c, "last_member"] for c in representative.index],
        }
    )
    exchange = (
        assets.loc[assets["status"] != "active"].drop_duplicates("alpaca_symbol").set_index("alpaca_symbol")["exchange"]
        if len(assets) else pd.Series(dtype=object)
    )
    securities["exchange"] = [
        exchange.get(t) if s == "delisted" else None for t, s in zip(securities["ticker"], securities["status"])
    ]
    securities = securities.sort_values("sec_code").reset_index(drop=True)

    # 6. Delistings (Gap B).
    delistings = classify_delistings(raw, adjusted, securities, last_day)

    kinds = pd.Series([value[1] for value in identity.values()]).value_counts()
    per_day = membership.groupby("trade_date").size()
    counts = {
        "sp500_history_last_date": str(history["date"].iloc[-1].date()),
        "trading_days_after_history": int((days > history["date"].iloc[-1]).sum()),
        "episodes": int(len(spans)),
        "episodes_without_bars": int(len(spans) - len(priced)),
        "companies": int(len(representative)),
        "companies_with_several_tickers": int((tickers.str.contains(r"\|")).sum()),
        "identity": {str(k): int(v) for k, v in kinds.items()},
        "members_per_day_min": int(per_day.min()),
        "members_per_day_median": int(per_day.median()),
        "members_per_day_max": int(per_day.max()),
        "delisted_merger": int((delistings["kind"] == "merger").sum()),
        "delisted_performance": int((delistings["kind"] == "performance").sum()),
        "delisted_exchange_unknown": int(delistings["exchange"].isna().sum()),
    }
    return {
        "adjusted": adjusted.drop(columns=["ticker"], errors="ignore"),
        "raw": raw.drop(columns=["ticker"], errors="ignore"),
        "membership": membership,
        "securities": securities,
        "delistings": delistings,
        "counts": counts,
    }


def classify_delistings(
    raw: pd.DataFrame, adjusted: pd.DataFrame, securities: pd.DataFrame, last_day: pd.Timestamp
) -> pd.DataFrame:
    """One row per company whose bars stop before ``last_day``.

    performance: last raw close < $1, or the last adjusted close is below half
    the highest adjusted close of its final 20 sessions; otherwise merger.
    delta: 30% NYSE/AMEX, 55% Nasdaq or unknown exchange (the conservative one).
    """
    rows: List[Dict[str, Any]] = []
    exchange = securities.set_index("sec_code")["exchange"]
    for code, bars in adjusted.groupby("alpaca_symbol"):
        bars = bars.sort_values("trade_date")
        last = bars["trade_date"].iloc[-1]
        if last >= last_day:
            continue
        raw_close = raw.loc[(raw["alpaca_symbol"] == code) & (raw["trade_date"] == last), "close"]
        tail = bars["close"].iloc[-PERFORMANCE_DRAWDOWN_SESSIONS:]
        performance = (
            (len(raw_close) and float(raw_close.iloc[0]) < PERFORMANCE_MAX_RAW_CLOSE)
            or float(tail.iloc[-1]) < PERFORMANCE_DRAWDOWN * float(tail.max())
        )
        venue = exchange.get(code)
        venue = None if venue is None or pd.isna(venue) else str(venue)
        delta = SHUMWAY_DELTA_NYSE_AMEX if venue in NYSE_AMEX_EXCHANGES else SHUMWAY_DELTA_NASDAQ
        rows.append(
            {
                "sec_code": code,
                "last_trade_date": last,
                "kind": "performance" if performance else "merger",
                "exchange": venue,
                "delta": delta if performance else 0.0,
                "exit_factor": 1.0 - delta if performance else 1.0,
            }
        )
    return pd.DataFrame(rows, columns=list(DELISTING_COLUMNS))
