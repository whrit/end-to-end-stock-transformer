#!/usr/bin/env -S uv run
"""Fetch US daily equities from Alpaca, map identities through OpenFIGI, and
write the canonical raw tables plus fetch_manifest.json to a local dir or R2.

``--universe listed`` (v1): today's listed common stocks, filtered by price and
dollar volume. ``--universe sp500`` (v2): point-in-time S&P 500 members since
``--start``, including removed and delisted companies, plus delistings.parquet
for the label's delisting exits (see us_data/sp500.py)."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
from rich.console import Group
from rich.table import Table
from rich.text import Text

from market_guided_stock_alpha.progress import board, kv_table, note, run_main
from market_guided_stock_alpha.storage import is_remote, r2_client_from_env
from market_guided_stock_alpha.us_data.alpaca import (
    NEW_YORK,
    US_EXCHANGES,
    alpaca_version,
    fetch_daily_bars,
    fetch_trading_days,
    last_complete_date,
    list_assets,
)
from market_guided_stock_alpha.us_data.openfigi import map_tickers
from market_guided_stock_alpha.us_data.universe import (
    BENCHMARK_CODE,
    COMMON_STOCK_TYPE,
    DOLLAR_VOLUME_WINDOW,
    EQUITY_SECTOR,
    MARKET63_ETFS,
    MARKET84_ETFS,
    build_securities,
    build_us_tables,
    write_raw_tables,
)
from market_guided_stock_alpha.us_data.sp500 import (
    PERFORMANCE_DRAWDOWN,
    PERFORMANCE_DRAWDOWN_SESSIONS,
    PERFORMANCE_MAX_RAW_CLOSE,
    SHUMWAY_DELTA_NASDAQ,
    SHUMWAY_DELTA_NYSE_AMEX,
    SP500_HISTORY_URL,
    build_sp500_universe,
    load_history,
)


DEFAULT_OUTPUT = "r2://stock-transformer-data/us/v1/raw"
REQUIRED_ENV = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "OPENFIGI_API_KEY")


def required_env(environ: Mapping[str, str] = os.environ) -> Dict[str, str]:
    missing = [name for name in REQUIRED_ENV if not environ.get(name)]
    if missing:
        raise SystemExit(
            "error: missing required environment variables: {}".format(", ".join(missing))
        )
    return {name: environ[name] for name in REQUIRED_ENV}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Alpaca SIP daily bars for US common stocks and ETFs."
    )
    parser.add_argument("--start", default="2016-01-01", help="First New York date.")
    parser.add_argument(
        "--end", default=None,
        help="Last New York date (default today; clamped to the last final session).",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="New local directory or r2://bucket/prefix for the raw tables.",
    )
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=5000000.0)
    parser.add_argument(
        "--symbols", default=None,
        help="Comma-separated Alpaca symbols to keep (e.g. AAPL,MSFT) for smoke runs; "
        "default: the whole listed universe.",
    )
    parser.add_argument(
        "--universe", choices=("listed", "sp500"), default="listed",
        help="listed: today's listed common stocks (v1); sp500: point-in-time "
        "S&P 500 members including removed and delisted companies (v2).",
    )
    parser.add_argument(
        "--sp500-history", default=SP500_HISTORY_URL,
        help="fja05680/sp500 'Historical Components & Changes (Updated)' CSV (URL or path).",
    )
    return parser


STAGES = (
    "list Alpaca assets",
    "map tickers through OpenFIGI",
    "fetch daily bars (adjusted, raw)",
    "fetch trading calendar",
    "build canonical tables",
    "write tables",
)
QUALITY: Dict[str, Any] = {}


def _stage(index: int, text: str) -> None:
    table = Table.grid(padding=(0, 2))
    for position, name in enumerate(STAGES, start=1):
        if position < index:
            table.add_row(Text("✓", style="bold green"), Text(name, style="green"))
        elif position == index:
            table.add_row(Text("▶", style="bold dodger_blue1"), Text(name, style="bold"))
        else:
            table.add_row(Text("·", style="dim"), Text(name, style="dim"))
    board(Group(table, kv_table(QUALITY, title="data quality")) if QUALITY else table, "FETCH")
    note(("━━ {}/{} ".format(index, len(STAGES)), "bold dodger_blue1"), (text, "bold"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    wanted = (
        None if args.symbols is None
        else {value.strip().upper() for value in args.symbols.split(",") if value.strip()}
    )
    if wanted is not None and not wanted:
        raise SystemExit("error: --symbols is empty")
    start = pd.Timestamp(args.start).normalize()
    requested_end = (
        pd.Timestamp(pd.Timestamp.now(tz=NEW_YORK).date())
        if args.end is None
        else pd.Timestamp(args.end).normalize()
    )
    end = min(requested_end, last_complete_date())
    if start > end:
        raise SystemExit(
            "error: start {} is after the last complete session {}".format(
                start.date(), end.date()
            )
        )
    secrets = required_env()
    remote = is_remote(args.output)
    r2_client = r2_client_from_env() if remote else None
    local_output = None if remote else Path(args.output)
    if local_output is not None and local_output.exists():
        raise SystemExit("error: output already exists: {}".format(local_output))

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    # Assets/calendar live on the paper or live trading endpoint, matching the
    # key type; APCA_PAPER=false selects live keys (default: paper).
    trading = TradingClient(
        secrets["APCA_API_KEY_ID"],
        secrets["APCA_API_SECRET_KEY"],
        paper=os.environ.get("APCA_PAPER", "true").strip().lower() != "false",
    )
    data = StockHistoricalDataClient(
        secrets["APCA_API_KEY_ID"], secrets["APCA_API_SECRET_KEY"]
    )

    _stage(1, "listing Alpaca assets")
    assets = list_assets(trading)
    sp500_source: Optional[Dict[str, Any]] = None
    if args.universe == "sp500":
        _stage(2, "building the point-in-time S&P 500 universe")
        history, history_sha256 = load_history(args.sp500_history)
        if wanted is not None:
            history = history.assign(tickers=[tuple(t for t in ts if t in wanted) for ts in history["tickers"]])
        trading_days = fetch_trading_days(trading, start, end)
        universe = build_sp500_universe(
            history,
            [day for day in trading_days if start <= day <= end],
            lambda symbols, asof, adjustment: fetch_daily_bars(
                data, symbols, start, end, adjustment, asof="-" if asof is None else asof, quiet=True
            ),
            lambda tickers, unlisted: map_tickers(
                tickers, secrets["OPENFIGI_API_KEY"], include_unlisted=unlisted
            ),
            assets,
        )
        counts = dict(universe["counts"])
        identity = counts.get("identity", {})
        QUALITY.update(
            {
                "S&P episodes": counts["episodes"],
                "companies": counts["companies"],
                "renamed (several tickers)": counts["companies_with_several_tickers"],
                "episodes without bars": counts["episodes_without_bars"],
                "FIGI (current/verified/delisted)": "{}/{}/{}".format(
                    identity.get("figi_current", 0), identity.get("figi_verified", 0), identity.get("figi_delisted", 0)
                ),
                "TICKER.US.YYYYMMDD codes": sum(v for k, v in identity.items() if k.startswith("fallback")),
                "delisted: merger / performance": "{} / {}".format(
                    counts["delisted_merger"], counts["delisted_performance"]
                ),
                "days after S&P history": counts["trading_days_after_history"],
            }
        )
        sp500_source = {"location": str(args.sp500_history), "sha256": history_sha256}
        _stage(3, "fetching ETF bars (adjusted, raw)")
        adjusted = pd.concat(
            [universe["adjusted"], fetch_daily_bars(data, MARKET84_ETFS, start, end, "all")], ignore_index=True
        )
        raw = pd.concat(
            [universe["raw"], fetch_daily_bars(data, MARKET84_ETFS, start, end, "raw")], ignore_index=True
        )
        securities = universe["securities"]
        # Stock bars are already keyed by sec_code (in alpaca_symbol).
        key = pd.DataFrame({"sec_code": securities["sec_code"], "alpaca_symbol": securities["sec_code"]})
        for column in ("ticker", "name", "status", "exchange", "security_type2"):
            key[column] = None
    else:
        if wanted is not None:
            missing = sorted(wanted.difference(assets["alpaca_symbol"]))
            if missing:
                raise SystemExit("error: unknown Alpaca symbols: {}".format(",".join(missing)))
            assets = assets.loc[assets["alpaca_symbol"].isin(wanted)].reset_index(drop=True)
        _stage(2, "mapping {:,} tickers through OpenFIGI".format(assets["alpaca_symbol"].nunique()))
        figi = map_tickers(sorted(set(assets["alpaca_symbol"])), secrets["OPENFIGI_API_KEY"])
        securities, counts = build_securities(assets, figi)
        key = securities
        QUALITY.update(
            {
                "tickers seen": counts.get("symbols", 0),
                "mapped to FIGI": counts.get("mapped", 0),
                "unmapped": counts.get("unmapped", 0),
                "ambiguous FIGI (excluded)": counts.get("ambiguous_figi", 0),
                "excluded: not common stock": counts.get("excluded_by_type", 0),
                "common stocks": counts.get("common_stocks", 0),
            }
        )
        symbols = sorted(set(securities["alpaca_symbol"]) | set(MARKET84_ETFS))
        _stage(3, "fetching daily bars for {:,} symbols (adjusted, then raw)".format(len(symbols)))
        adjusted = fetch_daily_bars(data, symbols, start, end, "all")
        raw = fetch_daily_bars(data, symbols, start, end, "raw")
    QUALITY.update(
        {
            "symbols with bars": int(adjusted["alpaca_symbol"].nunique()),
            "bar rows": len(adjusted),
        }
    )
    _stage(4, "fetching trading calendar")
    trading_days = fetch_trading_days(trading, start, end)
    _stage(5, "building canonical tables")
    tables = build_us_tables(
        key, adjusted, raw, trading_days, start, end,
        min_price=args.min_price, min_dollar_volume=args.min_dollar_volume,
        membership=universe["membership"] if args.universe == "sp500" else None,
    )
    if args.universe == "sp500":
        tables["securities"] = securities
        tables["delistings"] = universe["delistings"]
    stock = tables["daily_stock"]
    numeric = stock.select_dtypes("number")
    QUALITY.update(
        {
            "daily_stock rows": len(stock),
            "missing numeric values": float(numeric.isna().to_numpy().mean()) if numeric.size else 0.0,
            "membership rows": len(tables["membership"]),
            "membership securities": int(tables["membership"]["sec_code"].nunique()),
        }
    )
    counts.update(
        {
            "symbols_with_bars": int(adjusted["alpaca_symbol"].nunique()),
            "open_trading_days": int(tables["calendar"]["is_open"].sum()),
            "daily_stock_securities": int(tables["daily_stock"]["sec_code"].nunique()),
            "membership_securities": int(tables["membership"]["sec_code"].nunique()),
        }
    )
    manifest = {
        "source": "alpaca",
        "feed": "sip",
        "alpaca_py_version": alpaca_version(),
        "start": str(start.date()),
        "end": str(end.date()),
        "requested_end": str(requested_end.date()),
        "symbols": None if wanted is None else sorted(wanted),
        "universe": args.universe,
        "filters": (
            {
                "exchanges": list(US_EXCHANGES),
                "security_type2": COMMON_STOCK_TYPE,
                "market_sector": EQUITY_SECTOR,
                "min_price": args.min_price,
                "min_dollar_volume": args.min_dollar_volume,
                "dollar_volume_window_trading_days": DOLLAR_VOLUME_WINDOW,
            }
            if args.universe == "listed"
            else {
                "membership": "point-in-time S&P 500 member and traded that day",
                "sp500_history": sp500_source,
                "delisting_policy": {
                    "merger_exit": "last adjusted close",
                    "performance_exit": "last adjusted close x (1 - delta)",
                    "delta_nyse_amex": SHUMWAY_DELTA_NYSE_AMEX,
                    "delta_nasdaq_or_unknown": SHUMWAY_DELTA_NASDAQ,
                    "performance_if_last_raw_close_below": PERFORMANCE_MAX_RAW_CLOSE,
                    "performance_if_last_close_below_fraction_of_max": PERFORMANCE_DRAWDOWN,
                    "drawdown_sessions": PERFORMANCE_DRAWDOWN_SESSIONS,
                },
            }
        ),
        "etfs": {"us_market63": list(MARKET63_ETFS), "us_market84": list(MARKET84_ETFS)},
        "benchmark": BENCHMARK_CODE,
        "counts": counts,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    _stage(6, "writing tables to {}".format(args.output))
    write_raw_tables(tables, manifest, args.output, client=r2_client)
    note(kv_table(QUALITY, title="data quality"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(main))
