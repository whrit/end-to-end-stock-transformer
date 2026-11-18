#!/usr/bin/env -S uv run
"""Fetch US daily equities from Alpaca, map identities through OpenFIGI, and
write the canonical raw tables plus fetch_manifest.json to a local dir or R2."""

import argparse
import datetime as dt
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

from market_guided_stock_alpha.artifacts import atomic_write_json, sha256_path
from market_guided_stock_alpha.progress import note
from market_guided_stock_alpha.storage import is_remote, publish, r2_client_from_env
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
    TABLE_FILES,
    build_securities,
    build_us_tables,
)


DEFAULT_OUTPUT = "r2://stock-transformer-data/us/v1/raw"
REQUIRED_ENV = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "OPENFIGI_API_KEY")
MANIFEST_NAME = "fetch_manifest.json"


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
    return parser


def _write_tables(
    tables: Mapping[str, pd.DataFrame], manifest: Dict[str, Any], directory: Path
) -> None:
    directory.mkdir(parents=True)
    files = {}
    for name, filename in TABLE_FILES.items():
        path = directory / filename
        tables[name].to_parquet(path, index=False)
        files[name] = {
            "file": filename,
            "rows": int(len(tables[name])),
            "sha256": sha256_path(path),
        }
    manifest["tables"] = files
    atomic_write_json(directory / MANIFEST_NAME, manifest)


def _stage(index: int, text: str) -> None:
    note(("━━ {}/6 ".format(index), "bold magenta"), (text, "bold"))


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
    if wanted is not None:
        missing = sorted(wanted.difference(assets["alpaca_symbol"]))
        if missing:
            raise SystemExit("error: unknown Alpaca symbols: {}".format(",".join(missing)))
        assets = assets.loc[assets["alpaca_symbol"].isin(wanted)].reset_index(drop=True)
    _stage(2, "mapping {:,} tickers through OpenFIGI".format(assets["alpaca_symbol"].nunique()))
    figi = map_tickers(sorted(set(assets["alpaca_symbol"])), secrets["OPENFIGI_API_KEY"])
    securities, counts = build_securities(assets, figi)
    symbols = sorted(set(securities["alpaca_symbol"]) | set(MARKET84_ETFS))
    _stage(3, "fetching daily bars for {:,} symbols (adjusted, then raw)".format(len(symbols)))
    adjusted = fetch_daily_bars(data, symbols, start, end, "all")
    raw = fetch_daily_bars(data, symbols, start, end, "raw")
    _stage(4, "fetching trading calendar")
    trading_days = fetch_trading_days(trading, start, end)
    _stage(5, "building canonical tables")
    tables = build_us_tables(
        securities, adjusted, raw, trading_days, start, end,
        min_price=args.min_price, min_dollar_volume=args.min_dollar_volume,
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
        "filters": {
            "exchanges": list(US_EXCHANGES),
            "security_type2": COMMON_STOCK_TYPE,
            "market_sector": EQUITY_SECTOR,
            "min_price": args.min_price,
            "min_dollar_volume": args.min_dollar_volume,
            "dollar_volume_window_trading_days": DOLLAR_VOLUME_WINDOW,
        },
        "etfs": {"us_market63": list(MARKET63_ETFS), "us_market84": list(MARKET84_ETFS)},
        "benchmark": BENCHMARK_CODE,
        "counts": counts,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }

    _stage(6, "writing tables to {}".format(args.output))
    if remote:
        with tempfile.TemporaryDirectory() as workdir:
            staged = Path(workdir) / "raw"
            _write_tables(tables, manifest, staged)
            publish(staged, args.output, client=r2_client)
    else:
        local_output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(local_output.parent)) as workdir:
            staged = Path(workdir) / "raw"
            _write_tables(tables, manifest, staged)
            os.replace(str(staged), str(local_output))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
