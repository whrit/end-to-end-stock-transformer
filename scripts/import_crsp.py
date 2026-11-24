#!/usr/bin/env -S uv run
"""Build the US v3 raw tables (point-in-time top-N by market cap, CRSP delisting
returns) from a WRDS CRSP CIZ daily stock file export; see us_data/crsp.py.

--input takes the WRDS download (.zip or .csv, converted once to a Parquet with
only the needed columns, kept next to it) or that Parquet. Output is a new local
directory or an r2:// location, with the same tables and fetch_manifest.json as
fetch_us_daily.py plus delistings.parquet."""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional, Sequence

from market_guided_stock_alpha.progress import kv_table, note, run_main
from market_guided_stock_alpha.storage import is_remote, r2_client_from_env
from market_guided_stock_alpha.us_data.crsp import (
    DEFAULT_MIN_PRICE,
    DEFAULT_TOP_N,
    ELIGIBLE,
    ETF_PERMNOS,
    EXCHANGES,
    build_crsp_tables,
    convert_export,
    read_crsp_rows,
)
from market_guided_stock_alpha.us_data.sp500 import SHUMWAY_DELTA_NASDAQ, SHUMWAY_DELTA_NYSE_AMEX
from market_guided_stock_alpha.us_data.universe import BENCHMARK_CODE, MARKET63_ETFS, MARKET84_ETFS, write_raw_tables


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build US v3 raw tables from a WRDS CRSP CIZ daily stock export.")
    parser.add_argument("--input", required=True, type=Path, help="WRDS dsf_v2 export (.zip/.csv) or its Parquet.")
    parser.add_argument("--output", required=True, help="New local directory or r2:// location.")
    # IJH and IJR (us_market63) begin trading on 2000-05-26.
    parser.add_argument("--start", default="2000-06-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="Members per day by market cap.")
    parser.add_argument("--min-price", type=float, default=DEFAULT_MIN_PRICE, help="Minimum raw close of a member.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    remote = is_remote(args.output)
    if not remote and Path(args.output).exists():
        raise SystemExit("error: output already exists: {}".format(args.output))
    client = r2_client_from_env() if remote else None
    source = args.input
    if source.suffix in (".zip", ".csv"):
        parquet = source.with_name(source.name.split(".")[0] + ".parquet")
        if not parquet.exists():
            note("converting {} to {}".format(source, parquet))
            partial = parquet.with_name(parquet.name + ".partial")
            rows = convert_export(source, partial)
            partial.replace(parquet)
            note("converted {:,} rows".format(rows))
        source = parquet
    note("reading the top {} stocks by market cap from {}".format(args.top, source))
    rows = read_crsp_rows(source, args.start, args.end, args.top, args.min_price)
    note("building tables from {:,} rows".format(len(rows)))
    tables, counts = build_crsp_tables(rows, args.start, args.end, args.top, args.min_price)
    del rows
    manifest = {
        "source": "crsp_ciz_dsf_v2",
        "input": source.name,
        "start": args.start,
        "end": args.end,
        "universe": "crsp_top_market_cap",
        "filters": {
            "eligible": dict(ELIGIBLE),
            "primary_exchanges": list(EXCHANGES),
            "price_flag": "TR",
            "top_n_by_market_cap": args.top,
            "min_price": args.min_price,
            "delisting_policy": {
                "exit": "last adjusted close x total-return growth to the CRSP delisting value",
                "missing_return_performance": "x (1 - delta)",
                "delta_nyse_amex": SHUMWAY_DELTA_NYSE_AMEX,
                "delta_nasdaq": SHUMWAY_DELTA_NASDAQ,
            },
            "vwap": "typical price (high + low + close) / 3",
        },
        "etfs": {"us_market63": list(MARKET63_ETFS), "us_market84": list(MARKET84_ETFS), "permnos": ETF_PERMNOS},
        "benchmark": BENCHMARK_CODE,
        "counts": counts,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_raw_tables(tables, manifest, args.output, client=client)
    note(kv_table(counts, title="CRSP v3 universe"))
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main(main))
