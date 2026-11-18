# Public Data Contract

The public research logic consumes canonical pandas tables. It does not require a company database or a particular vendor. Convert vendor-specific files to the schemas below before invoking the builders; validators reject missing columns, duplicate keys, invalid dates/codes, and non-numeric values where applicable.

Security identifiers are explicit and never guessed. Accepted forms:

- China A-shares and indices: `NNNNNN.SH`, `NNNNNN.SZ`, or `NNNNNN.BJ`.
- US stocks: OpenFIGI share-class FIGI, `BBG` + 9 uppercase alphanumerics (for example `BBG001S5S399`). It stays the same across ticker changes.
- US historical stocks with no verifiable FIGI (US v2 only): `<TICKER>.US.<YYYYMMDD>`, the company's last S&P 500 ticker and the date of its last bar (for example `SIVB.US.20230309`). Used for bankruptcies OpenFIGI no longer maps, tickers since reused by another company, and companies renamed after leaving the index.
- US ETFs used as market state or benchmark: `<TICKER>.US` (for example `SPY.US`).

## Daily stock

| Column | Meaning | Requirement |
|---|---|---|
| `trade_date` | Exchange trading date | parseable date |
| `sec_code` | Exchange-qualified stock code | unique with `trade_date` |
| `open`, `high`, `low`, `close` | Daily prices | numeric |
| `vwap` | Daily volume-weighted average price | numeric |
| `volume` | Daily traded volume | numeric |
| `adj_open` | Split/dividend-adjusted open | additionally required by the formal label builder |

Alpha158 uses the base OHLCV/VWAP contract. The relative-excess label requires `adj_open`; adjustment methodology belongs to the user's market-data preparation and must be applied consistently.

## Daily index

| Column | Meaning | Requirement |
|---|---|---|
| `trade_date` | Trading date | parseable date |
| `index_code` | Exchange-qualified index code | unique with `trade_date` |
| `open`, `close` | Daily index prices | numeric |
| `amount` | Daily turnover amount | numeric |

The label selects CSI500 (`000905.SH`) and uses `open`. Market63 requires CSI300, CSI500, and CSI800 (`000300.SH`, `000905.SH`, `000906.SH`). Market84 requires CSI300, CSI500, CSI1000, and CSI All (`000300.SH`, `000905.SH`, `000852.SH`, `000985.SH`).

## Minute stock and index

Canonical long-form minute tables share six numeric fields:

| Table | Key columns | Value columns |
|---|---|---|
| Minute stock | `timestamp`, `sec_code` | `open`, `high`, `low`, `close`, `volume`, `amt` |
| Minute index | `timestamp`, `index_code` | `open`, `high`, `low`, `close`, `volume`, `amt` |

Keys must be unique and timestamps must be parseable. Intraday Price-Volume Features (248D) and Intraday Market Features (192D) use the canonical Chinese cash-equity session contract of 240 bars per day (120 morning + 120 afternoon). Missing observations are retained as missing values and handled by explicit coverage/eligibility logic; they are not silently fabricated.

For trusted legacy research files, the intraday CLI also accepts six aligned wide timestamp-by-instrument pandas pickles through `--wide-open`, `--wide-high`, `--wide-low`, `--wide-close`, `--wide-volume`, and `--wide-amt`. All six axes must match exactly before conversion to the same canonical long DataFrame.

## Membership

Required columns are `trade_date` and `sec_code`, unique as a pair. Optional `index_code` and numeric `weight` columns are preserved. The canonical research universe is dynamic All A-shares; membership controls which securities are eligible on each date.

## Trading calendar

| Column | Meaning |
|---|---|
| `calendar_date` | Calendar date, unique and sorted after validation |
| `is_open` | Integer `0` or `1` |

The open-date sequence is the single source of truth for `t+1` and `t+6` label offsets, rolling targets, lookback dates, and purge boundaries.

## File formats and trust

`read_table()` is intentionally thin: suffix dispatch calls pandas CSV, Parquet, or pickle readers, and all formats immediately enter the same canonical DataFrame validators. Parquet requires the optional `pyarrow` dependency (`uv sync --extra parquet`, or `--extra us`). Pandas pickle can execute arbitrary code while loading and must only be used for files from trusted sources.

CSV/Parquet/pickle support is IO convenience, not a loader framework: there is no registry, schema inference, automatic vendor detection, or format-specific research logic.

## US equities profile

US data is produced by `scripts/fetch_us_daily.py` from Alpaca (consolidated SIP feed, history since 2016) and OpenFIGI, then stored in Cloudflare R2. It feeds the same validators and builders as the China data.

| Item | US choice |
|---|---|
| Universe (`universe`) | `dynamic_us_common_stocks`: OpenFIGI `securityType2 == "Common Stock"`. ADRs, ETFs and preferreds are excluded |
| Daily membership | raw close ≥ $5 and 20-trading-day mean raw dollar volume (raw close × raw volume, including day t) ≥ $5M |
| Daily stock prices | `open/high/low/close/vwap/volume` and `adj_open` use Alpaca `adjustment=all` (split + dividend + spin-off). `raw_close` and `raw_volume` are kept for the membership filter |
| Label benchmark | `SPY.US` (dividend-adjusted open); label contract `spy_relative_excess_return_t1_t6` |
| Market state | `us_market63` = SPY/IJH/IJR; `us_market84` adds IWM. Same 21-feature-per-index formulas as Market63/84; `amount` = raw volume × raw VWAP |
| Calendar | Alpaca trading calendar; `is_open` for every calendar day |
| Storage | `r2://<bucket>/us/v1/raw/*.parquet` (canonical tables + `securities.parquet` + `fetch_manifest.json`), `.../derived/` (artifacts and label), `.../runs/` |

**US v2: point-in-time S&P 500** (`fetch_us_daily.py --universe sp500`, `us/v2/`), which replaces both limits below for the S&P 500:
- **Members.** On each trading day the members are the latest list in the [fja05680/sp500](https://github.com/fja05680/sp500) history (MIT) on or before that day, kept only on days the stock has a bar. Days after the history's last date reuse its last list, and the manifest counts them.
- **Identity.** Each membership span is priced with Alpaca bars requested `asof` its last member day; that follows the company through later renames (FB → META) and never returns a later holder of the ticker. Spans with identical raw price histories are one company.
- **FIGIs.** Companies still in the index on the last day map through their current ticker. A removed, delisted company gets its delisted FIGI only if no bar has been printed under its ticker since its last bar. A removed company that still trades keeps a FIGI only if its ticker still prices the same company. Everything else gets `<TICKER>.US.<YYYYMMDD>`. `securities.parquet` records the rule used (`identity`) and every S&P ticker of the company (`sp500_tickers`). REITs are included.
- **Delistings.** `delistings.parquet` (`sec_code, last_trade_date, kind, exchange, delta, exit_factor`) lists every company whose bars stop before the last day. A delisting is a performance delisting if the last raw close is below $1 or the last close is below half the highest close of its final 20 sessions; otherwise it is a merger. `build_relative_excess_label.py --delistings` then labels anchors whose stock stops trading between t+1 and t+6 with the last adjusted close × `exit_factor` in place of the missing t+6 open. `exit_factor` is 1 for mergers and 1 − δ for performance delistings, with δ = 30% on NYSE/AMEX and 55% on Nasdaq or an unknown exchange (Shumway 1997; Shumway & Warther 1999).

Known limits of US v1 (`--universe listed`):
- **Survivorship bias.** Delisted stocks are included only if Alpaca still lists them as inactive assets and OpenFIGI maps their symbol. `fetch_manifest.json` records how many were mapped and how many were dropped. Alpaca does serve full daily bars for delisted tickers missing from its asset list, so the gap is the ticker list, not prices. See [the US Phase 2 plan](us_phase2_plan.md).
- **No delisting returns.** A stock with no price at t+6 has no label for that anchor.
- **No REITs.** OpenFIGI types REITs as `REIT`, not `Common Stock`.

Known limits of both:
- **Daily open is a close proxy for the official open (measured).**
  - Against the primary exchange's opening-auction print (condition `O` on the listing venue; 463 stock-days, 12 symbols, June 2021 and October 2026), Alpaca's daily `open` matches exactly 43% of the time.
  - Error: median absolute 0.8 bps, 95th percentile 33 bps, mean signed +1.8 bps. The offset appears at both label endpoints (t+1 and t+6), so it largely cancels in the ratio.
  - The first 09:30 minute bar is no better (median 1.6 bps, 95th percentile 35 bps), so the fetch keeps the daily `open`.
  - The exact auction price is an optional upgrade in the Phase 2 plan.

R2 credentials come from `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`. Any script argument or config path may be a local path or an `r2://bucket/key` URI. Jobs download remote inputs into a temporary directory and delete it when they finish. Memory-mapped feature artifacts therefore need free local disk equal to their size while a job runs.

Build the US Alpha158 artifact with `--dtype float32 --membership <raw>/membership.parquet`. Float32 halves the size, and training consumes float32 anyway. The membership filter drops stocks that never pass the price/liquidity filter while keeping full history for those that do. China artifacts keep the float64 default.
