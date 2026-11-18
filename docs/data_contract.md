# Public Data Contract

The public research logic consumes canonical pandas tables. It does not require a company database or a particular vendor. Convert vendor-specific files to the schemas below before invoking the builders; validators reject missing columns, duplicate keys, invalid dates/codes, and non-numeric values where applicable.

Security identifiers are explicit and never guessed. Accepted forms:

- China A-shares and indices: `NNNNNN.SH`, `NNNNNN.SZ`, or `NNNNNN.BJ`.
- US stocks: OpenFIGI share-class FIGI, `BBG` + 9 uppercase alphanumerics (for example `BBG001S5S399`). It stays the same across ticker changes.
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

Known limits:
- **Survivorship bias.** Delisted stocks are included only if Alpaca still lists them as inactive assets and OpenFIGI maps their symbol. `fetch_manifest.json` records how many were mapped and how many were dropped.
- **No delisting returns.** A stock with no price at t+6 has no label for that anchor.
- **Daily open not yet verified.** Whether Alpaca's daily `open` equals the official opening-auction price has not been checked. Verify it against the first regular-session minute bar before trusting label-level results.

R2 credentials come from `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`. Any script argument or config path may be a local path or an `r2://bucket/key` URI. Jobs download remote inputs into a temporary directory and delete it when they finish. Memory-mapped feature artifacts therefore need free local disk equal to their size while a job runs.

Build the US Alpha158 artifact with `--dtype float32 --membership <raw>/membership.parquet`. Float32 halves the size, and training consumes float32 anyway. The membership filter drops stocks that never pass the price/liquidity filter while keeping full history for those that do. China artifacts keep the float64 default.
