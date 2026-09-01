# Public Data Contract

The public research logic consumes canonical pandas tables. It does not require a company database or a particular vendor. Convert vendor-specific files to the schemas below before invoking the builders; validators reject missing columns, duplicate keys, invalid dates/codes, and non-numeric values where applicable.

Security identifiers use the explicit form `NNNNNN.SH`, `NNNNNN.SZ`, or `NNNNNN.BJ`. Exchange guessing is deliberately disabled.

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

`read_table()` is intentionally thin: suffix dispatch calls pandas CSV, Parquet, or pickle readers, and all formats immediately enter the same canonical DataFrame validators. Parquet requires the optional `pyarrow` dependency (`pip install -e ".[parquet]"`). Pandas pickle can execute arbitrary code while loading and must only be used for files from trusted sources.

CSV/Parquet/pickle support is IO convenience, not a loader framework: there is no registry, schema inference, automatic vendor detection, or format-specific research logic.
