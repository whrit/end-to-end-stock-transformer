"""US equities raw data: Alpaca bars, OpenFIGI identity, canonical table assembly.

alpaca-py is imported lazily inside the Alpaca fetch functions, so table
assembly works without the ``us`` extra.
"""

from market_guided_stock_alpha.us_data.alpaca import (
    BAR_SYMBOL_CHUNK,
    US_EXCHANGES,
    alpaca_version,
    bars_frame_to_table,
    fetch_daily_bars,
    fetch_trading_days,
    last_complete_date,
    list_assets,
)
from market_guided_stock_alpha.us_data.openfigi import (
    OpenFigiError,
    map_tickers,
    openfigi_ticker,
)
from market_guided_stock_alpha.us_data.universe import (
    BENCHMARK_CODE,
    MARKET63_ETFS,
    MARKET84_ETFS,
    TABLE_FILES,
    build_calendar,
    build_daily_index,
    build_daily_stock,
    build_membership,
    build_securities,
    build_us_tables,
    etf_code,
    merge_adjusted_and_raw,
)
