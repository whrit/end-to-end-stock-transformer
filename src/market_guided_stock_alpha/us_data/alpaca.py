"""Thin Alpaca wrappers over injected alpaca-py clients.

Clients are built by the caller; this module never sees or stores credentials.
"""

from typing import Any, List, Optional, Sequence

import pandas as pd

from market_guided_stock_alpha.data.contracts import DataQualityError
from market_guided_stock_alpha.progress import progress


NEW_YORK = "America/New_York"
US_EXCHANGES = ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
BAR_SYMBOL_CHUNK = 200
# Free-plan SIP queries must end at least 15 minutes in the past.
SIP_DELAY = pd.Timedelta(minutes=16)
# After-hours trading ends at 20:00 New York; a date's daily bar is final after it.
SESSION_FINAL_HOUR = 20
ASSET_COLUMNS = ("asset_id", "alpaca_symbol", "name", "status", "exchange", "tradable")
BAR_COLUMNS = (
    "alpaca_symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
)


def _enum_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def last_complete_date(now: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Latest New York date whose daily bar can no longer change."""
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = current.tz_convert(NEW_YORK)
    day = pd.Timestamp(local.date())
    if local.hour < SESSION_FINAL_HOUR:
        day -= pd.Timedelta(days=1)
    return day


def list_assets(trading_client: Any) -> pd.DataFrame:
    """Active and inactive US equities on the major exchanges, tradable or not."""
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

    rows = []
    for status in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        assets = trading_client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=status)
        )
        for asset in assets:
            exchange = _enum_text(asset.exchange)
            if exchange not in US_EXCHANGES:
                continue
            rows.append(
                {
                    "asset_id": str(asset.id),
                    "alpaca_symbol": str(asset.symbol).upper(),
                    "name": asset.name,
                    "status": _enum_text(asset.status),
                    "exchange": exchange,
                    "tradable": bool(asset.tradable),
                }
            )
    return pd.DataFrame(rows, columns=list(ASSET_COLUMNS))


def bars_frame_to_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn a ``BarSet.df`` (symbol, timestamp UTC) frame into New York session dates."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    flat = frame.reset_index()
    missing = [
        column
        for column in ("symbol", "timestamp", "open", "high", "low", "close", "volume", "vwap")
        if column not in flat.columns
    ]
    if missing:
        raise DataQualityError(
            "Alpaca bars missing columns: {}".format(",".join(missing))
        )
    stamps = pd.to_datetime(flat["timestamp"], utc=True)
    output = pd.DataFrame(
        {
            "alpaca_symbol": flat["symbol"].astype(str).str.upper(),
            "trade_date": stamps.dt.tz_convert(NEW_YORK)
            .dt.tz_localize(None)
            .dt.normalize()
            .astype("datetime64[ns]"),
        }
    )
    for column in BAR_COLUMNS[2:]:
        output[column] = pd.to_numeric(flat[column], errors="raise").astype("float64")
    duplicated = output.duplicated(["alpaca_symbol", "trade_date"], keep=False)
    if duplicated.any():
        first = output.loc[duplicated].iloc[0]
        raise DataQualityError(
            "Alpaca returned several daily bars for {} on {}".format(
                first["alpaca_symbol"], first["trade_date"].date()
            )
        )
    return output.reset_index(drop=True)


def fetch_daily_bars(
    data_client: Any,
    symbols: Sequence[str],
    start: Any,
    end: Any,
    adjustment: str,
    chunk_size: int = BAR_SYMBOL_CHUNK,
    now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Daily SIP bars for ``symbols`` between New York dates ``start``..``end``.

    ``adjustment`` is ``"raw"`` or ``"all"``.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive int")
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    request_start = pd.Timestamp(start).tz_localize(NEW_YORK)
    request_end = min(
        (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(NEW_YORK)
        - pd.Timedelta(seconds=1),
        current - SIP_DELAY,
    )
    unique = sorted(set(str(symbol).upper() for symbol in symbols))
    pieces = []
    bar = progress(
        total=len(unique), desc="Alpaca daily bars ({})".format(adjustment), unit="symbol"
    )
    for offset in range(0, len(unique), chunk_size):
        result = data_client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=unique[offset:offset + chunk_size],
                timeframe=TimeFrame.Day,
                start=request_start.to_pydatetime(),
                end=request_end.to_pydatetime(),
                adjustment=Adjustment(adjustment),
                feed=DataFeed.SIP,
            )
        )
        pieces.append(bars_frame_to_table(result.df))
        bar.update(len(unique[offset:offset + chunk_size]))
    bar.close()
    if not pieces:
        return pd.DataFrame(columns=list(BAR_COLUMNS))
    return pd.concat(pieces, ignore_index=True)


def fetch_trading_days(trading_client: Any, start: Any, end: Any) -> List[pd.Timestamp]:
    """Open New York session dates from Alpaca's market calendar."""
    from alpaca.trading.requests import GetCalendarRequest

    sessions = trading_client.get_calendar(
        GetCalendarRequest(
            start=pd.Timestamp(start).date(), end=pd.Timestamp(end).date()
        )
    )
    return sorted(set(pd.Timestamp(session.date) for session in sessions))


def alpaca_version() -> str:
    from importlib.metadata import version

    return version("alpaca-py")
