"""Ticker to share-class FIGI mapping through the OpenFIGI v3 API (stdlib only)."""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd


OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_EXCHANGE = "US"
MAX_JOBS_PER_REQUEST = 100
MAX_REQUESTS_PER_MINUTE = 250
MAX_RETRIES = 6
REQUEST_TIMEOUT_SECONDS = 60.0
OPENFIGI_COLUMNS = (
    "ticker",
    "share_class_figi",
    "security_type2",
    "market_sector",
    "name",
    "mapping_note",
)


class OpenFigiError(RuntimeError):
    """Raised when OpenFIGI cannot answer a mapping request unambiguously."""


def openfigi_ticker(symbol: str) -> str:
    """Alpaca writes share classes as ``BRK.B``; OpenFIGI expects ``BRK/B``."""
    return str(symbol).strip().upper().replace(".", "/")


def _post(
    jobs: List[Dict[str, str]],
    api_key: str,
    opener: Callable[..., Any],
    sleep: Callable[[float], Any],
) -> List[Dict[str, Any]]:
    body = json.dumps(jobs).encode("utf-8")
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(
            OPENFIGI_URL,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-OPENFIGI-APIKEY": api_key,
            },
        )
        try:
            response = opener(request, timeout=REQUEST_TIMEOUT_SECONDS)
            try:
                payload = json.loads(response.read().decode("utf-8"))
            finally:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise OpenFigiError(
                    "OpenFIGI mapping request failed with HTTP {}".format(error.code)
                ) from error
            if attempt == MAX_RETRIES:
                raise OpenFigiError(
                    "OpenFIGI rate limit (HTTP 429) persisted after {} retries".format(
                        MAX_RETRIES
                    )
                ) from error
            sleep(2.0 ** attempt)
            continue
        if not isinstance(payload, list) or len(payload) != len(jobs):
            raise OpenFigiError(
                "OpenFIGI returned {} results for {} jobs".format(
                    len(payload) if isinstance(payload, list) else "non-list",
                    len(jobs),
                )
            )
        return payload
    raise AssertionError("unreachable")


def _row(ticker: str, result: Any) -> Dict[str, Optional[str]]:
    row = {column: None for column in OPENFIGI_COLUMNS}
    row["ticker"] = ticker
    if not isinstance(result, dict):
        raise OpenFigiError("OpenFIGI result for {} is not an object".format(ticker))
    if "error" in result:
        row["mapping_note"] = "error: {}".format(result["error"])
        return row
    data = result.get("data")
    if not data:
        row["mapping_note"] = "warning: {}".format(
            result.get("warning", "no data returned")
        )
        return row
    candidates = [item for item in data if item.get("exchCode") == OPENFIGI_EXCHANGE]
    figis = sorted(
        set(item.get("shareClassFIGI") for item in candidates)
        - {None}
    )
    if not figis:
        row["mapping_note"] = "warning: no US composite share-class FIGI"
        return row
    if len(figis) > 1:
        raise OpenFigiError(
            "ticker {} maps to several US share-class FIGIs: {}".format(
                ticker, ",".join(figis)
            )
        )
    chosen = next(item for item in candidates if item.get("shareClassFIGI") == figis[0])
    row["share_class_figi"] = figis[0]
    row["security_type2"] = chosen.get("securityType2")
    row["market_sector"] = chosen.get("marketSector")
    row["name"] = chosen.get("name")
    row["mapping_note"] = "mapped"
    return row


def map_tickers(
    tickers: Sequence[str],
    api_key: str,
    opener: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], Any] = time.sleep,
) -> pd.DataFrame:
    """Map tickers to share-class FIGIs, one output row per input ticker.

    Batches 100 jobs per request, spaces requests to stay within 250 per minute,
    and backs off exponentially on HTTP 429. Unmapped tickers keep null fields
    and explain why in ``mapping_note``.
    """
    if not api_key:
        raise OpenFigiError("OpenFIGI API key is empty")
    if opener is None:
        opener = urllib.request.urlopen
    tickers = [str(value) for value in tickers]
    rows = []
    interval = 60.0 / MAX_REQUESTS_PER_MINUTE
    for start in range(0, len(tickers), MAX_JOBS_PER_REQUEST):
        if start:
            sleep(interval)
        batch = tickers[start:start + MAX_JOBS_PER_REQUEST]
        jobs = [
            {
                "idType": "TICKER",
                "idValue": openfigi_ticker(ticker),
                "exchCode": OPENFIGI_EXCHANGE,
            }
            for ticker in batch
        ]
        results = _post(jobs, api_key, opener, sleep)
        rows.extend(_row(ticker, result) for ticker, result in zip(batch, results))
    return pd.DataFrame(rows, columns=list(OPENFIGI_COLUMNS))
