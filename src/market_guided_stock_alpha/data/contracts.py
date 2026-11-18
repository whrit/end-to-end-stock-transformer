"""Strict canonical DataFrame contracts used by public research formulas."""

import re
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DAILY_STOCK_COLUMNS = (
    "trade_date",
    "sec_code",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
)
DAILY_INDEX_COLUMNS = (
    "trade_date",
    "index_code",
    "open",
    "close",
    "amount",
)
MINUTE_FIELDS = ("open", "high", "low", "close", "volume", "amt")
MINUTE_STOCK_COLUMNS = ("timestamp", "sec_code") + MINUTE_FIELDS
MINUTE_INDEX_COLUMNS = ("timestamp", "index_code") + MINUTE_FIELDS
MEMBERSHIP_COLUMNS = ("trade_date", "sec_code")
CALENDAR_COLUMNS = ("calendar_date", "is_open")
US_SECURITY_CODE_PATTERN = r"BBG[0-9A-Z]{9}|[A-Z][A-Z0-9]{0,5}(\.[A-Z])?\.US"
US_SECURITY_CODE = re.compile(r"^({})$".format(US_SECURITY_CODE_PATTERN))
_SECURITY_CODE = re.compile(
    r"^(\d{{6}}\.(SH|SZ|BJ)|{})$".format(US_SECURITY_CODE_PATTERN)
)


class SchemaError(ValueError):
    """Raised when a canonical DataFrame is missing required structure."""


class DataQualityError(ValueError):
    """Raised when canonical keys or values violate research requirements."""


def normalize_security_code(value: Any, context: str = "security code") -> str:
    """Validate an exchange-suffixed code without guessing its exchange."""
    if value is None or isinstance(value, bool):
        raise DataQualityError("{} is empty".format(context))
    text = str(value).strip().upper()
    if not text or _SECURITY_CODE.fullmatch(text) is None:
        raise DataQualityError(
            "{}={!r} is not NNNNNN.SH/SZ/BJ, BBG FIGI, or TICKER.US; "
            "exchange guessing is disabled".format(context, value)
        )
    return text


def normalize_date_values(values: Any, context: str) -> pd.Series:
    parsed = pd.Series(pd.to_datetime(values, errors="coerce"))
    if parsed.isna().any():
        raise DataQualityError("{} contains invalid dates".format(context))
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize().astype("datetime64[ns]")


def normalize_timestamp_values(values: Any, context: str) -> pd.Series:
    parsed = pd.Series(pd.to_datetime(values, errors="coerce"))
    if parsed.isna().any():
        raise DataQualityError("{} contains invalid timestamps".format(context))
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.astype("datetime64[ns]")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("{} must be a pandas DataFrame".format(name))
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaError(
            "{} missing required columns: {}".format(name, ",".join(missing))
        )


def _numeric_columns(
    frame: pd.DataFrame, columns: Iterable[str], name: str
) -> None:
    for column in columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce").astype("float64")
        if (original.notna() & numeric.isna()).any():
            raise DataQualityError(
                "{} column {} contains non-numeric values".format(name, column)
            )
        numeric.loc[~np.isfinite(numeric.to_numpy())] = np.nan
        frame[column] = numeric


def _unique_keys(frame: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    if frame.duplicated(list(keys)).any():
        raise DataQualityError(
            "{} contains duplicate {} keys".format(name, "/".join(keys))
        )


def validate_daily_stock(
    frame: pd.DataFrame, require_adj_open: bool = False
) -> pd.DataFrame:
    required = list(DAILY_STOCK_COLUMNS)
    if require_adj_open:
        required.append("adj_open")
    _require_columns(frame, required, "daily stock")
    output = frame.loc[:, required].copy()
    output["trade_date"] = normalize_date_values(
        output["trade_date"], "daily stock trade_date"
    ).to_numpy()
    output["sec_code"] = [
        normalize_security_code(value, "daily stock sec_code")
        for value in output["sec_code"]
    ]
    _numeric_columns(output, required[2:], "daily stock")
    _unique_keys(output, ("trade_date", "sec_code"), "daily stock")
    return output.sort_values(
        ["trade_date", "sec_code"], kind="mergesort"
    ).reset_index(drop=True)


def validate_daily_index(frame: pd.DataFrame) -> pd.DataFrame:
    required = list(DAILY_INDEX_COLUMNS)
    _require_columns(frame, required, "daily index")
    output = frame.loc[:, required].copy()
    output["trade_date"] = normalize_date_values(
        output["trade_date"], "daily index trade_date"
    ).to_numpy()
    output["index_code"] = [
        normalize_security_code(value, "daily index index_code")
        for value in output["index_code"]
    ]
    _numeric_columns(output, required[2:], "daily index")
    _unique_keys(output, ("trade_date", "index_code"), "daily index")
    return output.sort_values(
        ["trade_date", "index_code"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_minute(
    frame: pd.DataFrame, code_column: str, name: str
) -> pd.DataFrame:
    required = ("timestamp", code_column) + MINUTE_FIELDS
    _require_columns(frame, required, name)
    output = frame.loc[:, list(required)].copy()
    output["timestamp"] = normalize_timestamp_values(
        output["timestamp"], "{} timestamp".format(name)
    ).to_numpy()
    output[code_column] = [
        normalize_security_code(value, "{} {}".format(name, code_column))
        for value in output[code_column]
    ]
    _numeric_columns(output, MINUTE_FIELDS, name)
    _unique_keys(output, ("timestamp", code_column), name)
    return output.sort_values(
        ["timestamp", code_column], kind="mergesort"
    ).reset_index(drop=True)


def validate_minute_stock(frame: pd.DataFrame) -> pd.DataFrame:
    return _validate_minute(frame, "sec_code", "minute stock")


def validate_minute_index(frame: pd.DataFrame) -> pd.DataFrame:
    return _validate_minute(frame, "index_code", "minute index")


def validate_membership(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, MEMBERSHIP_COLUMNS, "membership")
    optional = [
        column for column in ("index_code", "weight") if column in frame.columns
    ]
    output = frame.loc[:, list(MEMBERSHIP_COLUMNS) + optional].copy()
    output["trade_date"] = normalize_date_values(
        output["trade_date"], "membership trade_date"
    ).to_numpy()
    output["sec_code"] = [
        normalize_security_code(value, "membership sec_code")
        for value in output["sec_code"]
    ]
    if "index_code" in output:
        output["index_code"] = [
            normalize_security_code(value, "membership index_code")
            for value in output["index_code"]
        ]
    if "weight" in output:
        _numeric_columns(output, ("weight",), "membership")
    _unique_keys(output, MEMBERSHIP_COLUMNS, "membership")
    return output.sort_values(
        list(MEMBERSHIP_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def validate_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, CALENDAR_COLUMNS, "calendar")
    output = frame.loc[:, list(CALENDAR_COLUMNS)].copy()
    output["calendar_date"] = normalize_date_values(
        output["calendar_date"], "calendar_date"
    ).to_numpy()
    numeric = pd.to_numeric(output["is_open"], errors="coerce")
    if numeric.isna().any() or not numeric.isin((0, 1)).all():
        raise DataQualityError("calendar is_open must contain only 0 or 1")
    output["is_open"] = numeric.astype("int8")
    _unique_keys(output, ("calendar_date",), "calendar")
    return output.sort_values("calendar_date", kind="mergesort").reset_index(
        drop=True
    )


def validate_date_range(start: Any, end: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
    values = normalize_date_values([start, end], "date range")
    if values.iloc[0] > values.iloc[1]:
        raise DataQualityError("start date is after end date")
    return pd.Timestamp(values.iloc[0]), pd.Timestamp(values.iloc[1])
