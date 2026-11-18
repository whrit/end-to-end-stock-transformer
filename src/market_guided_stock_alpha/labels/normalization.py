"""Normalization of pandas label artifacts into canonical long form."""

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import US_SECURITY_CODE

LABEL_KEY_COLUMNS = ("trade_date", "sec_code")
_CANONICAL_CODE = re.compile(r"^(\d{6})\.(SH|SZ|BJ)$", re.IGNORECASE)
_PREFIXED_CODE = re.compile(r"^(SH|SZ|BJ)[.\-_]?(\d{6})$", re.IGNORECASE)
_ALT_SUFFIX_CODE = re.compile(
    r"^(\d{6})\.(XSHG|SHSE|XSHE|SZSE|XBJE|BSE)$", re.IGNORECASE
)


def normalize_security_code(value: Any) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError("security code is empty")
    if isinstance(value, (int, np.integer)):
        text = "{:06d}".format(int(value))
    else:
        text = str(value).strip().upper()
    direct = _CANONICAL_CODE.fullmatch(text)
    if direct:
        return "{}.{}".format(direct.group(1), direct.group(2).upper())
    if US_SECURITY_CODE.fullmatch(text):
        return text
    prefixed = _PREFIXED_CODE.fullmatch(text)
    if prefixed:
        return "{}.{}".format(prefixed.group(2), prefixed.group(1).upper())
    alternate = _ALT_SUFFIX_CODE.fullmatch(text)
    if alternate:
        exchange = {
            "XSHG": "SH",
            "SHSE": "SH",
            "XSHE": "SZ",
            "SZSE": "SZ",
            "XBJE": "BJ",
            "BSE": "BJ",
        }[alternate.group(2).upper()]
        return "{}.{}".format(alternate.group(1), exchange)
    digits = re.fullmatch(r"\d{1,6}", text)
    if digits:
        code = digits.group(0).zfill(6)
        exchange = (
            "BJ"
            if code[0] in {"4", "8"}
            else ("SH" if code[0] in {"5", "6", "9"} else "SZ")
        )
        return "{}.{}".format(code, exchange)
    raise ValueError("unsupported security code {!r}".format(value))


def normalize_trade_dates(values: Any) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    series = pd.Series(parsed)
    if series.isna().any():
        raise ValueError("label artifact contains unparseable trade_date values")
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series.dt.normalize().astype("datetime64[ns]")


def _date_score(values: pd.Index) -> float:
    if len(values) == 0:
        return 0.0
    sample = pd.Series(values[: min(50, len(values))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def _code_score(values: pd.Index) -> float:
    if len(values) == 0:
        return 0.0
    good = 0
    sample = values[: min(50, len(values))]
    for value in sample:
        try:
            normalize_security_code(value)
            good += 1
        except (TypeError, ValueError):
            pass
    return good / float(len(sample))


def normalize_label_object(obj: Any, value_column: str) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        frame = obj.rename(value_column).to_frame()
    elif isinstance(obj, pd.DataFrame):
        frame = obj.copy(deep=False)
    else:
        raise TypeError("label artifact must contain pandas Series or DataFrame")

    if isinstance(frame.index, pd.MultiIndex):
        level_names = list(frame.index.names)
        date_level = next(
            (
                i
                for i, name in enumerate(level_names)
                if str(name).lower() in {"trade_date", "date", "datetime"}
            ),
            None,
        )
        code_level = next(
            (
                i
                for i, name in enumerate(level_names)
                if str(name).lower()
                in {"sec_code", "code", "instrument", "symbol"}
            ),
            None,
        )
        if date_level is None:
            date_level = int(
                np.argmax(
                    [
                        _date_score(frame.index.get_level_values(i))
                        for i in range(frame.index.nlevels)
                    ]
                )
            )
        if code_level is None:
            scores = [
                _code_score(frame.index.get_level_values(i))
                for i in range(frame.index.nlevels)
            ]
            scores[date_level] = -1.0
            code_level = int(np.argmax(scores))
        if date_level == code_level:
            raise ValueError(
                "unable to distinguish label date and security index levels"
            )
        names = ["label_index_{}".format(i) for i in range(frame.index.nlevels)]
        names[date_level], names[code_level] = "trade_date", "sec_code"
        frame = frame.rename_axis(index=names).reset_index()
    elif value_column not in frame.columns:
        if _date_score(frame.index) < 0.8:
            frame = frame.reset_index()
        else:
            try:
                stacked = (
                    frame.stack(future_stack=True)
                    .rename(value_column)
                    .reset_index()
                )
            except (TypeError, ValueError):
                stacked = (
                    frame.stack(dropna=False).rename(value_column).reset_index()
                )
            stacked.columns = ["trade_date", "sec_code", value_column]
            frame = stacked
    else:
        frame = frame.reset_index(drop=False)

    lower = {str(column).lower(): column for column in frame.columns}
    date_column = next(
        (
            lower[name]
            for name in ("trade_date", "date", "datetime")
            if name in lower
        ),
        None,
    )
    code_column = next(
        (
            lower[name]
            for name in ("sec_code", "code", "instrument", "symbol")
            if name in lower
        ),
        None,
    )
    if date_column is None or code_column is None:
        raise ValueError(
            "label object does not expose identifiable date/security keys"
        )
    if value_column not in frame.columns:
        candidates = [
            column
            for column in frame.columns
            if column not in {date_column, code_column}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if len(candidates) != 1:
            raise ValueError("label value column {!r} is missing".format(value_column))
        value_column = candidates[0]
    result = frame.loc[:, [date_column, code_column, value_column]].copy()
    result.columns = ["trade_date", "sec_code", "label_value"]
    result["trade_date"] = normalize_trade_dates(result["trade_date"]).to_numpy()
    result["sec_code"] = [
        normalize_security_code(value) for value in result["sec_code"]
    ]
    result["label_value"] = pd.to_numeric(
        result["label_value"], errors="coerce"
    ).astype("float64")
    return result.sort_values(
        list(LABEL_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
