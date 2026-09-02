"""Transparent file IO for canonical pandas tables and wide minute pickles."""

from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    MINUTE_FIELDS,
    normalize_security_code,
    validate_minute_index,
    validate_minute_stock,
)


def read_table(path: Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("Input file is missing: {}".format(source))
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(source)
    if suffix in (".pkl", ".pickle"):
        value = pd.read_pickle(source)
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Pickle input must contain a pandas DataFrame")
        return value
    raise ValueError("use CSV, Parquet, or pandas pickle input")


def read_wide_minute_pickles(
    field_paths: Mapping[str, Path]
) -> Dict[str, pd.DataFrame]:
    """Read six explicit wide timestamp-by-security pandas pickle files."""
    if set(field_paths) != set(MINUTE_FIELDS):
        raise ValueError("wide minute field paths must equal {}".format(MINUTE_FIELDS))
    result = {}
    for field in MINUTE_FIELDS:
        path = Path(field_paths[field])
        if path.suffix.lower() not in (".pkl", ".pickle"):
            raise ValueError("wide minute {} input must be pandas pickle".format(field))
        value = pd.read_pickle(path)
        if not isinstance(value, pd.DataFrame):
            raise TypeError("wide minute {} pickle must contain DataFrame".format(field))
        result[field] = value
    return result


def wide_minute_frames_to_long(
    frames: Mapping[str, pd.DataFrame], code_column: str = "sec_code"
) -> pd.DataFrame:
    """Convert six aligned timestamp-by-instrument frames to canonical long data."""
    if set(frames) != set(MINUTE_FIELDS):
        raise ValueError("wide minute frames must contain exactly six fields")
    normalized = {}
    reference_index = None
    reference_columns = None
    for field in MINUTE_FIELDS:
        frame = frames[field]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("wide minute {} value must be DataFrame".format(field))
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
        if index.hasnans or index.has_duplicates:
            raise DataQualityError(
                "wide minute {} has invalid or duplicate timestamps".format(field)
            )
        if index.tz is not None:
            index = index.tz_localize(None)
        columns = tuple(normalize_security_code(value) for value in frame.columns)
        if len(columns) != len(set(columns)):
            raise DataQualityError(
                "wide minute {} has duplicate instruments".format(field)
            )
        if reference_index is None:
            reference_index, reference_columns = index, columns
        elif not index.equals(reference_index) or columns != reference_columns:
            raise DataQualityError("wide minute field axes must match exactly")
        value = frame.copy(deep=False)
        value.index = index
        value.columns = columns
        normalized[field] = value
    long_index = pd.MultiIndex.from_product(
        [reference_index, reference_columns], names=["timestamp", code_column]
    )
    pieces = [
        pd.Series(
            normalized[field].to_numpy().reshape(-1),
            index=long_index,
            name=field,
        )
        for field in MINUTE_FIELDS
    ]
    output = pd.concat(pieces, axis=1).reset_index()
    output.columns = ["timestamp", code_column] + list(MINUTE_FIELDS)
    if code_column == "sec_code":
        return validate_minute_stock(output)
    if code_column == "index_code":
        return validate_minute_index(output)
    raise ValueError("code_column must be sec_code or index_code")
