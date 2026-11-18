"""Simple NPY/mmap derived-feature artifact with a verified JSON manifest."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.artifacts import (
    atomic_write_json,
    atomic_write_npy,
    sha256_mapping,
    sha256_path,
)
from market_guided_stock_alpha.data.contracts import normalize_security_code


SCHEMA_VERSION = 1
EPOCH = np.datetime64("1970-01-01", "D")


def dates_to_epoch_days(values: Sequence[Any]) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if dates.hasnans:
        raise ValueError("artifact dates contain invalid values")
    dates = dates.tz_localize(None).normalize()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("artifact dates must be unique and increasing")
    return (dates.to_numpy(dtype="datetime64[D]") - EPOCH).astype("int32")


def epoch_days_to_dates(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(EPOCH + np.asarray(values, dtype="timedelta64[D]"))


@dataclass(frozen=True)
class FeatureArtifact:
    path: Path
    family: str
    values: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: Tuple[str, ...]
    security_codes: Tuple[str, ...]
    source_observed: np.ndarray
    feature_finite_count: np.ndarray
    bar_coverage: Optional[np.ndarray]
    manifest: Dict[str, Any]


def _array_file(family: str) -> str:
    return "{}_raw.npy".format(family)


# Bound per-pass temporaries (isinf/isfinite masks) to ~32M elements so a
# disk-backed [D,S,F] array is scanned without materializing full-size masks.
_SCAN_ELEMENTS = 32 * 1024 * 1024


def _row_blocks(raw: np.ndarray):
    per_row = max(1, int(np.prod(raw.shape[1:])))
    step = max(1, _SCAN_ELEMENTS // per_row)
    for start in range(0, raw.shape[0], step):
        yield slice(start, start + step)


def write_feature_artifact(
    path: Path,
    family: str,
    values: np.ndarray,
    dates: Sequence[Any],
    feature_names: Sequence[str],
    security_codes: Optional[Sequence[str]] = None,
    source_observed: Optional[np.ndarray] = None,
    bar_coverage: Optional[np.ndarray] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    adopt_values_file: bool = False,
) -> Dict[str, Any]:
    """Publish one immutable derived artifact directory atomically.

    With ``adopt_values_file`` the values must be an ``np.lib.format.open_memmap``
    ``.npy`` on the target's filesystem; that file is moved into the artifact
    instead of being copied, so large builds never hold two copies on disk.
    """
    target = Path(path)
    if target.exists():
        raise FileExistsError("artifact target already exists: {}".format(target))
    family_name = str(family).strip()
    if not family_name or not family_name.replace("_", "").isalnum():
        raise ValueError("artifact family must be a simple identifier")
    names = tuple(str(name) for name in feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("artifact feature names must be non-empty and unique")
    if adopt_values_file and not (
        isinstance(values, np.memmap)
        and values.filename is not None
        and str(values.filename).endswith(".npy")
    ):
        raise ValueError("adopt_values_file requires an open_memmap .npy array")
    raw = values if adopt_values_file else np.asarray(values)
    if raw.ndim not in (2, 3) or raw.shape[-1] != len(names):
        raise ValueError("artifact values must be [D,F] or [D,S,F]")
    if any(np.isinf(raw[block]).any() for block in _row_blocks(raw)):
        raise ValueError("artifact values must not contain infinity")
    date_ids = dates_to_epoch_days(dates)
    if len(date_ids) == 0:
        raise ValueError("artifact requires at least one date")
    if len(date_ids) != raw.shape[0]:
        raise ValueError("artifact date count does not match values")
    codes: Tuple[str, ...] = ()
    if raw.ndim == 3:
        if security_codes is None:
            raise ValueError("cross-sectional artifact requires security_codes")
        codes = tuple(
            normalize_security_code(value, "artifact security code")
            for value in security_codes
        )
        if len(codes) != raw.shape[1] or len(codes) != len(set(codes)):
            raise ValueError("artifact security order is invalid")
    elif security_codes:
        raise ValueError("market-only artifact must not contain security_codes")
    row_shape = raw.shape[:-1]
    observed = (
        np.asarray(source_observed, dtype="bool")
        if source_observed is not None
        else np.ones(row_shape, dtype="bool")
    )
    if observed.shape != row_shape:
        raise ValueError("source_observed shape does not match artifact rows")
    finite_count = np.empty(row_shape, dtype="uint16")
    for block in _row_blocks(raw):
        finite_count[block] = np.isfinite(raw[block]).sum(axis=-1)
    coverage = None
    if bar_coverage is not None:
        coverage = np.asarray(bar_coverage, dtype="float32")
        if coverage.shape != row_shape:
            raise ValueError("bar_coverage shape does not match artifact rows")
        if np.any(np.isfinite(coverage) & ((coverage < 0) | (coverage > 1))):
            raise ValueError("bar_coverage must be within [0,1]")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=target.name + ".partial-", dir=target.parent)
    )
    try:
        arrays = {
            "date_ids.npy": date_ids,
            _array_file(family_name): raw,
            "source_observed.npy": observed,
            "feature_finite_count.npy": finite_count,
        }
        if codes:
            # CN codes are 9 chars (stays U9); US FIGIs are 12.
            width = max(len(code) for code in codes)
            arrays["security_ids.npy"] = np.asarray(codes, dtype="U{}".format(width))
        if coverage is not None:
            arrays["bar_coverage.npy"] = coverage
        array_contract = {}
        for filename, array in arrays.items():
            if adopt_values_file and array is raw:
                raw.flush()
                os.replace(str(raw.filename), str(temporary / filename))
                digest = sha256_path(temporary / filename)
            else:
                digest = atomic_write_npy(temporary / filename, array)
            array_contract[filename] = {
                "sha256": digest,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "family": family_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "value_file": _array_file(family_name),
            "feature_count": len(names),
            "feature_names": list(names),
            "feature_order_sha256": sha256_mapping({"feature_names": list(names)}),
            "date_count": int(len(date_ids)),
            "date_start": str(epoch_days_to_dates(date_ids).min().date()),
            "date_end": str(epoch_days_to_dates(date_ids).max().date()),
            "security_count": len(codes),
            "security_codes": list(codes),
            "source_observed_semantics": (
                "true when source data passes the family eligibility contract"
            ),
            "arrays": array_contract,
            "provenance": dict(provenance or {}),
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(str(temporary), str(target))
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_feature_artifact(
    path: Path,
    expected_family: Optional[str] = None,
    verify_hashes: bool = True,
    mmap_mode: Optional[str] = "r",
) -> FeatureArtifact:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("artifact manifest is missing: {}".format(root))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")
    family = str(manifest.get("family", ""))
    if expected_family is not None and family != expected_family:
        raise ValueError("artifact family does not match {}".format(expected_family))
    names = tuple(manifest.get("feature_names", []))
    if (
        len(names) != manifest.get("feature_count")
        or len(names) != len(set(names))
        or sha256_mapping({"feature_names": list(names)})
        != manifest.get("feature_order_sha256")
    ):
        raise ValueError("artifact feature-order metadata is invalid")
    arrays = {}
    for filename, contract in manifest.get("arrays", {}).items():
        array_path = root / filename
        if verify_hashes and sha256_path(array_path) != contract.get("sha256"):
            raise ValueError("artifact array hash mismatch: {}".format(filename))
        value = np.load(array_path, mmap_mode=mmap_mode, allow_pickle=False)
        if list(value.shape) != contract.get("shape") or str(value.dtype) != contract.get("dtype"):
            raise ValueError("artifact array contract mismatch: {}".format(filename))
        arrays[filename] = value
    required = {
        "date_ids.npy",
        manifest["value_file"],
        "source_observed.npy",
        "feature_finite_count.npy",
    }
    if not required.issubset(arrays):
        raise ValueError("artifact arrays are incomplete")
    values = arrays[manifest["value_file"]]
    if values.shape[-1] != len(names) or np.isinf(values).any():
        raise ValueError("artifact value contract is invalid")
    finite_count = np.isfinite(values).sum(axis=-1).astype("uint16")
    if not np.array_equal(finite_count, arrays["feature_finite_count.npy"]):
        raise ValueError("artifact finite-count metadata is inconsistent")
    codes = tuple(manifest.get("security_codes", []))
    row_shape = values.shape[:-1]
    observed = arrays["source_observed.npy"]
    if observed.shape != row_shape or observed.dtype != np.dtype("bool"):
        raise ValueError("artifact source-observed metadata is inconsistent")
    coverage = arrays.get("bar_coverage.npy")
    if coverage is not None:
        if coverage.shape != row_shape:
            raise ValueError("artifact bar-coverage shape is inconsistent")
        if np.any(np.isfinite(coverage) & ((coverage < 0) | (coverage > 1))):
            raise ValueError("artifact bar coverage is outside [0,1]")
    if values.ndim == 3:
        if len(codes) != values.shape[1] or "security_ids.npy" not in arrays:
            raise ValueError("artifact security metadata is inconsistent")
        if tuple(arrays["security_ids.npy"].tolist()) != codes:
            raise ValueError("artifact security IDs are inconsistent")
    elif codes or "security_ids.npy" in arrays:
        raise ValueError("market-only artifact contains security metadata")
    dates = epoch_days_to_dates(arrays["date_ids.npy"])
    if (
        len(dates) != values.shape[0]
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or len(dates) != manifest.get("date_count")
        or str(dates.min().date()) != manifest.get("date_start")
        or str(dates.max().date()) != manifest.get("date_end")
    ):
        raise ValueError("artifact date metadata is inconsistent")
    if len(codes) != manifest.get("security_count"):
        raise ValueError("artifact security count is inconsistent")
    for value in arrays.values():
        value.flags.writeable = False
    return FeatureArtifact(
        path=root,
        family=family,
        values=values,
        dates=dates,
        feature_names=names,
        security_codes=codes,
        source_observed=observed,
        feature_finite_count=arrays["feature_finite_count.npy"],
        bar_coverage=coverage,
        manifest=manifest,
    )
