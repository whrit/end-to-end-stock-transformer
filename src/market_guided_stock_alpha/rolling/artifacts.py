"""Thin pickle + JSON prediction artifact contract."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from market_guided_stock_alpha.artifacts import (
    atomic_write_json,
    atomic_write_pickle,
    sha256_path,
)
from market_guided_stock_alpha.evaluation.metrics import validate_prediction_frame


def write_prediction_artifact(
    path: Path, frame: pd.DataFrame, metadata: Mapping[str, Any]
) -> Tuple[Path, Path, Dict[str, Any]]:
    target = Path(path)
    if target.suffix.lower() not in (".pkl", ".pickle"):
        raise ValueError("canonical prediction artifact uses pandas pickle")
    values = validate_prediction_frame(frame)
    digest = atomic_write_pickle(values, target)
    manifest = {
        "format_version": 1,
        "artifact_type": "canonical_oos_predictions",
        "file_name": target.name,
        "sha256": digest,
        "row_count": len(values),
        "target_ids": list(values["target_id"].drop_duplicates()),
        "trade_date_start": str(values["trade_date"].min().date()),
        "trade_date_end": str(values["trade_date"].max().date()),
        "columns": list(values.columns),
        "metadata": dict(metadata),
    }
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    atomic_write_json(manifest_path, manifest)
    return target, manifest_path, manifest


def read_prediction_artifact(
    path: Path, verify_hash: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    target = Path(path)
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    if not target.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("prediction artifact or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported prediction artifact format")
    if verify_hash and sha256_path(target) != manifest.get("sha256"):
        raise ValueError("prediction artifact hash mismatch")
    frame = validate_prediction_frame(pd.read_pickle(target))
    if len(frame) != int(manifest.get("row_count", -1)):
        raise ValueError("prediction artifact row count mismatch")
    return frame, manifest
