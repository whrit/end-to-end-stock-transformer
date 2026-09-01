"""Simple JSON persistence for train-fit preprocessing statistics."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from market_guided_stock_alpha.dataset.normalization import (
    DatasetStatistics,
    FeatureStatistics,
)


LABEL_NORMALIZATION_CONTRACT = {
    "scope": "daily_cross_section",
    "lower_quantile": 0.025,
    "upper_quantile": 0.975,
    "quantile_method": "linear",
    "zscore_ddof": 0,
    "minimum_finite_labels": 2,
}


def preprocessing_to_dict(statistics: DatasetStatistics) -> Dict[str, Any]:
    families: Dict[str, Any] = {}
    for family, value in sorted(statistics.by_family.items()):
        value.validate()
        families[family] = {
            "feature_names": list(value.feature_names),
            "median": value.median.tolist(),
            "mad": value.mad.tolist(),
            "scale": value.scale.tolist(),
            "finite_count": value.finite_count.tolist(),
            "zero_mad_mask": value.zero_mad_mask.tolist(),
            "epsilon": value.epsilon,
            "mad_multiplier": value.mad_multiplier,
            "clip_min": value.clip_min,
            "clip_max": value.clip_max,
        }
    return {
        "format_version": 1,
        "fit_scope": "training_dates_only",
        "feature_statistics": families,
        "label_normalization": dict(LABEL_NORMALIZATION_CONTRACT),
    }


def preprocessing_from_dict(document: Mapping[str, Any]) -> DatasetStatistics:
    if document.get("format_version") != 1:
        raise ValueError("unsupported preprocessing format")
    values = {}
    for family, item in dict(document.get("feature_statistics", {})).items():
        values[family] = FeatureStatistics(
            feature_names=tuple(item["feature_names"]),
            median=np.asarray(item["median"], dtype="float64"),
            mad=np.asarray(item["mad"], dtype="float64"),
            scale=np.asarray(item["scale"], dtype="float64"),
            finite_count=np.asarray(item["finite_count"], dtype="int64"),
            zero_mad_mask=np.asarray(item["zero_mad_mask"], dtype="bool"),
            epsilon=float(item["epsilon"]),
            mad_multiplier=float(item["mad_multiplier"]),
            clip_min=float(item["clip_min"]),
            clip_max=float(item["clip_max"]),
        )
        values[family].validate()
    if not values:
        raise ValueError("preprocessing state contains no feature family")
    return DatasetStatistics(values)


def save_preprocessing(path: Path, statistics: DatasetStatistics) -> Dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = preprocessing_to_dict(statistics)
    temporary = target.with_name(target.name + ".partial")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return document


def load_preprocessing(path: Path) -> DatasetStatistics:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return preprocessing_from_dict(document)
