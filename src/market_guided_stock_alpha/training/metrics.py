"""Minimal trainer-validation summaries; formal evaluation is a later layer."""

import math
from typing import Any, Dict, Iterable

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype="float64")
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[order[left:right]] = (left + right - 1) / 2.0
        left = right
    return ranks


def daily_rank_ic(predictions: Any, labels_raw: Any, label_mask: Any) -> float:
    prediction = np.asarray(predictions, dtype="float64")
    target = np.asarray(labels_raw, dtype="float64")
    mask = np.asarray(label_mask, dtype="bool")
    if prediction.ndim != 1 or target.shape != prediction.shape or mask.shape != prediction.shape:
        raise ValueError("daily RankIC shapes differ")
    usable = mask & np.isfinite(prediction) & np.isfinite(target)
    if int(usable.sum()) < 2:
        return float("nan")
    left = _average_ranks(prediction[usable])
    right = _average_ranks(target[usable])
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else float("nan")


def aggregate_validation_losses(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    values = list(rows)
    total_count = sum(int(row["valid_count"]) for row in values)
    if not values or total_count <= 0:
        raise ValueError("validation produced no supervised observation")
    weighted = sum(float(row["loss"]) * int(row["valid_count"]) for row in values)
    rank_ics = np.asarray(
        [float(row["rank_ic"]) for row in values if math.isfinite(float(row["rank_ic"]))],
        dtype="float64",
    )
    return {
        "validation_loss": float(weighted / total_count),
        "mean_daily_validation_loss": float(np.mean([row["loss"] for row in values])),
        "validation_date_count": len(values),
        "validation_observation_count": total_count,
        "mean_rank_ic": float(rank_ics.mean()) if len(rank_ics) else float("nan"),
        "rank_ic_date_count": int(len(rank_ics)),
    }
