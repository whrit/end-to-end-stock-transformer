"""Seed robustness summaries and the sole canonical prediction ensemble."""

from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from market_guided_stock_alpha.evaluation.metrics import (
    PREDICTION_KEY_COLUMNS,
    PREDICTION_VALUE_COLUMNS,
    metric_names,
    validate_prediction_frame,
)


def raw_equal_weight_ensemble(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Strict key-aligned arithmetic mean of raw predictions."""
    if len(frames) < 2:
        raise ValueError("prediction ensemble requires at least two seeds")
    canonical = [validate_prediction_frame(frame) for frame in frames]
    indexed = [
        frame.set_index(list(PREDICTION_KEY_COLUMNS)).sort_index() for frame in canonical
    ]
    reference = indexed[0]
    for position, candidate in enumerate(indexed[1:], start=1):
        if not candidate.index.equals(reference.index):
            missing = reference.index.difference(candidate.index)
            extra = candidate.index.difference(reference.index)
            raise ValueError(
                "seed {} prediction keys differ (missing={}, extra={})".format(
                    position, len(missing), len(extra)
                )
            )
        for column in ("label_raw", "label"):
            if not np.allclose(
                reference[column].to_numpy(dtype="float64"),
                candidate[column].to_numpy(dtype="float64"),
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            ):
                raise ValueError("seed prediction {} values differ".format(column))
        if not np.array_equal(
            reference["label_mask"].to_numpy(dtype="bool"),
            candidate["label_mask"].to_numpy(dtype="bool"),
        ):
            raise ValueError("seed prediction label_mask values differ")
    prediction = np.mean(
        np.stack(
            [item["prediction"].to_numpy(dtype="float64") for item in indexed],
            axis=0,
        ),
        axis=0,
    )
    output = reference.copy()
    output["prediction"] = prediction
    return output.reset_index().loc[
        :, list(PREDICTION_KEY_COLUMNS + PREDICTION_VALUE_COLUMNS)
    ]


def summarize_seed_robustness(
    seed_overalls: Mapping[int, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Equal-weight mean and sample standard deviation; this is not an ensemble."""
    if len(seed_overalls) < 2:
        raise ValueError("seed robustness requires at least two seeds")
    result: Dict[str, Any] = {
        "seeds": [int(seed) for seed in seed_overalls],
        "seed_count": len(seed_overalls),
        "standard_deviation_ddof": 1,
        "kind": "individual_seed_robustness_not_prediction_ensemble",
        "metrics": {},
    }
    for name in metric_names():
        values = np.asarray(
            [row[name] for row in seed_overalls.values() if row.get(name) is not None],
            dtype="float64",
        )
        result["metrics"][name] = {
            "mean": float(values.mean()) if len(values) else None,
            "sample_std": float(values.std(ddof=1)) if len(values) >= 2 else None,
            "seed_count": int(len(values)),
        }
    return result
