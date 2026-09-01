"""Shared neural prediction and historical LightGBM sequence prediction."""

from typing import Any, List

import numpy as np
import pandas as pd
import torch

from market_guided_stock_alpha.modeling.lightgbm import flatten_sequence_sample
from market_guided_stock_alpha.modeling.transport import (
    forward_date_batch,
    move_date_batch,
)
from market_guided_stock_alpha.training.seed import resolve_device


def predict_neural_model(
    model: torch.nn.Module,
    loader: Any,
    target_id: str,
    device: Any = None,
) -> pd.DataFrame:
    target_device = resolve_device(device)
    model.to(target_device)
    model.eval()
    rows: List[pd.DataFrame] = []
    with torch.no_grad():
        for cpu_batch in loader:
            batch = move_date_batch(cpu_batch, target_device)
            prediction = forward_date_batch(model, batch).detach().cpu().numpy()
            rows.append(
                pd.DataFrame(
                    {
                        "target_id": str(target_id),
                        "trade_date": pd.Timestamp(batch.prediction_date),
                        "sec_code": list(batch.security_codes),
                        "prediction": prediction.astype("float64"),
                        "label_raw": batch.labels_raw.detach().cpu().numpy().astype("float64"),
                        "label": batch.labels.detach().cpu().numpy().astype("float64"),
                        "label_mask": batch.label_mask.detach().cpu().numpy().astype("bool"),
                    }
                )
            )
    if not rows:
        raise RuntimeError("prediction loader is empty")
    from market_guided_stock_alpha.evaluation.metrics import validate_prediction_frame

    return validate_prediction_frame(pd.concat(rows, ignore_index=True))


def predict_lightgbm_model(model: Any, dataset: Any, target_id: str) -> pd.DataFrame:
    """Predict from the formal eight-day day-then-feature flattened view."""
    rows = []
    for position in range(len(dataset)):
        sample = dataset[position]
        prediction = np.asarray(
            model.predict(flatten_sequence_sample(sample)), dtype="float64"
        )
        rows.append(
            pd.DataFrame(
                {
                    "target_id": str(target_id),
                    "trade_date": pd.Timestamp(sample.prediction_date),
                    "sec_code": list(sample.security_codes),
                    "prediction": prediction,
                    "label_raw": np.asarray(sample.labels_raw, dtype="float64"),
                    "label": np.asarray(sample.labels, dtype="float64"),
                    "label_mask": np.asarray(sample.label_mask, dtype="bool"),
                }
            )
        )
    if not rows:
        raise RuntimeError("prediction dataset is empty")
    from market_guided_stock_alpha.evaluation.metrics import validate_prediction_frame

    return validate_prediction_frame(pd.concat(rows, ignore_index=True))
