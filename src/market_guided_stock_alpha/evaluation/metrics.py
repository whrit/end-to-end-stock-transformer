"""Faithful daily cross-sectional evaluation for OOS prediction artifacts."""

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd


PREDICTION_KEY_COLUMNS = ("target_id", "trade_date", "sec_code")
PREDICTION_VALUE_COLUMNS = (
    "prediction",
    "label_raw",
    "label",
    "label_mask",
)
PREDICTION_COLUMNS = PREDICTION_KEY_COLUMNS + PREDICTION_VALUE_COLUMNS


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable canonical copy and reject ambiguous observations."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("prediction artifact must be a pandas DataFrame")
    missing = set(PREDICTION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError("prediction columns missing: {}".format(sorted(missing)))
    output = frame.loc[:, PREDICTION_COLUMNS].copy()
    output["target_id"] = output["target_id"].astype(str)
    output["trade_date"] = pd.to_datetime(output["trade_date"], errors="coerce")
    if output["trade_date"].isna().any():
        raise ValueError("prediction trade_date contains invalid values")
    if output["trade_date"].dt.tz is not None:
        output["trade_date"] = output["trade_date"].dt.tz_localize(None)
    output["trade_date"] = output["trade_date"].dt.normalize()
    output["sec_code"] = output["sec_code"].astype(str)
    if output["target_id"].eq("").any() or output["sec_code"].eq("").any():
        raise ValueError("prediction keys must be non-empty")
    for column in ("prediction", "label_raw", "label"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["label_mask"] = output["label_mask"].astype(bool)
    if output.duplicated(list(PREDICTION_KEY_COLUMNS)).any():
        raise ValueError("prediction keys are not unique")
    return output.sort_values(
        list(PREDICTION_KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def _average_tie_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype="float64")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left.astype("float64"), right.astype("float64"))[0, 1])


def _metric_value(total: float, count: int) -> Any:
    return float(total / count) if count else None


def evaluate_target(
    frame: pd.DataFrame, minimum_count: int = 2
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate one target using the formal 04 daily and pooling semantics.

    IC/RankIC and percentile portfolios use raw labels.  MSE/R-squared use the
    daily normalized training label.  Top5/Top10 mean top 5%/10%, not five or
    ten securities, and are raw decimal returns without benchmark subtraction.
    """
    if type(minimum_count) is not int or minimum_count < 2:
        raise ValueError("minimum_count must be an integer of at least two")
    values = validate_prediction_frame(frame)
    target_ids = tuple(values["target_id"].drop_duplicates())
    if len(target_ids) != 1:
        raise ValueError("target evaluation requires exactly one target_id")

    daily_rows = []
    metric_sums = {name: 0.0 for name in ("ic", "rank_ic", "top5", "top10")}
    metric_counts = {name: 0 for name in metric_sums}
    sse = 0.0
    label_sum = 0.0
    label_sum_sq = 0.0
    observation_count = 0

    for day, group in values.groupby("trade_date", sort=True):
        prediction = group["prediction"].to_numpy(dtype="float64")
        raw = group["label_raw"].to_numpy(dtype="float64")
        processed = group["label"].to_numpy(dtype="float64")
        declared = group["label_mask"].to_numpy(dtype="bool")
        raw_mask = declared & np.isfinite(prediction) & np.isfinite(raw)
        processed_mask = declared & np.isfinite(prediction) & np.isfinite(processed)

        row: Dict[str, Any] = {
            "target_id": target_ids[0],
            "trade_date": pd.Timestamp(day),
            "raw_valid_count": int(raw_mask.sum()),
            "processed_valid_count": int(processed_mask.sum()),
            "ic": np.nan,
            "rank_ic": np.nan,
            "top5": np.nan,
            "top10": np.nan,
        }
        if int(raw_mask.sum()) >= minimum_count:
            prediction_raw = prediction[raw_mask]
            label_raw = raw[raw_mask]
            if prediction_raw.std(ddof=0) > 0.0 and label_raw.std(ddof=0) > 0.0:
                ic = _correlation(prediction_raw, label_raw)
                rank_ic = _correlation(
                    _average_tie_ranks(prediction_raw),
                    _average_tie_ranks(label_raw),
                )
                if np.isfinite(ic):
                    row["ic"] = ic
                    metric_sums["ic"] += ic
                    metric_counts["ic"] += 1
                if np.isfinite(rank_ic):
                    row["rank_ic"] = rank_ic
                    metric_sums["rank_ic"] += rank_ic
                    metric_counts["rank_ic"] += 1

            # Stable ascending order faithfully preserves the historical tie rule.
            order = np.argsort(prediction_raw, kind="mergesort")
            for name, fraction in (("top5", 0.05), ("top10", 0.10)):
                top_count = max(1, int(np.ceil(len(order) * fraction)))
                top_value = float(label_raw[order[-top_count:]].mean())
                row[name] = top_value
                metric_sums[name] += top_value
                metric_counts[name] += 1

        if processed_mask.any():
            prediction_processed = prediction[processed_mask]
            label_processed = processed[processed_mask]
            residual = prediction_processed - label_processed
            sse += float(np.dot(residual, residual))
            label_sum += float(label_processed.sum())
            label_sum_sq += float(np.dot(label_processed, label_processed))
            observation_count += int(processed_mask.sum())
        daily_rows.append(row)

    mse = sse / observation_count if observation_count else None
    if observation_count:
        sst = label_sum_sq - (label_sum * label_sum / observation_count)
        r2 = 1.0 - sse / sst if sst > 0.0 else None
    else:
        r2 = None
    summary: Dict[str, Any] = {
        "target_id": target_ids[0],
        "test_trading_day_count": int(values["trade_date"].nunique()),
        "ic": _metric_value(metric_sums["ic"], metric_counts["ic"]),
        "rank_ic": _metric_value(metric_sums["rank_ic"], metric_counts["rank_ic"]),
        "top5": _metric_value(metric_sums["top5"], metric_counts["top5"]),
        "top10": _metric_value(metric_sums["top10"], metric_counts["top10"]),
        "mse": None if mse is None else float(mse),
        "r2": None if r2 is None else float(r2),
        "evaluation_observation_count": int(observation_count),
        "evaluation_sse": float(sse),
        "evaluation_label_sum": float(label_sum),
        "evaluation_label_sum_sq": float(label_sum_sq),
    }
    for name in metric_sums:
        summary["{}_sum".format(name)] = float(metric_sums[name])
        summary["{}_valid_day_count".format(name)] = int(metric_counts[name])
    return summary, pd.DataFrame(daily_rows)


def metric_names() -> Tuple[str, ...]:
    return ("ic", "rank_ic", "top5", "top10", "mse", "r2")
