"""Correct multi-target OOS aggregation from sufficient statistics."""

from typing import Any, Dict, Iterable, Mapping


def aggregate_overall(targets: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(targets)
    if not rows:
        raise ValueError("overall aggregation requires at least one target")
    output: Dict[str, Any] = {"target_count": len(rows)}
    for name in ("ic", "rank_ic", "top5", "top10"):
        total = sum(float(row["{}_sum".format(name)]) for row in rows)
        count = sum(int(row["{}_valid_day_count".format(name)]) for row in rows)
        output[name] = float(total / count) if count else None
        output["{}_valid_day_count".format(name)] = int(count)

    count = sum(int(row["evaluation_observation_count"]) for row in rows)
    sse = sum(float(row["evaluation_sse"]) for row in rows)
    label_sum = sum(float(row["evaluation_label_sum"]) for row in rows)
    label_sum_sq = sum(float(row["evaluation_label_sum_sq"]) for row in rows)
    output.update(
        {
            "evaluation_observation_count": int(count),
            "evaluation_sse": float(sse),
            "evaluation_label_sum": float(label_sum),
            "evaluation_label_sum_sq": float(label_sum_sq),
            "mse": float(sse / count) if count else None,
        }
    )
    sst = label_sum_sq - label_sum * label_sum / count if count else 0.0
    output["r2"] = float(1.0 - sse / sst) if sst > 0.0 else None
    return output
