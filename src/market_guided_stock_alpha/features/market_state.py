"""Thin shared 21-feature-per-index market-state mathematics."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.features.rolling_ops import (
    rolling_mean,
    rolling_std,
    sanitize_non_finite,
)
from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    normalize_date_values,
)


MARKET_WINDOWS = (5, 10, 20, 30, 60)


class MarketStateDataError(DataQualityError):
    """Raised when canonical index history violates the market-state contract."""


def market_prefix(universe: str) -> str:
    return "MKT_{}_".format(universe.upper())


def build_market_manifest(
    universes: Sequence[str], total_feature_count: int
) -> pd.DataFrame:
    """Build the fixed 21-feature block for each universe in order."""
    rows: List[Dict[str, Any]] = []
    for universe in universes:
        prefix = market_prefix(universe)
        rows.append(
            {
                "feature_name": prefix + "RET1",
                "universe": universe,
                "family": "RET1",
                "window": 1,
                "expression": "close/Ref(close,1)-1",
                "source": "mkt_index.close",
            }
        )
        for window in MARKET_WINDOWS:
            definitions = (
                ("RET_MEAN", "Mean(market_return,d)", "mkt_index.close"),
                ("RET_STD", "Std(market_return,d)", "mkt_index.close"),
                (
                    "AMOUNT_MEAN_RATIO",
                    "Mean(amount,d)/current_amount",
                    "mkt_index.amount",
                ),
                (
                    "AMOUNT_STD_RATIO",
                    "Std(amount,d)/current_amount",
                    "mkt_index.amount",
                ),
            )
            for family, expression, source in definitions:
                rows.append(
                    {
                        "feature_name": prefix + family + str(window),
                        "universe": universe,
                        "family": family,
                        "window": window,
                        "expression": expression,
                        "source": source,
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.insert(0, "feature_index", range(len(manifest)))
    if (
        len(manifest) != total_feature_count
        or manifest["feature_name"].duplicated().any()
    ):
        raise ValueError(
            "market-state manifest is not {} unique features".format(
                total_feature_count
            )
        )
    return manifest


def transform_market_history(
    index_market_history: pd.DataFrame,
    universes: Sequence[str],
    feature_names: Sequence[str],
    contract_name: str,
) -> Tuple[pd.DataFrame, int]:
    """Apply the shared 21-feature formulas and return sanitation count."""
    required = ("trade_date", "universe", "close", "amount")
    missing = [
        name for name in required if name not in index_market_history.columns
    ]
    if missing:
        raise ValueError(
            "{} history missing columns: {}".format(
                contract_name, ",".join(missing)
            )
        )
    frame = index_market_history.loc[:, list(required)].copy(deep=True)
    frame["trade_date"] = normalize_date_values(
        frame["trade_date"], "{} history trade_date".format(contract_name)
    ).to_numpy()
    frame["universe"] = frame["universe"].map(
        lambda value: str(value).strip().lower()
    )
    if not frame["universe"].isin(tuple(universes)).all():
        raise MarketStateDataError(
            "{} history contains an unknown universe".format(contract_name)
        )

    non_finite_count = 0

    def clean(values: pd.Series) -> pd.Series:
        nonlocal non_finite_count
        inherited = int(values.attrs.get("non_finite_to_nan_count", 0))
        numeric = pd.to_numeric(values, errors="coerce").astype("float64")
        non_finite_count += inherited + int(
            (numeric.notna() & ~np.isfinite(numeric.to_numpy())).sum()
        )
        return sanitize_non_finite(numeric)

    def divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return clean(clean(numerator) / clean(denominator))

    frame["close"] = clean(frame["close"])
    frame["amount"] = clean(frame["amount"])
    parts = []
    for universe in universes:
        group = frame.loc[frame["universe"] == universe].copy()
        if group.empty:
            raise MarketStateDataError(
                "{} history has no rows for {}".format(contract_name, universe)
            )
        if group.duplicated(["trade_date"]).any():
            raise MarketStateDataError(
                "{} history has duplicate dates for {}".format(
                    contract_name, universe
                )
            )
        group = group.sort_values("trade_date", kind="mergesort").reset_index(
            drop=True
        )
        close = group["close"]
        amount = group["amount"]
        market_return = clean(divide(close, close.shift(1)) - 1.0)
        data: Dict[str, Any] = {"trade_date": group["trade_date"].to_numpy()}
        prefix = market_prefix(universe)
        data[prefix + "RET1"] = market_return.to_numpy()
        for window in MARKET_WINDOWS:
            data[prefix + "RET_MEAN" + str(window)] = clean(
                rolling_mean(market_return, window)
            ).to_numpy()
            data[prefix + "RET_STD" + str(window)] = clean(
                rolling_std(market_return, window)
            ).to_numpy()
            data[prefix + "AMOUNT_MEAN_RATIO" + str(window)] = divide(
                rolling_mean(amount, window), amount
            ).to_numpy()
            data[prefix + "AMOUNT_STD_RATIO" + str(window)] = divide(
                rolling_std(amount, window), amount
            ).to_numpy()
        parts.append(pd.DataFrame(data))

    output = parts[0]
    for part in parts[1:]:
        output = output.merge(
            part,
            on="trade_date",
            how="outer",
            validate="one_to_one",
            sort=True,
        )
    output = output.sort_values("trade_date", kind="mergesort").reset_index(
        drop=True
    )
    return output.loc[:, ["trade_date"] + list(feature_names)], non_finite_count
