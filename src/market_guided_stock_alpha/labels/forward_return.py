"""Auxiliary forward-return labels on the complete trading calendar.

This baseline label is not the formal relative-excess target used by the
derived-label research pipeline.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from market_guided_stock_alpha.labels.normalization import normalize_security_code


LABEL_COLUMNS = (
    "trade_date",
    "sec_code",
    "label_date_t1",
    "label_date_t5",
    "close_t1",
    "close_t5",
    "label_raw",
    "label_available",
)


class ForwardReturnLabelGenerator:
    """Generate ``close(t+5) / close(t+1) - 1`` on the market-date axis."""

    def __init__(self, horizon_end: int = 5, horizon_start: int = 1) -> None:
        if type(horizon_start) is not int or horizon_start <= 0:
            raise ValueError("horizon_start must be a positive integer")
        if type(horizon_end) is not int or horizon_end < horizon_start:
            raise ValueError("horizon_end must be an integer >= horizon_start")
        self.horizon_start = horizon_start
        self.horizon_end = horizon_end

    def get_label_spec(self) -> Dict[str, Any]:
        return {
            "name": "master_forward_return_t1_to_t5",
            "formula": "close(t+5)/close(t+1)-1",
            "formula_expression": "Ref($close,-5)/Ref($close,-1)-1",
            "horizon_start_trading_days": self.horizon_start,
            "horizon_end_trading_days": self.horizon_end,
            "calendar_axis": "complete_market_trading_calendar",
            "price_field": "canonical_adjusted_close",
        }

    @staticmethod
    def _normalize_target(target_grid: pd.DataFrame) -> pd.DataFrame:
        required = ("trade_date", "sec_code")
        missing = [name for name in required if name not in target_grid.columns]
        if missing:
            raise ValueError(
                "Label target grid missing columns: {}".format(",".join(missing))
            )
        target = target_grid.loc[:, list(required)].copy(deep=True)
        target["trade_date"] = pd.to_datetime(target["trade_date"], errors="coerce")
        if target["trade_date"].isna().any():
            raise ValueError("Label target grid has invalid trade_date")
        target["trade_date"] = (
            target["trade_date"].dt.tz_localize(None).dt.normalize()
        )
        target["sec_code"] = [
            normalize_security_code(value) for value in target["sec_code"]
        ]
        if target.duplicated(["trade_date", "sec_code"]).any():
            raise ValueError("Label target grid has duplicate keys")
        return target

    @staticmethod
    def _normalize_close_history(close_history: pd.DataFrame) -> pd.DataFrame:
        required = ("trade_date", "sec_code", "close")
        missing = [name for name in required if name not in close_history.columns]
        if missing:
            raise ValueError(
                "Label close history missing columns: {}".format(",".join(missing))
            )
        history = close_history.loc[:, list(required)].copy(deep=True)
        history["trade_date"] = pd.to_datetime(
            history["trade_date"], errors="coerce"
        )
        if history["trade_date"].isna().any():
            raise ValueError("Label close history has invalid trade_date")
        history["trade_date"] = (
            history["trade_date"].dt.tz_localize(None).dt.normalize()
        )
        history["sec_code"] = [
            normalize_security_code(value) for value in history["sec_code"]
        ]
        history["close"] = pd.to_numeric(
            history["close"], errors="coerce"
        ).astype("float64")
        history.loc[~np.isfinite(history["close"].to_numpy()), "close"] = np.nan
        if history.duplicated(["trade_date", "sec_code"]).any():
            raise ValueError("Label close history has duplicate keys")
        return history

    def generate(
        self,
        close_history: pd.DataFrame,
        target_grid: pd.DataFrame,
        trading_calendar: Any,
    ) -> pd.DataFrame:
        target = self._normalize_target(target_grid)
        if target.empty:
            return pd.DataFrame(columns=list(LABEL_COLUMNS))
        history = self._normalize_close_history(close_history)
        target_start = target["trade_date"].min()
        target_end = target["trade_date"].max()
        label_future_end = trading_calendar.next_trading_date(
            target_end, self.horizon_end
        )
        market_dates = pd.DatetimeIndex(
            trading_calendar.load_open_dates(target_start, label_future_end)
        ).tz_localize(None).normalize()
        if market_dates.has_duplicates or not market_dates.is_monotonic_increasing:
            raise ValueError("Label trading calendar must be unique and increasing")
        positions = market_dates.get_indexer(target["trade_date"])
        if (positions < 0).any():
            raise ValueError("Label target grid contains a non-trading date")
        start_positions = positions + self.horizon_start
        end_positions = positions + self.horizon_end
        if (end_positions >= len(market_dates)).any():
            raise ValueError("Label calendar does not cover the complete horizon")

        result = target.copy(deep=True)
        result["label_date_t1"] = market_dates.take(start_positions).to_numpy()
        result["label_date_t5"] = market_dates.take(end_positions).to_numpy()
        close_t1 = history.rename(
            columns={"trade_date": "label_date_t1", "close": "close_t1"}
        )
        close_t5 = history.rename(
            columns={"trade_date": "label_date_t5", "close": "close_t5"}
        )
        result = result.merge(
            close_t1,
            on=["sec_code", "label_date_t1"],
            how="left",
            validate="many_to_one",
            sort=False,
        ).merge(
            close_t5,
            on=["sec_code", "label_date_t5"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        valid = (
            np.isfinite(result["close_t1"].to_numpy(dtype="float64"))
            & np.isfinite(result["close_t5"].to_numpy(dtype="float64"))
            & result["close_t1"].gt(0.0).to_numpy()
            & result["close_t5"].gt(0.0).to_numpy()
        )
        raw = pd.Series(np.nan, index=result.index, dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            raw.loc[valid] = (
                result.loc[valid, "close_t5"] / result.loc[valid, "close_t1"] - 1.0
            )
        valid &= np.isfinite(raw.to_numpy())
        raw.loc[~valid] = np.nan
        result["label_raw"] = raw
        result["label_available"] = valid.astype(bool)
        return result.loc[:, list(LABEL_COLUMNS)]
