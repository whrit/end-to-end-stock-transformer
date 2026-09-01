"""Exploratory 48-per-index intraday market feature implementation."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


UNIVERSES = ("csi300", "csi500", "csi1000", "csi_all")
FEATURE_SUFFIXES = (
    "RET_OC", "RET_FIRST15", "RET_FIRST30", "RET_FIRST60", "RET_MORNING",
    "RET_AFTERNOON", "RET_LAST60", "RET_LAST30", "RET_LAST15", "RET_PM_MINUS_AM",
    "RV_1M", "RV_5M", "RV_15M", "DAILY_RANGE", "MEAN_MINUTE_RANGE",
    "P90_MINUTE_RANGE", "MAX_ABS_1M_RET", "UPSIDE_SEMIVAR",
    "DOWNSIDE_SEMIVAR", "DOWNSIDE_VAR_SHARE", "PATH_EFFICIENCY",
    "POSITIVE_MINUTE_RATIO", "SIGN_CHANGE_RATIO", "RET_AUTOCORR_LAG1",
    "RET_AUTOCORR_LAG5", "MAX_DRAWDOWN", "MAX_RUNUP", "CLOSE_LOCATION",
    "HIGH_TIME", "LOW_TIME", "RET_SKEW", "RET_EXCESS_KURT", "RET_Q05",
    "RET_Q95", "RET_IQR", "RET_MAD", "JUMP_FRACTION", "MEAN_ABS_RET",
    "FIRST30_AMOUNT_SHARE", "LAST30_AMOUNT_SHARE", "FIRST30_VOLUME_SHARE",
    "LAST30_VOLUME_SHARE", "PM_AM_VOLUME_RATIO", "VOLUME_CV", "VOLUME_HHI",
    "AMOUNT_HHI", "ABSRET_VOLUME_CORR", "VOLUME_TREND",
)


def feature_names() -> List[str]:
    names = [
        "IMKT_{}_{}".format(universe.upper(), suffix)
        for universe in UNIVERSES for suffix in FEATURE_SUFFIXES
    ]
    if len(names) != 192 or len(set(names)) != 192:
        raise ValueError("intraday market feature order is invalid")
    return names


def validate_spec(spec: Mapping[str, Any]) -> None:
    if tuple(spec.get("universes", [])) != UNIVERSES:
        raise ValueError("intraday market universe order changed")
    if spec.get("bars_per_day") != 240:
        raise ValueError("intraday market features require 240 bars")
    if spec.get("feature_count_per_index") != 48 or spec.get("total_feature_count") != 192:
        raise ValueError("intraday market feature counts changed")
    if float(spec.get("epsilon", 0.0)) != 1e-12:
        raise ValueError("intraday market epsilon changed")
    expected = {
        "minute_volume_semantics": "increment_not_cumulative",
        "minute_amount_semantics": "increment_not_cumulative",
        "volume_positive_scale_invariant": True,
        "amount_positive_scale_invariant": True,
        "ohlc_repair": (
            "high=max(high,open,close);low=min(low,open,close);in_memory_only"
        ),
        "block_returns_cross_lunch": False,
        "one_minute_return_preserves_lunch_jump": True,
        "return_shape_moments": (
            "z=(r-mean(r))/std_pop(r);if_std_le_epsilon_skew_and_"
            "excess_kurt_are_zero"
        ),
        "return_excess_kurtosis": "mean(z^4)-3",
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("intraday market source/formula semantics changed")


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.log(numerator / denominator)
    result[~np.isfinite(result)] = np.nan
    return result


def _corr(left: np.ndarray, right: np.ndarray, lag: int = 0) -> float:
    if lag:
        left, right = left[lag:], right[:-lag]
    finite = np.isfinite(left) & np.isfinite(right)
    x, y = left[finite], right[finite]
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _nonoverlap_rv(open_values: np.ndarray, close_values: np.ndarray, size: int) -> float:
    returns = []
    for start, end in ((0, 120), (120, 240)):
        for position in range(start, end, size):
            returns.append(np.log(close_values[position + size - 1] / open_values[position]))
    values = np.asarray(returns, dtype="float64")
    return float(np.sqrt(np.sum(values * values)))


def _segment_return(open_values: np.ndarray, close_values: np.ndarray, start: int, end: int) -> float:
    return float(np.log(close_values[end - 1] / open_values[start]))


class IntradayMarketGenerator:
    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_spec(spec)
        self.spec = dict(spec)
        self.epsilon = float(spec["epsilon"])
        self.names = feature_names()

    def get_feature_names(self) -> List[str]:
        return list(self.names)

    def compute_index_day(
        self,
        open_values: Sequence[float], high_values: Sequence[float],
        low_values: Sequence[float], close_values: Sequence[float],
        volume_values: Sequence[float], amount_values: Sequence[float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        arrays = [
            np.asarray(value, dtype="float64")
            for value in (open_values, high_values, low_values, close_values,
                          volume_values, amount_values)
        ]
        if any(value.shape != (240,) for value in arrays):
            raise ValueError("intraday market index-day requires six [240] arrays")
        if not all(np.isfinite(value).all() for value in arrays):
            return np.full(48, np.nan, dtype="float64"), {
                "source_observed": False, "ohlc_repair_count": 0, "repairs": []
            }
        open_, high, low, close, volume, amount = arrays
        if (volume < 0).any() or (amount < 0).any():
            raise ValueError("intraday market volume/amount must be non-negative")
        high_fixed = np.maximum.reduce([high, open_, close])
        low_fixed = np.minimum.reduce([low, open_, close])
        high_changed = high_fixed != high
        low_changed = low_fixed != low
        repair_positions = np.flatnonzero(high_changed | low_changed)
        repairs = [{
            "bar_index": int(position),
            "high_original": float(high[position]),
            "high_fixed": float(high_fixed[position]),
            "low_original": float(low[position]),
            "low_fixed": float(low_fixed[position]),
        } for position in repair_positions]

        returns = np.empty(240, dtype="float64")
        returns[0] = np.log(close[0] / open_[0])
        returns[1:] = np.log(close[1:] / close[:-1])
        ret_oc = float(np.log(close[-1] / open_[0]))
        morning = _segment_return(open_, close, 0, 120)
        afternoon = _segment_return(open_, close, 120, 240)
        return_features = [
            ret_oc,
            _segment_return(open_, close, 0, 15),
            _segment_return(open_, close, 0, 30),
            _segment_return(open_, close, 0, 60),
            morning, afternoon,
            _segment_return(open_, close, 180, 240),
            _segment_return(open_, close, 210, 240),
            _segment_return(open_, close, 225, 240),
            afternoon - morning,
        ]
        squared = returns * returns
        minute_ranges = np.log(high_fixed / low_fixed)
        downside_sum = float(squared[returns < 0].sum())
        volatility = [
            float(np.sqrt(squared.sum())),
            _nonoverlap_rv(open_, close, 5),
            _nonoverlap_rv(open_, close, 15),
            float(np.log(high_fixed.max() / low_fixed.min())),
            float(minute_ranges.mean()),
            float(np.quantile(minute_ranges, 0.90)),
            float(np.max(np.abs(returns))),
            float(np.mean(squared * (returns > 0))),
            float(np.mean(squared * (returns < 0))),
            downside_sum / (float(squared.sum()) + self.epsilon),
        ]
        nonzero = np.sign(returns[returns != 0.0])
        sign_change = (
            float(np.mean(nonzero[1:] != nonzero[:-1])) if len(nonzero) > 1 else 0.0
        )
        running_max = np.maximum.accumulate(close)
        running_min = np.minimum.accumulate(close)
        day_high, day_low = float(high_fixed.max()), float(low_fixed.min())
        path = [
            abs(ret_oc) / (float(np.abs(returns).sum()) + self.epsilon),
            float(np.mean(returns > 0)), sign_change,
            _corr(returns, returns, lag=1), _corr(returns, returns, lag=5),
            float(np.max(np.log(running_max / close))),
            float(np.max(np.log(close / running_min))),
            (float(close[-1]) - day_low) / (day_high - day_low + self.epsilon),
            float(np.argmax(high_fixed)) / 239.0,
            float(np.argmin(low_fixed)) / 239.0,
        ]
        mean = float(returns.mean())
        centered = returns - mean
        std = float(np.std(returns, ddof=0))
        if std <= self.epsilon:
            return_skew = 0.0
            return_excess_kurtosis = 0.0
        else:
            standardized = centered / std
            return_skew = float(np.mean(standardized ** 3))
            return_excess_kurtosis = float(
                np.mean(standardized ** 4) - 3.0
            )
        median = float(np.median(returns))
        mad = float(np.median(np.abs(returns - median)))
        distribution = [
            return_skew, return_excess_kurtosis,
            float(np.quantile(returns, 0.05)), float(np.quantile(returns, 0.95)),
            float(np.quantile(returns, 0.75) - np.quantile(returns, 0.25)),
            mad,
            0.0 if mad <= self.epsilon else float(np.mean(np.abs(returns - median) > 3.0 * mad)),
            float(np.mean(np.abs(returns))),
        ]
        volume_total = float(volume.sum())
        amount_total = float(amount.sum())
        volume_mean = float(volume.mean())
        volume_share = volume / (volume_total + self.epsilon)
        amount_share = amount / (amount_total + self.epsilon)
        time = np.linspace(0.0, 1.0, 240)
        time_centered = time - time.mean()
        normalized_volume = volume / (volume_mean + self.epsilon)
        slope = float(
            np.sum(time_centered * (normalized_volume - normalized_volume.mean()))
            / np.sum(time_centered * time_centered)
        )
        trading = [
            float(amount[:30].sum()) / (amount_total + self.epsilon),
            float(amount[-30:].sum()) / (amount_total + self.epsilon),
            float(volume[:30].sum()) / (volume_total + self.epsilon),
            float(volume[-30:].sum()) / (volume_total + self.epsilon),
            float(volume[120:].sum()) / (float(volume[:120].sum()) + self.epsilon),
            float(np.std(volume, ddof=0)) / (volume_mean + self.epsilon),
            float(np.sum(volume_share * volume_share)),
            float(np.sum(amount_share * amount_share)),
            _corr(np.abs(returns), volume), slope,
        ]
        output = np.asarray(return_features + volatility + path + distribution + trading,
                            dtype="float64")
        if output.shape != (48,) or np.isinf(output).any():
            raise ValueError("intraday market feature output contract changed")
        return output, {
            "source_observed": True,
            "ohlc_repair_count": int(len(repair_positions)),
            "repairs": repairs,
        }

    def compute_day(self, fields: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        required = ("open", "high", "low", "close", "volume", "amt")
        if any(name not in fields for name in required):
            raise ValueError("intraday market six-field panel is incomplete")
        if any(np.asarray(fields[name]).shape != (4, 240) for name in required):
            raise ValueError("intraday market day panel must be [4,240] per field")
        blocks = []
        diagnostics = []
        for position, universe in enumerate(UNIVERSES):
            values, diagnostic = self.compute_index_day(
                fields["open"][position], fields["high"][position],
                fields["low"][position], fields["close"][position],
                fields["volume"][position], fields["amt"][position],
            )
            blocks.append(values)
            diagnostics.append({"universe": universe, **diagnostic})
        output = np.concatenate(blocks)
        return output, {
            "source_observed": all(row["source_observed"] for row in diagnostics),
            "ohlc_repair_count": sum(row["ohlc_repair_count"] for row in diagnostics),
            "indices": diagnostics,
        }
