"""Fixed-spec raw Alpha158 generator for canonical adjusted stock market data."""

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    normalize_date_values,
    normalize_security_code,
)
from market_guided_stock_alpha.features.rolling_ops import (
    rolling_corr,
    rolling_idxmax,
    rolling_idxmin,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_ols,
    rolling_quantile,
    rolling_rank_last,
    rolling_std,
    rolling_sum,
    sanitize_non_finite,
)


KLINE_FEATURES = (
    "KMID",
    "KLEN",
    "KMID2",
    "KUP",
    "KUP2",
    "KLOW",
    "KLOW2",
    "KSFT",
    "KSFT2",
)
PRICE_FEATURES = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
ROLLING_FAMILIES = (
    "ROC",
    "MA",
    "STD",
    "BETA",
    "RSQR",
    "RESI",
    "MAX",
    "MIN",
    "QTLU",
    "QTLD",
    "RANK",
    "RSV",
    "IMAX",
    "IMIN",
    "IMXD",
    "CORR",
    "CORD",
    "CNTP",
    "CNTN",
    "CNTD",
    "SUMP",
    "SUMN",
    "SUMD",
    "VMA",
    "VSTD",
    "WVMA",
    "VSUMP",
    "VSUMN",
    "VSUMD",
)
REQUIRED_MARKET_COLUMNS = (
    "trade_date",
    "sec_code",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
)
class Alpha158SpecError(ValueError):
    pass


class Alpha158DataError(DataQualityError):
    """Raised when canonical Alpha158 input rows violate the input contract."""


def _normalize_stock_code(value: Any, context: str) -> str:
    try:
        return normalize_security_code(value, context)
    except DataQualityError as exc:
        raise Alpha158DataError(str(exc)) from exc


def _normalize_trade_dates(values: Any, context: str) -> pd.Series:
    try:
        return normalize_date_values(values, "{} trade_date".format(context))
    except DataQualityError as exc:
        raise Alpha158DataError("{} has invalid trade_date".format(context)) from exc


def validate_alpha158_spec(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping):
        raise Alpha158SpecError("Alpha158 spec must be an object")
    if spec.get("feature_count") != 158:
        raise Alpha158SpecError("Alpha158 feature_count must equal 158")
    if spec.get("rolling_windows") != [5, 10, 20, 30, 60]:
        raise Alpha158SpecError(
            "Alpha158 rolling_windows must equal 5,10,20,30,60"
        )
    if spec.get("rolling_min_periods") != 1 or spec.get("rolling_std_ddof") != 1:
        raise Alpha158SpecError(
            "Alpha158 rolling min_periods and std ddof must equal 1"
        )
    if spec.get("feature_group_order") != ["kbar", "price", "rolling"]:
        raise Alpha158SpecError("Alpha158 feature group order is invalid")
    if tuple(spec.get("kbar_features", [])) != KLINE_FEATURES:
        raise Alpha158SpecError("Alpha158 kbar order is invalid")
    if tuple(spec.get("price_features", [])) != PRICE_FEATURES:
        raise Alpha158SpecError("Alpha158 price order is invalid")
    if tuple(spec.get("rolling_family_order", [])) != ROLLING_FAMILIES:
        raise Alpha158SpecError("Alpha158 rolling family order is invalid")
    source = spec.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("repository") != "microsoft/qlib"
        or source.get("file") != "qlib/contrib/data/loader.py"
        or source.get("class") != "Alpha158DL"
        or source.get("blob_sha")
        != "4d11f3a34cc1d12728723956c8e9324556eee8e3"
    ):
        raise Alpha158SpecError(
            "Alpha158 source blob SHA is not the fixed project version"
        )
    epsilon = spec.get("epsilon")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or epsilon <= 0
    ):
        raise Alpha158SpecError("Alpha158 epsilon must be positive")
    if spec.get("sanitize_non_finite_to_nan") is not True:
        raise Alpha158SpecError("Alpha158 must sanitize non-finite output to NaN")


class Alpha158Generator:
    """Generate the fixed 9 + 4 + 29*5 raw feature ordering."""

    def __init__(self, spec: Mapping[str, Any]) -> None:
        validate_alpha158_spec(spec)
        self.spec = dict(spec)
        self.windows = list(spec["rolling_windows"])
        self.epsilon = float(spec["epsilon"])
        self.last_non_finite_to_nan_count = 0
        self._manifest = self._build_manifest()

    def _build_manifest(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        kbar_expressions = {
            "KMID": "(close-open)/open",
            "KLEN": "(high-low)/open",
            "KMID2": "(close-open)/(high-low+epsilon)",
            "KUP": "(high-max(open,close))/open",
            "KUP2": "(high-max(open,close))/(high-low+epsilon)",
            "KLOW": "(min(open,close)-low)/open",
            "KLOW2": "(min(open,close)-low)/(high-low+epsilon)",
            "KSFT": "(2*close-high-low)/open",
            "KSFT2": "(2*close-high-low)/(high-low+epsilon)",
        }
        for name in KLINE_FEATURES:
            rows.append(
                {
                    "feature_name": name,
                    "family": "kbar",
                    "window": None,
                    "expression": kbar_expressions[name],
                    "source": "canonical OHLC",
                }
            )
        for name, field in zip(
            PRICE_FEATURES, ("open", "high", "low", "vwap")
        ):
            rows.append(
                {
                    "feature_name": name,
                    "family": "price",
                    "window": 0,
                    "expression": "{}/close".format(field),
                    "source": "canonical OHLCV",
                }
            )
        expressions = {
            "ROC": "Ref(close,d)/close",
            "MA": "Mean(close,d)/close",
            "STD": "Std(close,d)/close",
            "BETA": "Slope(close,d)/close",
            "RSQR": "Rsquare(close,d)",
            "RESI": "Residual(close,d)/close",
            "MAX": "Max(high,d)/close",
            "MIN": "Min(low,d)/close",
            "QTLU": "Quantile(close,d,0.8)/close",
            "QTLD": "Quantile(close,d,0.2)/close",
            "RANK": "RankPctLast(close,d)",
            "RSV": "(close-Min(low,d))/(Max(high,d)-Min(low,d)+epsilon)",
            "IMAX": "IdxMax(high,d)/d",
            "IMIN": "IdxMin(low,d)/d",
            "IMXD": "(IdxMax(high,d)-IdxMin(low,d))/d",
            "CORR": "Corr(close,log(volume+1),d)",
            "CORD": "Corr(close/Ref(close,1),log(volume/Ref(volume,1)+1),d)",
            "CNTP": "Mean(close>Ref(close,1),d)",
            "CNTN": "Mean(close<Ref(close,1),d)",
            "CNTD": "CNTPd-CNTNd",
            "SUMP": "Sum(max(delta_close,0),d)/(Sum(abs(delta_close),d)+epsilon)",
            "SUMN": "Sum(max(-delta_close,0),d)/(Sum(abs(delta_close),d)+epsilon)",
            "SUMD": "(positive_sum-negative_sum)/(absolute_sum+epsilon)",
            "VMA": "Mean(volume,d)/(volume+epsilon)",
            "VSTD": "Std(volume,d)/(volume+epsilon)",
            "WVMA": "Std(weighted_change,d)/(Mean(weighted_change,d)+epsilon)",
            "VSUMP": "Sum(max(delta_volume,0),d)/(Sum(abs(delta_volume),d)+epsilon)",
            "VSUMN": "Sum(max(-delta_volume,0),d)/(Sum(abs(delta_volume),d)+epsilon)",
            "VSUMD": "(positive_volume_sum-negative_volume_sum)/(absolute_volume_sum+epsilon)",
        }
        for family in ROLLING_FAMILIES:
            for window in self.windows:
                rows.append(
                    {
                        "feature_name": "{}{}".format(family, window),
                        "family": family,
                        "window": window,
                        "expression": expressions[family],
                        "source": "canonical close/high/low/volume",
                    }
                )
        manifest = pd.DataFrame(rows)
        manifest.insert(0, "feature_index", range(len(manifest)))
        if len(manifest) != 158 or manifest["feature_name"].duplicated().any():
            raise Alpha158SpecError(
                "Generated Alpha158 manifest is not 158 unique features"
            )
        return manifest

    def get_feature_names(self) -> List[str]:
        return self._manifest["feature_name"].tolist()

    def get_feature_manifest(self) -> pd.DataFrame:
        return self._manifest.copy(deep=True)

    def _clean(self, series: pd.Series) -> pd.Series:
        inherited_count = int(series.attrs.get("non_finite_to_nan_count", 0))
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        count = int((numeric.notna() & ~np.isfinite(numeric.to_numpy())).sum())
        self.last_non_finite_to_nan_count += inherited_count + count
        return sanitize_non_finite(numeric)

    def _divide(
        self, numerator: pd.Series, denominator: pd.Series
    ) -> pd.Series:
        clean_numerator = self._clean(numerator)
        clean_denominator = self._clean(denominator)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return self._clean(clean_numerator / clean_denominator)

    @staticmethod
    def _signed_parts(delta: pd.Series) -> Sequence[pd.Series]:
        valid = delta.notna()
        positive = pd.Series(np.nan, index=delta.index, dtype="float64")
        negative = pd.Series(np.nan, index=delta.index, dtype="float64")
        positive.loc[valid] = np.maximum(delta.loc[valid], 0.0)
        negative.loc[valid] = np.maximum(-delta.loc[valid], 0.0)
        return positive, negative

    @staticmethod
    def _comparison(
        left: pd.Series, right: pd.Series, operator: str
    ) -> pd.Series:
        result = pd.Series(np.nan, index=left.index, dtype="float64")
        valid = left.notna() & right.notna()
        if operator == "greater":
            result.loc[valid] = left.loc[valid].gt(right.loc[valid]).astype(
                "float64"
            )
        else:
            result.loc[valid] = left.loc[valid].lt(right.loc[valid]).astype(
                "float64"
            )
        return result

    def _transform_security(self, group: pd.DataFrame) -> pd.DataFrame:
        result = group.loc[:, ["trade_date", "sec_code"]].copy()
        open_price, high, low, close = (
            group[name] for name in ("open", "high", "low", "close")
        )
        vwap, volume = group["vwap"], group["volume"]
        spread = high - low
        max_open_close = pd.Series(
            np.maximum(open_price.to_numpy(), close.to_numpy()), index=group.index
        )
        min_open_close = pd.Series(
            np.minimum(open_price.to_numpy(), close.to_numpy()), index=group.index
        )
        result["KMID"] = self._divide(close - open_price, open_price)
        result["KLEN"] = self._divide(spread, open_price)
        result["KMID2"] = self._divide(close - open_price, spread + self.epsilon)
        result["KUP"] = self._divide(high - max_open_close, open_price)
        result["KUP2"] = self._divide(
            high - max_open_close, spread + self.epsilon
        )
        result["KLOW"] = self._divide(min_open_close - low, open_price)
        result["KLOW2"] = self._divide(
            min_open_close - low, spread + self.epsilon
        )
        result["KSFT"] = self._divide(
            2.0 * close - high - low, open_price
        )
        result["KSFT2"] = self._divide(
            2.0 * close - high - low, spread + self.epsilon
        )
        for name, values in (
            ("OPEN0", open_price),
            ("HIGH0", high),
            ("LOW0", low),
            ("VWAP0", vwap),
        ):
            result[name] = self._divide(values, close)

        close_ref1 = close.shift(1)
        volume_ref1 = volume.shift(1)
        delta_close = close - close_ref1
        delta_volume = volume - volume_ref1
        positive_close, negative_close = self._signed_parts(delta_close)
        positive_volume, negative_volume = self._signed_parts(delta_volume)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_volume = self._clean(np.log(volume + 1.0))
            close_ratio = self._divide(close, close_ref1)
            volume_ratio_log = self._clean(
                np.log(self._divide(volume, volume_ref1) + 1.0)
            )
            weighted_change = self._clean((close_ratio - 1.0).abs() * volume)
        count_positive = self._comparison(close, close_ref1, "greater")
        count_negative = self._comparison(close, close_ref1, "less")

        family_values: Dict[str, Dict[int, pd.Series]] = {
            family: {} for family in ROLLING_FAMILIES
        }
        for window in self.windows:
            slope, rsquare, residual = rolling_ols(close, window)
            max_high = rolling_max(high, window)
            min_low = rolling_min(low, window)
            idxmax = rolling_idxmax(high, window)
            idxmin = rolling_idxmin(low, window)
            pos_close_sum = rolling_sum(positive_close, window)
            neg_close_sum = rolling_sum(negative_close, window)
            abs_close_sum = rolling_sum(delta_close.abs(), window)
            pos_volume_sum = rolling_sum(positive_volume, window)
            neg_volume_sum = rolling_sum(negative_volume, window)
            abs_volume_sum = rolling_sum(delta_volume.abs(), window)
            family_values["ROC"][window] = self._divide(
                close.shift(window), close
            )
            family_values["MA"][window] = self._divide(
                rolling_mean(close, window), close
            )
            family_values["STD"][window] = self._divide(
                rolling_std(close, window), close
            )
            family_values["BETA"][window] = self._divide(slope, close)
            family_values["RSQR"][window] = self._clean(rsquare)
            family_values["RESI"][window] = self._divide(residual, close)
            family_values["MAX"][window] = self._divide(max_high, close)
            family_values["MIN"][window] = self._divide(min_low, close)
            family_values["QTLU"][window] = self._divide(
                rolling_quantile(close, window, 0.8), close
            )
            family_values["QTLD"][window] = self._divide(
                rolling_quantile(close, window, 0.2), close
            )
            family_values["RANK"][window] = self._clean(
                rolling_rank_last(close, window)
            )
            family_values["RSV"][window] = self._divide(
                close - min_low, max_high - min_low + self.epsilon
            )
            family_values["IMAX"][window] = self._clean(
                idxmax / float(window)
            )
            family_values["IMIN"][window] = self._clean(
                idxmin / float(window)
            )
            family_values["IMXD"][window] = self._clean(
                (idxmax - idxmin) / float(window)
            )
            family_values["CORR"][window] = self._clean(
                rolling_corr(close, log_volume, window)
            )
            family_values["CORD"][window] = self._clean(
                rolling_corr(close_ratio, volume_ratio_log, window)
            )
            cntp = rolling_mean(count_positive, window)
            cntn = rolling_mean(count_negative, window)
            family_values["CNTP"][window] = self._clean(cntp)
            family_values["CNTN"][window] = self._clean(cntn)
            family_values["CNTD"][window] = self._clean(cntp - cntn)
            family_values["SUMP"][window] = self._divide(
                pos_close_sum, abs_close_sum + self.epsilon
            )
            family_values["SUMN"][window] = self._divide(
                neg_close_sum, abs_close_sum + self.epsilon
            )
            family_values["SUMD"][window] = self._divide(
                pos_close_sum - neg_close_sum, abs_close_sum + self.epsilon
            )
            family_values["VMA"][window] = self._divide(
                rolling_mean(volume, window), volume + self.epsilon
            )
            family_values["VSTD"][window] = self._divide(
                rolling_std(volume, window), volume + self.epsilon
            )
            family_values["WVMA"][window] = self._divide(
                rolling_std(weighted_change, window),
                rolling_mean(weighted_change, window) + self.epsilon,
            )
            family_values["VSUMP"][window] = self._divide(
                pos_volume_sum, abs_volume_sum + self.epsilon
            )
            family_values["VSUMN"][window] = self._divide(
                neg_volume_sum, abs_volume_sum + self.epsilon
            )
            family_values["VSUMD"][window] = self._divide(
                pos_volume_sum - neg_volume_sum,
                abs_volume_sum + self.epsilon,
            )
        rolling_columns = {
            "{}{}".format(family, window): family_values[family][window].to_numpy()
            for family in ROLLING_FAMILIES
            for window in self.windows
        }
        result = pd.concat(
            [result, pd.DataFrame(rolling_columns, index=result.index)], axis=1
        )
        return result.loc[
            :, ["trade_date", "sec_code"] + self.get_feature_names()
        ]

    def transform_market_history(
        self, market_history: pd.DataFrame
    ) -> pd.DataFrame:
        missing = [
            name
            for name in REQUIRED_MARKET_COLUMNS
            if name not in market_history.columns
        ]
        if missing:
            raise ValueError(
                "Alpha158 market history missing columns: {}".format(
                    ",".join(missing)
                )
            )
        frame = market_history.loc[:, list(REQUIRED_MARKET_COLUMNS)].copy(
            deep=True
        )
        frame["trade_date"] = _normalize_trade_dates(
            frame["trade_date"], "Alpha158 market history"
        ).to_numpy()
        frame["sec_code"] = [
            _normalize_stock_code(value, "Alpha158 sec_code")
            for value in frame["sec_code"]
        ]
        self.last_non_finite_to_nan_count = 0
        for field in REQUIRED_MARKET_COLUMNS[2:]:
            frame[field] = self._clean(frame[field])
        if frame.duplicated(["trade_date", "sec_code"]).any():
            raise Alpha158DataError("Alpha158 market history has duplicate keys")
        frame = frame.sort_values(
            ["sec_code", "trade_date"], kind="mergesort"
        ).reset_index(drop=True)
        parts = [
            self._transform_security(group.reset_index(drop=True))
            for _, group in frame.groupby("sec_code", sort=False)
        ]
        if not parts:
            return pd.DataFrame(
                columns=["trade_date", "sec_code"] + self.get_feature_names()
            )
        output = pd.concat(parts, ignore_index=True)
        return output.sort_values(
            ["trade_date", "sec_code"], kind="mergesort"
        ).reset_index(drop=True)

    def transform_for_target_grid(
        self, market_history: pd.DataFrame, target_grid: pd.DataFrame
    ) -> pd.DataFrame:
        if "trade_date" not in target_grid or "sec_code" not in target_grid:
            raise ValueError("Alpha158 target grid requires trade_date and sec_code")
        target = target_grid.copy(deep=True)
        target["trade_date"] = _normalize_trade_dates(
            target["trade_date"], "Alpha158 target grid"
        ).to_numpy()
        target["sec_code"] = [
            _normalize_stock_code(value, "Alpha158 target sec_code")
            for value in target["sec_code"]
        ]
        if target.duplicated(["trade_date", "sec_code"]).any():
            raise Alpha158DataError("Alpha158 target grid has duplicate keys")
        features = self.transform_market_history(market_history)
        return target.merge(
            features,
            on=["trade_date", "sec_code"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
