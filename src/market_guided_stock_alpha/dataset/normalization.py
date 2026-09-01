"""Historical train-only robust features and daily label normalization."""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import (
    normalize_date_values,
    validate_membership,
)
from market_guided_stock_alpha.dataset.alignment import ArtifactLookup
from market_guided_stock_alpha.dataset.layout import (
    FeatureLayout,
    feature_names,
)
from market_guided_stock_alpha.derived.artifact import FeatureArtifact


@dataclass(frozen=True)
class FeatureStatistics:
    feature_names: Tuple[str, ...]
    median: np.ndarray
    mad: np.ndarray
    scale: np.ndarray
    finite_count: np.ndarray
    zero_mad_mask: np.ndarray
    epsilon: float = 1e-12
    mad_multiplier: float = 1.4826
    clip_min: float = -3.0
    clip_max: float = 3.0

    def validate(self) -> None:
        count = len(self.feature_names)
        arrays = (
            self.median,
            self.mad,
            self.scale,
            self.finite_count,
            self.zero_mad_mask,
        )
        if not count or any(value.shape != (count,) for value in arrays):
            raise ValueError("feature statistics shape does not match feature names")
        if self.median.dtype != np.float64 or self.mad.dtype != np.float64:
            raise ValueError("feature median/MAD must be float64")
        if self.scale.dtype != np.float64 or self.finite_count.dtype != np.int64:
            raise ValueError("feature scale/count dtype is invalid")
        if self.zero_mad_mask.dtype != np.bool_:
            raise ValueError("zero-MAD mask must be bool")
        if (self.finite_count <= 0).any():
            raise ValueError("a feature has no finite training observation")
        if not np.isfinite(self.median).all() or not np.isfinite(self.mad).all():
            raise ValueError("feature median/MAD contains non-finite values")
        if not np.isfinite(self.scale).all() or not (self.scale > 0).all():
            raise ValueError("feature scale must be finite and positive")

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        raw = np.asarray(values, dtype="float64")
        if raw.shape[-1] != len(self.feature_names):
            raise ValueError("feature transform order or dimension changed")
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            output = (raw - self.median) / self.scale
        output = np.clip(output, self.clip_min, self.clip_max)
        output[~np.isfinite(raw)] = 0.0
        output[~np.isfinite(output)] = 0.0
        return output.astype("float32", copy=False)


def fit_robust_statistics(
    values: np.ndarray,
    feature_names: Sequence[str],
    epsilon: float = 1e-12,
    mad_multiplier: float = 1.4826,
) -> FeatureStatistics:
    raw = np.asarray(values, dtype="float64")
    names = tuple(feature_names)
    if raw.ndim != 2 or raw.shape[1] != len(names) or not names:
        raise ValueError("robust fit requires [N,F] values and matching names")
    clean = raw.copy()
    finite = np.isfinite(clean)
    clean[~finite] = np.nan
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(clean, axis=0).astype("float64")
        mad = np.nanmedian(np.abs(clean - median), axis=0).astype("float64")
    statistics = FeatureStatistics(
        feature_names=names,
        median=median,
        mad=mad,
        scale=((mad + epsilon) * mad_multiplier).astype("float64"),
        finite_count=finite.sum(axis=0).astype("int64"),
        zero_mad_mask=(mad == 0.0).astype("bool"),
        epsilon=float(epsilon),
        mad_multiplier=float(mad_multiplier),
    )
    statistics.validate()
    return statistics


@dataclass(frozen=True)
class DatasetStatistics:
    by_family: Mapping[str, FeatureStatistics]

    def require(self, family: str, names: Sequence[str]) -> FeatureStatistics:
        if family not in self.by_family:
            raise ValueError("dataset statistics missing {}".format(family))
        statistics = self.by_family[family]
        if tuple(statistics.feature_names) != tuple(names):
            raise ValueError("{} statistics feature order changed".format(family))
        statistics.validate()
        return statistics


def normalize_cross_section_labels(
    frame: pd.DataFrame,
    lower_quantile: float = 0.025,
    upper_quantile: float = 0.975,
    minimum_finite_labels: int = 2,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """Clip then ddof=0 z-score each date, preserving missing-label masks."""
    required = {"trade_date", "sec_code", "label_raw", "label_mask"}
    if not required.issubset(frame.columns):
        raise ValueError("label normalization input is incomplete")
    output = frame.copy()
    output["label"] = np.nan
    output["label_mask"] = output["label_mask"].astype("bool")
    raw_values = pd.to_numeric(output["label_raw"], errors="coerce").to_numpy(
        dtype="float64"
    )
    if not np.array_equal(output["label_mask"].to_numpy(), np.isfinite(raw_values)):
        raise ValueError("label mask must equal finite raw-label availability")
    for _day, group in output.groupby("trade_date", sort=True):
        raw = group["label_raw"].to_numpy(dtype="float64")
        mask = group["label_mask"].to_numpy(dtype="bool")
        values = raw[mask]
        if len(values) < minimum_finite_labels:
            output.loc[group.index, "label_mask"] = False
            continue
        try:
            low, high = np.quantile(
                values, [lower_quantile, upper_quantile], method="linear"
            )
        except TypeError:
            low, high = np.quantile(
                values, [lower_quantile, upper_quantile], interpolation="linear"
            )
        clipped = np.clip(values, low, high)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0))
        if not np.isfinite(std) or std <= epsilon:
            output.loc[group.index, "label_mask"] = False
            continue
        output.loc[group.index[mask], "label"] = (clipped - mean) / std
    processed = output["label"].to_numpy(dtype="float64")
    output["label_mask"] &= np.isfinite(processed)
    return output


def _cross_section_fit_values(
    artifact: FeatureArtifact,
    membership: pd.DataFrame,
    fit_dates: pd.DatetimeIndex,
    allow_missing: bool,
) -> np.ndarray:
    lookup = ArtifactLookup(artifact)
    parts = []
    selected = membership.loc[membership["trade_date"].isin(fit_dates)]
    for day, group in selected.groupby("trade_date", sort=True):
        values, observed = lookup.cross_section_window(
            [day], tuple(group["sec_code"]), allow_missing=allow_missing
        )
        usable = observed[:, 0]
        if usable.any():
            parts.append(values[:, 0, :][usable])
    if not parts:
        raise ValueError("{} has no observed training rows".format(artifact.family))
    return np.concatenate(parts, axis=0)


def _market_fit_values(
    artifact: FeatureArtifact,
    fit_dates: pd.DatetimeIndex,
    require_all_observed: bool,
) -> np.ndarray:
    values, observed = ArtifactLookup(artifact).market_window(fit_dates)
    if require_all_observed and not observed.all():
        raise ValueError("{} is unobserved on a fit date".format(artifact.family))
    values = values[observed]
    if not len(values):
        raise ValueError("{} has no observed fit date".format(artifact.family))
    return values


def fit_dataset_statistics(
    layout: FeatureLayout,
    alpha158: FeatureArtifact,
    market: FeatureArtifact,
    membership: pd.DataFrame,
    train_dates: Sequence[object],
    intraday_price_volume: Optional[FeatureArtifact] = None,
    intraday_market: Optional[FeatureArtifact] = None,
) -> DatasetStatistics:
    """Fit historical train-only statistics without coupling them to Dataset init."""
    members = validate_membership(membership)
    dates = pd.DatetimeIndex(
        normalize_date_values(train_dates, "statistics fit dates")
    ).drop_duplicates().sort_values()
    if dates.empty:
        raise ValueError("statistics fit dates are empty")
    statistics = {
        "alpha158": fit_robust_statistics(
            _cross_section_fit_values(alpha158, members, dates, False),
            feature_names("alpha158"),
        ),
        layout.market_family: fit_robust_statistics(
            _market_fit_values(market, dates, True),
            feature_names(layout.market_family),
        ),
    }
    if "intraday_price_volume" in layout.stock_families:
        if intraday_price_volume is None:
            raise ValueError("Stage-2 statistics require Intraday Price-Volume Features")
        statistics["intraday_price_volume"] = fit_robust_statistics(
            _cross_section_fit_values(intraday_price_volume, members, dates, True),
            feature_names("intraday_price_volume"),
        )
    if layout.intraday_market_family:
        if intraday_market is None:
            raise ValueError("dual-frequency statistics require intraday market features")
        statistics["intraday_market"] = fit_robust_statistics(
            _market_fit_values(intraday_market, dates, False),
            feature_names("intraday_market"),
        )
    return DatasetStatistics(statistics)
