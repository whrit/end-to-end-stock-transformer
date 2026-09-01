"""Lazy neutral rolling-sequence dataset over canonical mmap artifacts."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.calendar import TradingCalendar
from market_guided_stock_alpha.data.contracts import (
    normalize_date_values,
    validate_membership,
)
from market_guided_stock_alpha.dataset.alignment import ArtifactLookup
from market_guided_stock_alpha.dataset.layout import (
    FeatureLayout,
    feature_names,
)
from market_guided_stock_alpha.dataset.normalization import (
    DatasetStatistics,
    normalize_cross_section_labels,
)
from market_guided_stock_alpha.derived.artifact import FeatureArtifact
from market_guided_stock_alpha.labels.derived_label import DerivedLabelReader
from market_guided_stock_alpha.labels.relative_excess_return import (
    DEFAULT_VALUE_COLUMN,
)


@dataclass(frozen=True)
class SequenceSample:
    prediction_date: pd.Timestamp
    history_dates: pd.DatetimeIndex
    security_ids: np.ndarray
    security_codes: Tuple[str, ...]
    stock_features: np.ndarray
    market_features: np.ndarray
    labels_raw: np.ndarray
    labels: np.ndarray
    label_mask: np.ndarray
    stock_mask: np.ndarray
    component_masks: Mapping[str, np.ndarray]
    market_mask: np.ndarray
    metadata: Mapping[str, Any]
    intraday_market_features: Optional[np.ndarray] = None
    intraday_market_mask: Optional[np.ndarray] = None


class SequenceDataset:
    """One item per prediction date, preserving its whole dynamic cross-section."""

    def __init__(
        self,
        alpha158: FeatureArtifact,
        market: FeatureArtifact,
        membership: pd.DataFrame,
        label_reader: DerivedLabelReader,
        calendar: TradingCalendar,
        statistics: DatasetStatistics,
        layout: Optional[FeatureLayout] = None,
        intraday_price_volume: Optional[FeatureArtifact] = None,
        intraday_market: Optional[FeatureArtifact] = None,
        prediction_dates: Optional[Sequence[Any]] = None,
        minimum_security_count: int = 2,
    ) -> None:
        self.layout = layout or FeatureLayout.stage1()
        if label_reader.column != DEFAULT_VALUE_COLUMN:
            raise ValueError(
                "canonical dataset requires label column {}".format(
                    DEFAULT_VALUE_COLUMN
                )
            )
        if type(minimum_security_count) is not int or minimum_security_count < 1:
            raise ValueError("minimum_security_count must be a positive integer")
        self.minimum_security_count = minimum_security_count
        self.calendar = calendar
        self.statistics = statistics
        self.artifacts = {
            "alpha158": alpha158,
            self.layout.market_family: market,
        }
        if intraday_price_volume is not None:
            self.artifacts["intraday_price_volume"] = intraday_price_volume
        if intraday_market is not None:
            self.artifacts["intraday_market"] = intraday_market
        self._validate_artifact_contracts(intraday_price_volume, intraday_market)
        self.lookups = {
            family: ArtifactLookup(artifact)
            for family, artifact in self.artifacts.items()
        }

        members = validate_membership(membership)
        if not members["trade_date"].isin(calendar.open_dates).all():
            raise ValueError("membership contains a non-trading date")
        members = members.sort_values(
            ["trade_date", "sec_code"], kind="mergesort"
        ).reset_index(drop=True)

        available_member_dates = pd.DatetimeIndex(
            members["trade_date"].drop_duplicates()
        )
        explicitly_requested = prediction_dates is not None
        if explicitly_requested:
            requested = pd.DatetimeIndex(
                normalize_date_values(prediction_dates, "prediction dates")
            ).drop_duplicates()
            missing_membership = requested.difference(available_member_dates)
            if len(missing_membership):
                raise ValueError(
                    "prediction dates missing membership: {}".format(
                        ",".join(str(day.date()) for day in missing_membership[:20])
                    )
                )
            members = members.loc[members["trade_date"].isin(requested)].copy()
            available_member_dates = requested.sort_values()

        eligible = []
        ineligible = []
        calendar_positions = {
            pd.Timestamp(day): position
            for position, day in enumerate(calendar.open_dates)
        }
        for day in available_member_dates:
            position = calendar_positions[pd.Timestamp(day)]
            if position < self.layout.lookback - 1:
                ineligible.append(pd.Timestamp(day))
            else:
                eligible.append(pd.Timestamp(day))
        self.anchor_dates = pd.DatetimeIndex(eligible)
        self.ineligible_anchor_dates = pd.DatetimeIndex(ineligible)
        if explicitly_requested and len(self.ineligible_anchor_dates):
            raise ValueError(
                "explicit prediction dates lack eight-day history: {}".format(
                    ",".join(
                        str(day.date()) for day in self.ineligible_anchor_dates[:20]
                    )
                )
            )
        if self.anchor_dates.empty:
            raise ValueError("no prediction date has a complete eight-day history")
        members = members.loc[members["trade_date"].isin(self.anchor_dates)].copy()
        counts = members.groupby("trade_date", sort=True).size()
        if not counts.ge(minimum_security_count).all():
            bad = counts.index[~counts.ge(minimum_security_count)]
            raise ValueError(
                "daily membership below minimum on {}".format(
                    ",".join(str(day.date()) for day in bad[:20])
                )
            )
        self.membership = members
        aligned_labels, self.label_metadata = label_reader.align(
            members.loc[:, ["trade_date", "sec_code"]]
        )
        self.labels = normalize_cross_section_labels(aligned_labels)
        self._member_groups = {
            pd.Timestamp(day): group.sort_values("sec_code", kind="mergesort")
            for day, group in self.labels.groupby("trade_date", sort=True)
        }
        ordered_registry = tuple(sorted(alpha158.security_codes))
        self.security_id_by_code = {
            code: position for position, code in enumerate(ordered_registry)
        }
        self._validate_required_alignment()

    def _validate_artifact_contracts(
        self,
        intraday_price_volume: Optional[FeatureArtifact],
        intraday_market: Optional[FeatureArtifact],
    ) -> None:
        required = {
            "alpha158": self.artifacts["alpha158"],
            self.layout.market_family: self.artifacts[self.layout.market_family],
        }
        if "intraday_price_volume" in self.layout.stock_families:
            if intraday_price_volume is None:
                raise ValueError("Stage-2 layout requires Intraday Price-Volume Features artifact")
            required["intraday_price_volume"] = intraday_price_volume
        elif intraday_price_volume is not None:
            raise ValueError("Stage-1 layout must not receive Intraday Price-Volume Features")
        if self.layout.intraday_market_family:
            if intraday_market is None:
                raise ValueError("dual-frequency layout requires intraday market features")
            required["intraday_market"] = intraday_market
        elif intraday_market is not None:
            raise ValueError("non-exploratory layout must not receive intraday market features")
        for family, artifact in required.items():
            if artifact.family != family:
                raise ValueError("expected {} artifact, got {}".format(family, artifact.family))
            if tuple(artifact.feature_names) != feature_names(family):
                raise ValueError("{} artifact feature order changed".format(family))
            self.statistics.require(family, artifact.feature_names)

    def history_dates(self, prediction_date: Any) -> pd.DatetimeIndex:
        day = pd.Timestamp(prediction_date).tz_localize(None).normalize()
        position = int(self.calendar.open_dates.get_indexer([day])[0])
        if position < self.layout.lookback - 1:
            raise ValueError("prediction date lacks an eight-day trading history")
        history = self.calendar.open_dates[
            position - (self.layout.lookback - 1) : position + 1
        ]
        if len(history) != self.layout.lookback or history[-1] != day:
            raise ValueError("eight-day trading-calendar history changed")
        return history

    def _validate_required_alignment(self) -> None:
        alpha = self.lookups["alpha158"]
        market = self.lookups[self.layout.market_family]
        for day in self.anchor_dates:
            history = self.history_dates(day)
            group = self._member_groups[pd.Timestamp(day)]
            codes = tuple(group["sec_code"])
            alpha.require_dates(history)
            alpha.require_securities(codes)
            market_positions = market.require_dates(history)
            observed = np.asarray(market.artifact.source_observed)[market_positions]
            if not observed.all():
                raise ValueError(
                    "{} source_observed is false in history ending {}".format(
                        self.layout.market_family, day.date()
                    )
                )

    def __len__(self) -> int:
        return len(self.anchor_dates)

    def __getitem__(self, position: int) -> SequenceSample:
        if type(position) is not int or position < 0 or position >= len(self):
            raise IndexError("dataset position is out of range")
        prediction_date = pd.Timestamp(self.anchor_dates[position])
        history = self.history_dates(prediction_date)
        group = self._member_groups[prediction_date]
        codes = tuple(group["sec_code"].tolist())
        ids = np.asarray(
            [self.security_id_by_code[code] for code in codes], dtype="int32"
        )

        alpha_raw, alpha_observed = self.lookups["alpha158"].cross_section_window(
            history, codes
        )
        alpha = self.statistics.require(
            "alpha158", feature_names("alpha158")
        ).transform(alpha_raw)
        stock_parts = [alpha]
        component_masks: Dict[str, np.ndarray] = {"alpha158": alpha_observed}
        if "intraday_price_volume" in self.layout.stock_families:
            intraday_raw, intraday_observed = self.lookups[
                "intraday_price_volume"
            ].cross_section_window(history, codes, allow_missing=True)
            intraday = self.statistics.require(
                "intraday_price_volume", feature_names("intraday_price_volume")
            ).transform(intraday_raw)
            stock_parts.append(intraday)
            component_masks["intraday_price_volume"] = intraday_observed
        stock = np.concatenate(stock_parts, axis=2).astype("float32", copy=False)

        market_raw, market_observed = self.lookups[
            self.layout.market_family
        ].market_window(history)
        market = self.statistics.require(
            self.layout.market_family,
            feature_names(self.layout.market_family),
        ).transform(market_raw)

        exploratory = None
        exploratory_mask = None
        if self.layout.intraday_market_family:
            exploratory_raw, exploratory_mask = self.lookups[
                "intraday_market"
            ].market_window(history)
            exploratory = self.statistics.require(
                "intraday_market", feature_names("intraday_market")
            ).transform(exploratory_raw)

        labels_raw = group["label_raw"].to_numpy(dtype="float64")
        labels = group["label"].to_numpy(dtype="float32")
        label_mask = group["label_mask"].to_numpy(dtype="bool")
        return SequenceSample(
            prediction_date=prediction_date,
            history_dates=history,
            security_ids=ids,
            security_codes=codes,
            stock_features=stock,
            market_features=market,
            labels_raw=labels_raw,
            labels=labels,
            label_mask=label_mask,
            stock_mask=alpha_observed,
            component_masks=component_masks,
            market_mask=market_observed,
            metadata={
                "layout_position": self.layout.position,
                "stock_families": self.layout.stock_families,
                "market_family": self.layout.market_family,
                "market_physically_repeated": False,
            },
            intraday_market_features=exploratory,
            intraday_market_mask=exploratory_mask,
        )


def last_day_tabular_view(sample: SequenceSample) -> Dict[str, Any]:
    """Thin baseline view; date-level market data is broadcast only here."""
    security_count = len(sample.security_codes)
    market = np.broadcast_to(
        sample.market_features[-1][None, :],
        (security_count, sample.market_features.shape[1]),
    )
    return {
        "features": np.concatenate(
            [sample.stock_features[:, -1, :], market], axis=1
        ).astype("float32", copy=False),
        "labels": sample.labels,
        "label_mask": sample.label_mask,
        "security_codes": sample.security_codes,
        "prediction_date": sample.prediction_date,
    }
