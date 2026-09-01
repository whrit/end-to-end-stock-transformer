"""Explicit date/security alignment over canonical mmap feature artifacts."""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from market_guided_stock_alpha.data.contracts import normalize_security_code
from market_guided_stock_alpha.derived.artifact import FeatureArtifact


class AlignmentError(ValueError):
    """Raised when a required canonical key is absent or duplicated."""


class ArtifactLookup:
    """Small key maps that retain the artifact and its mmap arrays in place."""

    def __init__(self, artifact: FeatureArtifact) -> None:
        self.artifact = artifact
        self.date_positions = {
            pd.Timestamp(day).normalize(): position
            for position, day in enumerate(artifact.dates)
        }
        self.security_positions = {
            code: position for position, code in enumerate(artifact.security_codes)
        }
        if len(self.date_positions) != len(artifact.dates):
            raise AlignmentError("artifact contains duplicate dates")
        if len(self.security_positions) != len(artifact.security_codes):
            raise AlignmentError("artifact contains duplicate securities")

    @staticmethod
    def _dates(values: Sequence[object]) -> Tuple[pd.Timestamp, ...]:
        dates = tuple(pd.Timestamp(value).tz_localize(None).normalize() for value in values)
        if len(dates) != len(set(dates)):
            raise AlignmentError("requested dates contain duplicates")
        return dates

    @staticmethod
    def _codes(values: Sequence[object]) -> Tuple[str, ...]:
        codes = tuple(normalize_security_code(value) for value in values)
        if len(codes) != len(set(codes)):
            raise AlignmentError("requested securities contain duplicates")
        return codes

    def require_dates(self, values: Sequence[object]) -> np.ndarray:
        dates = self._dates(values)
        missing = [day for day in dates if day not in self.date_positions]
        if missing:
            raise AlignmentError(
                "{} artifact missing required dates: {}".format(
                    self.artifact.family,
                    ",".join(str(day.date()) for day in missing[:20]),
                )
            )
        return np.asarray([self.date_positions[day] for day in dates], dtype="int64")

    def require_securities(self, values: Sequence[object]) -> np.ndarray:
        codes = self._codes(values)
        missing = [code for code in codes if code not in self.security_positions]
        if missing:
            raise AlignmentError(
                "{} artifact missing required securities: {}".format(
                    self.artifact.family, ",".join(missing[:20])
                )
            )
        return np.asarray(
            [self.security_positions[code] for code in codes], dtype="int64"
        )

    def cross_section_window(
        self,
        dates: Sequence[object],
        securities: Sequence[object],
        allow_missing: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.artifact.values.ndim != 3:
            raise AlignmentError("cross-section slicing requires a [D,S,F] artifact")
        requested_dates = self._dates(dates)
        requested_codes = self._codes(securities)
        if not allow_missing:
            date_positions = self.require_dates(requested_dates)
            security_positions = self.require_securities(requested_codes)
            features = np.asarray(self.artifact.values)[
                np.ix_(date_positions, security_positions, np.arange(self.artifact.values.shape[2]))
            ]
            observed = np.asarray(self.artifact.source_observed)[
                np.ix_(date_positions, security_positions)
            ]
            return features.transpose(1, 0, 2), observed.transpose(1, 0)

        feature_count = self.artifact.values.shape[2]
        features = np.full(
            (len(requested_codes), len(requested_dates), feature_count),
            np.nan,
            dtype=self.artifact.values.dtype,
        )
        observed = np.zeros(
            (len(requested_codes), len(requested_dates)), dtype="bool"
        )
        date_pairs = [
            (output, self.date_positions[day])
            for output, day in enumerate(requested_dates)
            if day in self.date_positions
        ]
        security_pairs = [
            (output, self.security_positions[code])
            for output, code in enumerate(requested_codes)
            if code in self.security_positions
        ]
        if date_pairs and security_pairs:
            out_dates, in_dates = map(
                lambda value: np.asarray(value, dtype="int64"), zip(*date_pairs)
            )
            out_codes, in_codes = map(
                lambda value: np.asarray(value, dtype="int64"), zip(*security_pairs)
            )
            selected = np.asarray(self.artifact.values)[
                np.ix_(in_dates, in_codes, np.arange(feature_count))
            ].transpose(1, 0, 2)
            selected_observed = np.asarray(self.artifact.source_observed)[
                np.ix_(in_dates, in_codes)
            ].transpose(1, 0)
            features[np.ix_(out_codes, out_dates, np.arange(feature_count))] = selected
            observed[np.ix_(out_codes, out_dates)] = selected_observed
        return features, observed

    def market_window(
        self, dates: Sequence[object]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.artifact.values.ndim != 2:
            raise AlignmentError("market slicing requires a [D,F] artifact")
        positions = self.require_dates(dates)
        return (
            np.asarray(self.artifact.values)[positions, :],
            np.asarray(self.artifact.source_observed)[positions],
        )
