"""Focused tests for strict 120+120 minute-session normalization."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.data import DataQualityError
from market_guided_stock_alpha.data.io import (
    read_wide_minute_pickles,
    wide_minute_frames_to_long,
)
from market_guided_stock_alpha.data.minute_session import (
    BARS_PER_DAY,
    EXPECTED_CLOCK_TIMES,
    normalize_minute_session,
)


def minute_stock():
    timestamps = pd.to_datetime(
        ["2024-01-02 " + clock for clock in EXPECTED_CLOCK_TIMES]
    )
    position = np.arange(BARS_PER_DAY, dtype="float64")
    close = 10.0 + position * 0.001
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "sec_code": "000001.SZ",
            "open": close - 0.001,
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
            "volume": 1000.0 + position,
            "amt": (1000.0 + position) * close,
        }
    )


class MinuteSessionTest(unittest.TestCase):
    def test_exact_240_bars_and_lunch_split(self):
        session = normalize_minute_session(minute_stock())
        self.assertEqual(len(session.timestamps), 240)
        self.assertEqual(session.timestamps[119].strftime("%H:%M"), "11:30")
        self.assertEqual(session.timestamps[120].strftime("%H:%M"), "13:01")
        self.assertFalse(any(value.hour == 12 for value in session.timestamps))
        self.assertEqual(session.fields["close"].shape, (240, 1))
        np.testing.assert_allclose(session.bar_coverage, 1.0)

    def test_missing_bar_is_nan_without_interpolation(self):
        source = minute_stock().drop(index=50)
        session = normalize_minute_session(source)
        self.assertTrue(np.isnan(session.fields["close"][50, 0]))
        self.assertAlmostEqual(session.bar_coverage[0], 239.0 / 240.0)
        self.assertAlmostEqual(session.fields["close"][49, 0], 10.049)
        self.assertAlmostEqual(session.fields["close"][51, 0], 10.051)

    def test_exact_90_percent_coverage_is_eligible(self):
        session = normalize_minute_session(minute_stock().iloc[24:].copy())
        self.assertAlmostEqual(session.bar_coverage[0], 0.90)
        self.assertTrue(bool(session.eligible[0]))
        self.assertTrue(np.isnan(session.fields["close"][:24, 0]).all())

    def test_duplicate_timestamp_key_is_rejected(self):
        duplicate = pd.concat([minute_stock(), minute_stock().iloc[[0]]])
        with self.assertRaisesRegex(DataQualityError, "duplicate"):
            normalize_minute_session(duplicate)

    def test_wide_frames_converge_to_same_long_contract(self):
        source = minute_stock().set_index("timestamp")
        frames = {
            field: source.pivot(columns="sec_code", values=field)
            for field in ("open", "high", "low", "close", "volume", "amt")
        }
        long = wide_minute_frames_to_long(frames)
        session = normalize_minute_session(long)
        self.assertEqual(long.shape, (240, 8))
        np.testing.assert_allclose(session.bar_coverage, 1.0)
        np.testing.assert_allclose(
            session.fields["close"][:, 0], source["close"].to_numpy()
        )

    def test_six_explicit_wide_pickles_are_a_thin_supported_input(self):
        source = minute_stock().set_index("timestamp")
        frames = {
            field: source.pivot(columns="sec_code", values=field)
            for field in ("open", "high", "low", "close", "volume", "amt")
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for field, frame in frames.items():
                path = Path(directory) / "{}.pkl".format(field)
                frame.to_pickle(path)
                paths[field] = path
            loaded = read_wide_minute_pickles(paths)
            long = wide_minute_frames_to_long(loaded)
            self.assertEqual(long.shape, (240, 8))
            self.assertEqual(long["sec_code"].unique().tolist(), ["000001.SZ"])


if __name__ == "__main__":
    unittest.main()
