"""Small synthetic build/artifact/read round trips for every feature family."""

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

from market_guided_stock_alpha.data.minute_session import EXPECTED_CLOCK_TIMES
from market_guided_stock_alpha.derived import (
    build_alpha158_artifact,
    build_intraday_market_artifact,
    build_intraday_price_volume_artifact,
    build_market63_artifact,
    build_market84_artifact,
    read_alpha158_artifact,
    read_intraday_market_artifact,
    read_intraday_price_volume_artifact,
    read_market63_artifact,
    read_market84_artifact,
)
from market_guided_stock_alpha.features.market63 import MARKET_UNIVERSES
from market_guided_stock_alpha.features.market84 import (
    MARKET84_INDEX_CODES,
    MARKET84_UNIVERSES,
)


def daily_stock(periods=65):
    offset = np.arange(periods, dtype="float64")
    close = 20.0 + offset * 0.1
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-02", periods=periods),
            "sec_code": "000001.SZ",
            "open": close - 0.05,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "vwap": close - 0.01,
            "volume": 1000.0 + offset,
        }
    )


def daily_index(universes, codes, periods=65):
    rows = []
    for block, (universe, code) in enumerate(zip(universes, codes), start=1):
        for offset, day in enumerate(pd.bdate_range("2024-01-02", periods=periods)):
            close = 100.0 * block + offset
            rows.append(
                {
                    "trade_date": day,
                    "index_code": code,
                    "open": close - 0.5,
                    "close": close,
                    "amount": 1e9 * block + offset * 1e6,
                }
            )
    return pd.DataFrame(rows)


def minute_panel(codes, code_column):
    timestamps = pd.to_datetime(
        ["2024-01-02 " + clock for clock in EXPECTED_CLOCK_TIMES]
    )
    rows = []
    for block, code in enumerate(codes):
        position = np.arange(240, dtype="float64")
        returns = 0.0002 * np.sin(2.0 * np.pi * position / (20.0 + block))
        close = (100.0 + block * 20.0) * np.exp(np.cumsum(returns))
        opening = np.r_[100.0 + block * 20.0, close[:-1]]
        volume = 1000.0 + block * 100.0 + position
        for index, timestamp in enumerate(timestamps):
            rows.append(
                {
                    "timestamp": timestamp,
                    code_column: code,
                    "open": opening[index],
                    "high": max(opening[index], close[index]) * 1.001,
                    "low": min(opening[index], close[index]) * 0.999,
                    "close": close[index],
                    "volume": volume[index],
                    "amt": volume[index] * close[index],
                }
            )
    return pd.DataFrame(rows)


class DerivedPipelineTest(unittest.TestCase):
    def test_alpha158_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha158"
            build_alpha158_artifact(daily_stock(), path, {"test": "synthetic"})
            artifact = read_alpha158_artifact(path, mmap_mode=None)
            self.assertEqual(artifact.values.shape, (65, 1, 158))
            self.assertEqual(artifact.security_codes, ("000001.SZ",))
            self.assertEqual(len(artifact.feature_names), 158)
            self.assertFalse(np.isinf(artifact.values).any())
            self.assertEqual(artifact.manifest["provenance"]["test"], "synthetic")

    def test_market63_and_market84_round_trips(self):
        cases = (
            (
                "market63", MARKET_UNIVERSES,
                ("000300.SH", "000905.SH", "000906.SH"),
                build_market63_artifact, read_market63_artifact, 63,
            ),
            (
                "market84", MARKET84_UNIVERSES, MARKET84_INDEX_CODES,
                build_market84_artifact, read_market84_artifact, 84,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for family, universes, codes, builder, reader, count in cases:
                path = Path(directory) / family
                builder(daily_index(universes, codes), path)
                artifact = reader(path, mmap_mode=None)
                self.assertEqual(artifact.values.shape, (65, count))
                self.assertEqual(len(artifact.feature_names), count)
                self.assertTrue(artifact.source_observed.all())
                self.assertEqual(artifact.manifest["family"], family)

    def test_intraday_price_volume_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intraday_price_volume"
            build_intraday_price_volume_artifact(
                minute_panel(("000001.SZ",), "sec_code"), path
            )
            artifact = read_intraday_price_volume_artifact(path, mmap_mode=None)
            self.assertEqual(artifact.values.shape, (1, 1, 248))
            self.assertEqual(len(artifact.feature_names), 248)
            np.testing.assert_allclose(artifact.bar_coverage, 1.0)
            self.assertTrue(artifact.source_observed.all())
            self.assertFalse(np.isinf(artifact.values).any())
            self.assertEqual(
                artifact.manifest["provenance"]["minute_session"],
                "09:31-11:30,13:01-15:00",
            )

    def test_intraday_market_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intraday_market"
            build_intraday_market_artifact(
                minute_panel(MARKET84_INDEX_CODES, "index_code"), path
            )
            artifact = read_intraday_market_artifact(path, mmap_mode=None)
            self.assertEqual(artifact.values.shape, (1, 192))
            self.assertEqual(len(artifact.feature_names), 192)
            self.assertTrue(artifact.source_observed.all())
            self.assertFalse(np.isinf(artifact.values).any())
            self.assertEqual(
                artifact.manifest["provenance"]["position"],
                "exploratory_market_state",
            )

    def test_artifact_is_immutable_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha158"
            build_alpha158_artifact(daily_stock(), path)
            with self.assertRaises(FileExistsError):
                build_alpha158_artifact(daily_stock(), path)
            value_path = path / "alpha158_raw.npy"
            with value_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_alpha158_artifact(path, mmap_mode=None)


if __name__ == "__main__":
    unittest.main()
