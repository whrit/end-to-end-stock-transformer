"""Focused tests for the canonical Alpha158 formula layer."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.features import (
    KLINE_FEATURES,
    PRICE_FEATURES,
    ROLLING_FAMILIES,
    Alpha158DataError,
    Alpha158Generator,
    Alpha158SpecError,
)


ROLLING_WINDOWS = [5, 10, 20, 30, 60]


def alpha158_spec():
    return {
        "name": "alpha158",
        "version": 1,
        "source": {
            "repository": "microsoft/qlib",
            "file": "qlib/contrib/data/loader.py",
            "class": "Alpha158DL",
            "blob_sha": "4d11f3a34cc1d12728723956c8e9324556eee8e3",
        },
        "epsilon": 1e-12,
        "rolling_windows": ROLLING_WINDOWS,
        "rolling_min_periods": 1,
        "rolling_std_ddof": 1,
        "sanitize_non_finite_to_nan": True,
        "feature_count": 158,
        "feature_group_order": ["kbar", "price", "rolling"],
        "kbar_features": list(KLINE_FEATURES),
        "price_features": list(PRICE_FEATURES),
        "rolling_family_order": list(ROLLING_FAMILIES),
    }


def synthetic_ohlcv(row_count=65):
    index = np.arange(row_count, dtype="float64")
    close = 20.0 + index * 0.25
    return pd.DataFrame(
        {
            "trade_date": pd.bdate_range("2024-01-02", periods=row_count),
            "sec_code": "000001.SZ",
            "open": close - 0.10,
            "high": close + 0.30,
            "low": close - 0.40,
            "close": close,
            "vwap": close - 0.02,
            "volume": 1000.0 + index * 5.0,
        }
    )


class Alpha158CanonicalTest(unittest.TestCase):
    def setUp(self):
        self.generator = Alpha158Generator(alpha158_spec())

    def test_feature_count_names_and_order_are_fixed(self):
        expected = list(KLINE_FEATURES) + list(PRICE_FEATURES) + [
            "{}{}".format(family, window)
            for family in ROLLING_FAMILIES
            for window in ROLLING_WINDOWS
        ]
        names = self.generator.get_feature_names()
        self.assertEqual(len(names), 158)
        self.assertEqual(len(set(names)), 158)
        self.assertEqual(names, expected)
        self.assertEqual(Alpha158Generator(alpha158_spec()).get_feature_names(), names)

    def test_synthetic_ohlcv_produces_158_features(self):
        history = synthetic_ohlcv()
        output = self.generator.transform_market_history(history)
        self.assertEqual(len(output), len(history))
        self.assertEqual(
            list(output.columns),
            ["trade_date", "sec_code"] + self.generator.get_feature_names(),
        )
        self.assertEqual(output.loc[:, self.generator.get_feature_names()].shape[1], 158)
        finite_or_nan = output.loc[:, self.generator.get_feature_names()].to_numpy(
            dtype="float64"
        )
        self.assertFalse(np.isinf(finite_or_nan).any())

    def test_rolling_warmup_and_nan_semantics_match_reference(self):
        output = self.generator.transform_market_history(synthetic_ohlcv())
        self.assertTrue(output.loc[:4, "ROC5"].isna().all())
        self.assertTrue(np.isfinite(output.loc[5, "ROC5"]))
        self.assertTrue(np.isnan(output.loc[0, "STD5"]))
        self.assertTrue(np.isfinite(output.loc[1, "STD5"]))
        self.assertAlmostEqual(output.loc[0, "MA60"], 1.0)
        self.assertTrue(np.isnan(output.loc[0, "CORD5"]))

    def test_invalid_inputs_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "missing columns.*volume"):
            self.generator.transform_market_history(
                synthetic_ohlcv().drop(columns="volume")
            )

        invalid_date = synthetic_ohlcv()
        invalid_date.loc[0, "trade_date"] = pd.NaT
        with self.assertRaisesRegex(Alpha158DataError, "invalid trade_date"):
            self.generator.transform_market_history(invalid_date)

        invalid_code = synthetic_ohlcv()
        invalid_code.loc[0, "sec_code"] = "000001"
        with self.assertRaisesRegex(Alpha158DataError, "exchange guessing is disabled"):
            self.generator.transform_market_history(invalid_code)

        duplicate = pd.concat(
            [synthetic_ohlcv(), synthetic_ohlcv().iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(Alpha158DataError, "duplicate keys"):
            self.generator.transform_market_history(duplicate)

    def test_invalid_fixed_spec_is_rejected(self):
        invalid = alpha158_spec()
        invalid["feature_count"] = 157
        with self.assertRaisesRegex(Alpha158SpecError, "must equal 158"):
            Alpha158Generator(invalid)


if __name__ == "__main__":
    unittest.main()
