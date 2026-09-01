"""Canonical synthetic tests for Market63 and Market84 formulas."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.features.market63 import (
    MARKET_UNIVERSES,
    Market63Generator,
)
from market_guided_stock_alpha.features.market84 import (
    MARKET84_INDEX_CODES,
    MARKET84_UNIVERSES,
    Market84Generator,
)


def market63_spec():
    return {
        "universes": list(MARKET_UNIVERSES),
        "windows": [5, 10, 20, 30, 60],
        "rolling_min_periods": 1,
        "rolling_std_ddof": 1,
        "epsilon": 1e-12,
        "feature_count_per_index": 21,
        "total_feature_count": 63,
    }


def market84_spec():
    return {
        "universes": list(MARKET84_UNIVERSES),
        "index_codes": list(MARKET84_INDEX_CODES),
        "windows": [5, 10, 20, 30, 60],
        "rolling_min_periods": 1,
        "rolling_std_ddof": 1,
        "epsilon": 1e-12,
        "feature_count_per_index": 21,
        "total_feature_count": 84,
    }


def synthetic_history(universes, periods=70):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    rows = []
    for block, universe in enumerate(universes, start=1):
        for offset, day in enumerate(dates):
            rows.append(
                {
                    "trade_date": day,
                    "universe": universe,
                    "close": 100.0 * block + offset,
                    "amount": 1000.0 * block + 10.0 * offset,
                }
            )
    return pd.DataFrame(rows)


class MarketStateFeatureTest(unittest.TestCase):
    def test_market63_contract_order_and_synthetic_output(self):
        generator = Market63Generator(market63_spec())
        names = generator.get_feature_names()
        self.assertEqual(len(names), 63)
        self.assertEqual(len(set(names)), 63)
        self.assertEqual(names[0], "MKT_CSI300_RET1")
        self.assertEqual(names[21], "MKT_CSI500_RET1")
        self.assertEqual(names[42], "MKT_CSI800_RET1")
        self.assertEqual(names[-1], "MKT_CSI800_AMOUNT_STD_RATIO60")
        output = generator.transform(synthetic_history(MARKET_UNIVERSES))
        self.assertEqual(output.shape, (70, 64))
        self.assertEqual(list(output.columns[1:]), names)
        self.assertAlmostEqual(output.loc[1, "MKT_CSI300_RET1"], 0.01)
        self.assertFalse(np.isinf(output.iloc[:, 1:].to_numpy()).any())

    def test_market84_contract_order_and_synthetic_output(self):
        generator = Market84Generator(market84_spec())
        names = generator.get_feature_names()
        self.assertEqual(len(names), 84)
        self.assertEqual(len(set(names)), 84)
        expected_prefixes = (
            "MKT_CSI300_",
            "MKT_CSI500_",
            "MKT_CSI1000_",
            "MKT_CSI_ALL_",
        )
        for block, prefix in enumerate(expected_prefixes):
            self.assertTrue(
                all(name.startswith(prefix) for name in names[block * 21 : (block + 1) * 21])
            )
        self.assertEqual(names[-1], "MKT_CSI_ALL_AMOUNT_STD_RATIO60")
        output = generator.transform(synthetic_history(MARKET84_UNIVERSES))
        self.assertEqual(output.shape, (70, 85))
        self.assertEqual(list(output.columns[1:]), names)
        self.assertFalse(np.isinf(output.iloc[:, 1:].to_numpy()).any())

    def test_shared_market63_and_market84_blocks_are_identical(self):
        history84 = synthetic_history(MARKET84_UNIVERSES)
        history63 = history84.loc[
            history84["universe"].isin(("csi300", "csi500"))
        ].copy()
        csi800 = history84.loc[history84["universe"].eq("csi1000")].copy()
        csi800["universe"] = "csi800"
        history63 = pd.concat([history63, csi800], ignore_index=True)
        output63 = Market63Generator(market63_spec()).transform(history63)
        output84 = Market84Generator(market84_spec()).transform(history84)
        np.testing.assert_allclose(
            output63.iloc[:, 1:43],
            output84.iloc[:, 1:43],
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
