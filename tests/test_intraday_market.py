"""Focused formula tests for exploratory intraday market features."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.features.intraday_market import (
    FEATURE_SUFFIXES,
    UNIVERSES,
    IntradayMarketGenerator,
    feature_names,
)


def intraday_market_spec():
    return {
        "universes": list(UNIVERSES),
        "bars_per_day": 240,
        "feature_count_per_index": 48,
        "total_feature_count": 192,
        "epsilon": 1e-12,
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


def synthetic_index_panel():
    position = np.arange(240, dtype="float64")
    fields = {name: [] for name in ("open", "high", "low", "close", "volume", "amt")}
    for block in range(4):
        returns = 0.0002 * np.sin(2.0 * np.pi * position / (20.0 + block))
        close = (100.0 + block * 20.0) * np.exp(np.cumsum(returns))
        opening = np.r_[100.0 + block * 20.0, close[:-1]]
        volume = 1000.0 + block * 100.0 + position
        values = {
            "open": opening,
            "high": np.maximum(opening, close) * 1.001,
            "low": np.minimum(opening, close) * 0.999,
            "close": close,
            "volume": volume,
            "amt": volume * close,
        }
        for name, value in values.items():
            fields[name].append(value)
    return {name: np.asarray(blocks) for name, blocks in fields.items()}


class IntradayMarketCanonicalTest(unittest.TestCase):
    def test_four_by_48_order_is_unique_and_stable(self):
        names = feature_names()
        self.assertEqual(len(FEATURE_SUFFIXES), 48)
        self.assertEqual(len(names), 192)
        self.assertEqual(len(set(names)), 192)
        self.assertEqual(names[0], "IMKT_CSI300_RET_OC")
        self.assertEqual(names[48], "IMKT_CSI500_RET_OC")
        self.assertEqual(names[96], "IMKT_CSI1000_RET_OC")
        self.assertEqual(names[144], "IMKT_CSI_ALL_RET_OC")
        self.assertEqual(names[-1], "IMKT_CSI_ALL_VOLUME_TREND")

    def test_synthetic_four_index_day_outputs_192(self):
        generator = IntradayMarketGenerator(intraday_market_spec())
        output, diagnostics = generator.compute_day(synthetic_index_panel())
        self.assertEqual(output.shape, (192,))
        self.assertFalse(np.isinf(output).any())
        self.assertTrue(diagnostics["source_observed"])
        self.assertEqual(len(diagnostics["indices"]), 4)
        self.assertEqual(generator.get_feature_names(), feature_names())


if __name__ == "__main__":
    unittest.main()
