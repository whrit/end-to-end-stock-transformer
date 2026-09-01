"""Focused mathematical and contract tests for intraday price-volume features."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import market_guided_stock_alpha.features.intraday.compute as compute
from market_guided_stock_alpha.features.intraday import IntradayPriceVolumeGenerator
from market_guided_stock_alpha.features.intraday.common import (
    IntradayPriceVolumeConfig,
    prepare_day,
)
from market_guided_stock_alpha.features.intraday.geometry import (
    geometry_summaries,
    trapezoid,
)
from market_guided_stock_alpha.features.intraday.registry import (
    FACTOR_IDS,
    FACTOR_NAMES,
    FACTOR_REGISTRY,
    factor_manifest,
)
from market_guided_stock_alpha.features.intraday.spectral import (
    fft_extended_summaries,
)


def intraday_price_volume_config(min_coverage=0.90):
    return {
        "name": "intraday_price_volume",
        "factor_count": 248,
        "min_bar_coverage": min_coverage,
        "structural_zero_threshold": 0.80,
        "epsilon": 1e-12,
        "zero_return_tolerance": 1e-12,
        "minimum_observations": {
            "correlation": 10,
            "multiscale_correlation": 4,
            "fft": 20,
            "geometry": 5,
        },
        "session_contract": {
            "bars_per_day": 240,
            "morning_bars": 120,
            "afternoon_bars": 120,
        },
        "stock_chunk_size": 8,
        "output_dtype": "float32",
    }


def synthetic_day(stock_count=3):
    position = np.arange(240, dtype="float64")[:, None]
    phase = np.arange(stock_count, dtype="float64")[None, :] * 0.4
    returns = 0.0003 * np.sin(2.0 * np.pi * position / 20.0 + phase)
    returns += 0.0001 * np.cos(2.0 * np.pi * position / 60.0 + phase)
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))
    opening = np.vstack([np.full((1, stock_count), 100.0), close[:-1]])
    volume = 1000.0 + 100.0 * np.sin(2.0 * np.pi * position / 30.0 + phase)
    return {
        "open": opening,
        "high": np.maximum(opening, close) * 1.001,
        "low": np.minimum(opening, close) * 0.999,
        "close": close,
        "volume": volume,
        "amt": volume * close,
    }


class IntradayPriceVolumeCanonicalTest(unittest.TestCase):
    def test_registry_is_exact_unique_f001_to_f248(self):
        self.assertEqual(len(FACTOR_REGISTRY), 248)
        self.assertEqual(FACTOR_IDS, tuple("F{:03d}".format(i) for i in range(1, 249)))
        self.assertEqual(len(set(FACTOR_IDS)), 248)
        self.assertEqual(len(set(FACTOR_NAMES)), 248)
        self.assertEqual([row[1] for row in FACTOR_REGISTRY], list(FACTOR_NAMES))
        manifest = factor_manifest()
        self.assertEqual([row["factor_index"] for row in manifest], list(range(248)))

    def test_registry_and_live_implementations_are_complete(self):
        context = prepare_day(
            synthetic_day(2), IntradayPriceVolumeConfig.from_mapping(intraday_price_volume_config())
        )
        implemented = []
        for module in compute.FACTOR_MODULES:
            implemented.extend(module.compute(context))
        self.assertEqual(len(implemented), 248)
        self.assertEqual(len(set(implemented)), 248)
        self.assertEqual(set(implemented), set(FACTOR_NAMES))

    def test_synthetic_240_bar_day_outputs_248_without_inf(self):
        generator = IntradayPriceVolumeGenerator(intraday_price_volume_config())
        matrix, coverage, eligible = generator.transform_day_arrays(
            synthetic_day(3)
        )
        self.assertEqual(matrix.shape, (3, 248))
        self.assertEqual(matrix.dtype, np.dtype("float32"))
        self.assertFalse(np.isinf(matrix).any())
        np.testing.assert_allclose(coverage, 1.0)
        self.assertEqual(eligible.tolist(), [True, True, True])

    def test_coverage_boundary_and_invalid_observation_mask(self):
        raw = synthetic_day(2)
        raw["close"][:24, 0] = np.nan
        raw["close"][:25, 1] = np.nan
        matrix, coverage, eligible = IntradayPriceVolumeGenerator(
            intraday_price_volume_config(0.90)
        ).transform_day_arrays(raw)
        self.assertAlmostEqual(float(coverage[0]), 0.90, places=6)
        self.assertAlmostEqual(float(coverage[1]), 215.0 / 240.0, places=6)
        self.assertEqual(eligible.tolist(), [True, False])
        self.assertTrue(np.isnan(matrix[1]).all())

    def test_structural_inactive_rows_are_excluded(self):
        raw = synthetic_day(5)
        raw["volume"][10, :4] = 0.0
        raw["amt"][10, :4] = 0.0
        raw["volume"][11, :3] = 0.0
        raw["amt"][11, :3] = 0.0
        context = prepare_day(
            raw, IntradayPriceVolumeConfig.from_mapping(intraday_price_volume_config())
        )
        self.assertTrue(bool(context.structural_inactive[10]))
        self.assertFalse(bool(context.structural_inactive[11]))
        self.assertTrue(np.isnan(context.returns[10]).all())

    def test_missing_and_duplicate_implementation_checks(self):
        generator = IntradayPriceVolumeGenerator(intraday_price_volume_config())
        empty = types.SimpleNamespace(compute=lambda context: {})
        with patch.object(compute, "FACTOR_MODULES", (empty,)):
            with self.assertRaisesRegex(ValueError, "mismatch missing"):
                generator.transform_day_arrays(synthetic_day(1))
        duplicate = types.SimpleNamespace(
            compute=lambda context: {"ret_skew": np.zeros(context.stock_count)}
        )
        with patch.object(compute, "FACTOR_MODULES", (duplicate, duplicate)):
            with self.assertRaisesRegex(ValueError, "duplicate intraday factor"):
                generator.transform_day_arrays(synthetic_day(1))

    def test_frequency_and_geometry_sanity(self):
        position = np.arange(240, dtype="float64")
        sine = np.sin(2.0 * np.pi * position / 20.0)[:, None]
        spectrum = fft_extended_summaries(sine, 20, 4, 1e-12)
        self.assertAlmostEqual(float(spectrum["dominant_period"][0]), 20.0)
        self.assertTrue(np.isfinite(spectrum["centroid"][0]))
        self.assertFalse(any(np.isinf(value).any() for value in spectrum.values()))

        paths = np.column_stack([sine[:, 0] * 0.001, np.full(240, 0.001)])
        geometry = geometry_summaries(paths, 5, 4, 1e-12)
        self.assertTrue(np.isfinite(geometry["path_efficiency"]).all())
        self.assertTrue(np.isfinite(geometry["abs_chord_area"]).all())
        self.assertTrue(np.isfinite(geometry["curvature_energy"]).all())
        self.assertFalse(any(np.isinf(value).any() for value in geometry.values()))

    def test_numpy_one_and_two_trapezoid_compatibility_entrypoint(self):
        x = np.asarray([0.0, 1.0, 2.0])
        y = np.asarray([0.0, 1.0, 0.0])
        self.assertAlmostEqual(float(trapezoid(y, x)), 1.0)

    def test_session_contract_and_day_fields_fail_loudly(self):
        invalid_config = intraday_price_volume_config()
        invalid_config["session_contract"] = {
            "bars_per_day": 239,
            "morning_bars": 120,
            "afternoon_bars": 119,
        }
        with self.assertRaisesRegex(ValueError, r"120\+120=240"):
            IntradayPriceVolumeGenerator(invalid_config)
        incomplete = synthetic_day(1)
        incomplete.pop("amt")
        with self.assertRaisesRegex(ValueError, "day fields must equal"):
            IntradayPriceVolumeGenerator(intraday_price_volume_config()).transform_day_arrays(incomplete)


if __name__ == "__main__":
    unittest.main()
