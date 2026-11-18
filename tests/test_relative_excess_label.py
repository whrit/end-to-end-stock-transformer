"""Focused tests for the canonical public relative-excess label pipeline."""

import importlib.util
import json
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

from market_guided_stock_alpha.labels.derived_label import DerivedLabelReader
from market_guided_stock_alpha.labels.relative_excess_return import (
    DEFAULT_BENCHMARK_CODE,
    DEFAULT_VALUE_COLUMN,
    HOLDING_PERIOD_TRADING_DAYS,
    LABEL_END_OFFSET_TRADING_DAYS,
    LABEL_START_OFFSET_TRADING_DAYS,
    PURGE_DEPENDENCY_OFFSET_TRADING_DAYS,
    build_calendar_mapping,
    build_relative_excess_label,
)


def load_builder_script():
    path = ROOT / "scripts" / "build_relative_excess_label.py"
    spec = importlib.util.spec_from_file_location("canonical_label_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load canonical label builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RelativeExcessLabelTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-02", periods=10)
        self.calendar = pd.DataFrame(
            {"calendar_date": self.dates, "is_open": 1}
        )
        self.benchmark = pd.DataFrame(
            {
                "trade_date": self.dates,
                "index_code": DEFAULT_BENCHMARK_CODE,
                "open": np.arange(200.0, 210.0),
            }
        )

    def _stock(self, t1=110.0, t6=132.0):
        return pd.DataFrame(
            {
                "trade_date": [self.dates[1], self.dates[2], self.dates[6]],
                "sec_code": ["000001.SZ"] * 3,
                "adj_open": [t1, 999.0, t6],
            }
        )

    def test_horizon_constants(self):
        self.assertEqual(DEFAULT_BENCHMARK_CODE, "000905.SH")
        self.assertEqual(LABEL_START_OFFSET_TRADING_DAYS, 1)
        self.assertEqual(LABEL_END_OFFSET_TRADING_DAYS, 6)
        self.assertEqual(HOLDING_PERIOD_TRADING_DAYS, 5)
        self.assertEqual(PURGE_DEPENDENCY_OFFSET_TRADING_DAYS, 6)

    def test_calendar_mapping_uses_t1_and_t6(self):
        mapping = build_calendar_mapping(self.calendar)
        first = mapping.iloc[0]
        self.assertEqual(first["trade_date"], self.dates[0])
        self.assertEqual(first["label_date_t1"], self.dates[1])
        self.assertEqual(first["label_date_t6"], self.dates[6])

    def test_formula_uses_calendar_t1_and_t6_not_stock_row_shift(self):
        output, diagnostics = build_relative_excess_label(
            self._stock(), self.benchmark, self.calendar
        )
        benchmark_gross = 206.0 / 201.0
        expected = (132.0 / 110.0) / benchmark_gross - 1.0
        self.assertAlmostEqual(
            output.loc[(self.dates[0], "000001.SZ"), DEFAULT_VALUE_COLUMN],
            expected,
        )
        self.assertNotAlmostEqual(
            output.iloc[0, 0], (999.0 / 110.0) / benchmark_gross - 1.0
        )
        self.assertEqual(
            diagnostics["formula"]["label_end_offset_trading_days"], 6
        )

    def test_missing_stock_t1_or_t6_is_omitted(self):
        for missing_date in (self.dates[1], self.dates[6]):
            with self.subTest(missing_date=missing_date):
                stock = self._stock()
                stock = stock.loc[~stock["trade_date"].eq(missing_date)]
                output, _ = build_relative_excess_label(
                    stock, self.benchmark, self.calendar
                )
                self.assertNotIn((self.dates[0], "000001.SZ"), output.index)

    def _delisting_stock(self):
        # Trades days 1..3 (adjusted close 120 on day 3), then delists: no t+6 open.
        return pd.DataFrame(
            {
                "trade_date": self.dates[1:4],
                "sec_code": ["BUST.US.20240105"] * 3,
                "adj_open": [110.0, 115.0, 118.0],
                "close": [112.0, 116.0, 120.0],
            }
        )

    def test_delisting_exits_at_last_close_times_exit_factor(self):
        benchmark_gross = 206.0 / 201.0
        for kind, factor in (("merger", 1.0), ("performance", 0.45)):
            with self.subTest(kind=kind):
                delistings = pd.DataFrame(
                    {"sec_code": ["BUST.US.20240105"], "last_trade_date": [self.dates[3]],
                     "kind": [kind], "exit_factor": [factor]}
                )
                output, diagnostics = build_relative_excess_label(
                    self._delisting_stock(), self.benchmark, self.calendar, delistings=delistings
                )
                value = output.loc[(self.dates[0], "BUST.US.20240105"), DEFAULT_VALUE_COLUMN]
                self.assertAlmostEqual(value, (120.0 * factor / 110.0) / benchmark_gross - 1.0)
                # Anchors whose t+1 is after the last trade stay dropped.
                self.assertNotIn((self.dates[3], "BUST.US.20240105"), output.index)
                self.assertEqual(diagnostics["delisting_adjusted_by_kind"], {kind: 3})

    def test_without_delistings_a_delisted_stock_has_no_label(self):
        output, diagnostics = build_relative_excess_label(
            self._delisting_stock(), self.benchmark, self.calendar
        )
        self.assertEqual(len(output), 0)
        self.assertNotIn("delisting_adjusted_count", diagnostics)

    def test_missing_benchmark_t1_or_t6_omits_anchor(self):
        for missing_date in (self.dates[1], self.dates[6]):
            with self.subTest(missing_date=missing_date):
                benchmark = self.benchmark.loc[
                    ~self.benchmark["trade_date"].eq(missing_date)
                ]
                output, _ = build_relative_excess_label(
                    self._stock(), benchmark, self.calendar
                )
                self.assertNotIn((self.dates[0], "000001.SZ"), output.index)

    def test_invalid_or_non_positive_prices_are_omitted(self):
        stock_cases = (
            self._stock(t1=0.0),
            self._stock(t6=-1.0),
            self._stock(t1=np.nan),
            self._stock(t6=np.inf),
        )
        for stock in stock_cases:
            with self.subTest(stock_prices=stock["adj_open"].tolist()):
                output, _ = build_relative_excess_label(
                    stock, self.benchmark, self.calendar
                )
                self.assertNotIn((self.dates[0], "000001.SZ"), output.index)

        for invalid in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(benchmark_price=invalid):
                benchmark = self.benchmark.copy()
                benchmark.loc[
                    benchmark["trade_date"].eq(self.dates[1]), "open"
                ] = invalid
                output, _ = build_relative_excess_label(
                    self._stock(), benchmark, self.calendar
                )
                self.assertNotIn((self.dates[0], "000001.SZ"), output.index)

    def test_output_is_finite_unique_sorted_multiindex(self):
        stock = pd.concat(
            [
                self._stock().assign(sec_code="600000.SH"),
                self._stock(),
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=7)
        output, _ = build_relative_excess_label(
            stock,
            self.benchmark.sample(frac=1.0, random_state=3),
            self.calendar.sample(frac=1.0, random_state=5),
            value_column="custom_relative_label",
        )
        self.assertIsInstance(output.index, pd.MultiIndex)
        self.assertEqual(list(output.columns), ["custom_relative_label"])
        self.assertEqual(list(output.index.names), ["trade_date", "sec_code"])
        self.assertFalse(output.index.has_duplicates)
        self.assertTrue(output.index.is_monotonic_increasing)
        self.assertTrue(np.isfinite(output.to_numpy(dtype="float64")).all())


class DerivedLabelReaderTest(unittest.TestCase):
    def test_configurable_column_and_missing_membership_mask(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.pkl"
            index = pd.MultiIndex.from_tuples(
                [("2024-01-02", "000001.SZ")],
                names=["trade_date", "sec_code"],
            )
            pd.DataFrame({"my_public_label": [0.125]}, index=index).to_pickle(path)
            target = pd.DataFrame(
                {
                    "trade_date": pd.to_datetime(
                        ["2024-01-02", "2024-01-02"]
                    ),
                    "sec_code": ["000001.SZ", "000002.SZ"],
                    "security_id": [1, 2],
                }
            )
            output, metadata = DerivedLabelReader(
                path, "my_public_label"
            ).align(target)
            self.assertEqual(output["label_mask"].tolist(), [True, False])
            self.assertAlmostEqual(output.loc[0, "label_raw"], 0.125)
            self.assertTrue(np.isnan(output.loc[1, "label_raw"]))
            self.assertEqual(metadata["raw_label_column"], "my_public_label")
            self.assertEqual(metadata["computed_label_count"], 0)


class RelativeExcessBuilderTest(unittest.TestCase):
    def test_builder_artifact_reader_round_trip(self):
        builder = load_builder_script()
        dates = pd.bdate_range("2024-01-02", periods=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stock_path = root / "stock.csv"
            benchmark_path = root / "benchmark.csv"
            calendar_path = root / "calendar.csv"
            output_path = root / "derived.pkl"
            manifest_path = root / "derived.manifest.json"
            pd.DataFrame(
                {
                    "trade_date": [dates[1], dates[6]],
                    "sec_code": ["000001.SZ", "000001.SZ"],
                    "adj_open": [10.0, 12.0],
                }
            ).to_csv(stock_path, index=False)
            pd.DataFrame(
                {
                    "trade_date": dates,
                    "index_code": DEFAULT_BENCHMARK_CODE,
                    "open": np.arange(100.0, 108.0),
                }
            ).to_csv(benchmark_path, index=False)
            pd.DataFrame(
                {"calendar_date": dates, "is_open": 1}
            ).to_csv(calendar_path, index=False)

            manifest = builder.build_from_files(
                stock_path,
                benchmark_path,
                calendar_path,
                output_path,
                manifest_path,
                value_column="historical_reproduction_column",
            )
            saved = pd.read_pickle(output_path)
            on_disk_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                list(saved.columns), ["historical_reproduction_column"]
            )
            self.assertEqual(
                manifest["output"]["sha256"],
                on_disk_manifest["output"]["sha256"],
            )
            self.assertEqual(
                manifest["formula"]["holding_period_trading_days"], 5
            )
            self.assertEqual(
                manifest["formula"]["purge_dependency_offset_trading_days"], 6
            )
            labels, metadata = DerivedLabelReader(
                output_path,
                "historical_reproduction_column",
                manifest_path,
            ).load()
            self.assertEqual(len(labels), 1)
            self.assertEqual(
                metadata["derived_label_manifest_sha256"],
                builder.sha256_path(manifest_path),
            )
            self.assertEqual(
                metadata["raw_label_column"], "historical_reproduction_column"
            )


if __name__ == "__main__":
    unittest.main()
