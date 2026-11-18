"""Canonical dataset alignment, layout, normalization, and mmap tests."""

import gc
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

from market_guided_stock_alpha.data import TradingCalendar
from market_guided_stock_alpha.dataset import (
    AlignmentError,
    DatasetStatistics,
    FeatureLayout,
    FeatureStatistics,
    SequenceDataset,
    build_split_contract,
    feature_names,
    fit_dataset_statistics,
    fit_robust_statistics,
    normalize_cross_section_labels,
)
from market_guided_stock_alpha.derived import (
    read_alpha158_artifact,
    read_intraday_market_artifact,
    read_intraday_price_volume_artifact,
    read_market63_artifact,
    read_market84_artifact,
    write_feature_artifact,
)
from market_guided_stock_alpha.labels import (
    DEFAULT_VALUE_COLUMN,
    PURGE_DEPENDENCY_OFFSET_TRADING_DAYS,
    DerivedLabelReader,
)


CODES_PHYSICAL = ("000003.SZ", "000001.SZ", "000004.SZ", "000002.SZ")


def calendar_frame(periods=30):
    return pd.DataFrame(
        {
            "calendar_date": pd.bdate_range("2024-01-02", periods=periods),
            "is_open": 1,
        }
    )


def identity_statistics(layout):
    result = {}
    families = tuple(layout.stock_families) + (layout.market_family,)
    if layout.intraday_market_family:
        families += (layout.intraday_market_family,)
    for family in families:
        names = feature_names(family)
        count = len(names)
        result[family] = FeatureStatistics(
            feature_names=names,
            median=np.zeros(count, dtype="float64"),
            mad=np.ones(count, dtype="float64"),
            scale=np.ones(count, dtype="float64"),
            finite_count=np.ones(count, dtype="int64"),
            zero_mad_mask=np.zeros(count, dtype="bool"),
        )
    return DatasetStatistics(result)


def write_cross_section(root, family, dates, codes, missing_date=None):
    names = feature_names(family)
    kept_dates = pd.DatetimeIndex(
        [date for date in dates if pd.Timestamp(date) != missing_date]
    )
    values = np.empty((len(kept_dates), len(codes), len(names)), dtype="float32")
    for date_position in range(len(kept_dates)):
        for code_position, code in enumerate(codes):
            code_value = int(code[:6])
            values[date_position, code_position, :] = code_value + date_position * 0.01
    observed = np.ones(values.shape[:2], dtype="bool")
    if family == "alpha158":
        observed[1, 1] = False
    path = root / family
    write_feature_artifact(
        path,
        family,
        values,
        kept_dates,
        names,
        security_codes=codes,
        source_observed=observed,
    )
    if family == "alpha158":
        return read_alpha158_artifact(path)
    return read_intraday_price_volume_artifact(path)


def write_market(root, family, dates, missing_date=None):
    names = feature_names(family)
    kept_dates = pd.DatetimeIndex(
        [date for date in dates if pd.Timestamp(date) != missing_date]
    )
    position = np.arange(len(kept_dates), dtype="float64")[:, None]
    values = position + np.arange(len(names), dtype="float64")[None, :] * 0.001
    path = root / family
    write_feature_artifact(
        path,
        family,
        values,
        kept_dates,
        names,
        source_observed=np.ones(len(kept_dates), dtype="bool"),
    )
    if family == "market63":
        return read_market63_artifact(path)
    if family == "market84":
        return read_market84_artifact(path)
    return read_intraday_market_artifact(path)


def membership_and_labels(root, dates, include_unknown=False):
    membership_codes = {
        dates[7]: ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"),
        dates[8]: ("000001.SZ", "000003.SZ", "000004.SZ"),
        dates[9]: ("000001.SZ", "000002.SZ", "000004.SZ"),
    }
    rows = []
    label_rows = []
    for day, codes in membership_codes.items():
        current = list(codes)
        if include_unknown and day == dates[7]:
            current.append("000005.SZ")
        for position, code in enumerate(current):
            rows.append(
                {
                    "trade_date": day,
                    "sec_code": code,
                }
            )
            value = float(position + 1)
            if day == dates[8] and code == "000004.SZ":
                value = np.nan
            label_rows.append(
                {
                    "trade_date": day,
                    "sec_code": code,
                    DEFAULT_VALUE_COLUMN: value,
                }
            )
    label_path = root / "labels.pkl"
    pd.DataFrame(label_rows).to_pickle(label_path)
    return pd.DataFrame(rows), DerivedLabelReader(label_path, DEFAULT_VALUE_COLUMN)


class DatasetCanonicalTest(unittest.TestCase):
    def _build(
        self,
        root,
        layout,
        missing_alpha_date=None,
        missing_market_date=None,
        include_unknown=False,
    ):
        dates = pd.bdate_range("2024-01-02", periods=12)
        alpha = write_cross_section(
            root, "alpha158", dates, CODES_PHYSICAL, missing_alpha_date
        )
        market = write_market(
            root, layout.market_family, dates, missing_market_date
        )
        intraday = None
        if "intraday_price_volume" in layout.stock_families:
            intraday = write_cross_section(
                root,
                "intraday_price_volume",
                dates,
                ("000002.SZ", "000001.SZ", "000003.SZ"),
                missing_date=dates[3],
            )
        membership, reader = membership_and_labels(root, dates, include_unknown)
        dataset = SequenceDataset(
            alpha,
            market,
            membership,
            reader,
            TradingCalendar(calendar_frame()),
            identity_statistics(layout),
            layout=layout,
            intraday_price_volume=intraday,
        )
        return dataset, alpha, market, intraday

    def test_stage1_layout_sequence_dynamic_membership_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, alpha, market, _ = self._build(
                Path(directory), FeatureLayout.stage1()
            )
            first = dataset[0]
            second = dataset[1]
            self.assertEqual(first.stock_features.shape, (4, 8, 158))
            self.assertEqual(first.market_features.shape, (8, 63))
            self.assertEqual(len(second.security_codes), 3)
            self.assertEqual(first.security_codes, tuple(sorted(first.security_codes)))
            self.assertEqual(first.security_ids.tolist(), [0, 1, 2, 3])
            self.assertAlmostEqual(float(first.stock_features[0, -1, 0]), 1.07, places=6)
            self.assertEqual(first.history_dates[-1], first.prediction_date)
            self.assertTrue((first.history_dates <= first.prediction_date).all())
            self.assertFalse(first.metadata["market_physically_repeated"])
            self.assertEqual(first.label_mask.tolist(), [True, True, True, True])
            self.assertEqual(second.label_mask.tolist(), [True, True, False])
            del dataset, first, second, alpha, market
            gc.collect()

    def test_optional_membership_metadata_does_not_define_the_universe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.bdate_range("2024-01-02", periods=12)
            alpha = write_cross_section(root, "alpha158", dates, CODES_PHYSICAL)
            market = write_market(root, "market63", dates)
            membership, reader = membership_and_labels(root, dates)
            membership["index_code"] = "000300.SH"
            membership["weight"] = 1.0
            dataset = SequenceDataset(
                alpha,
                market,
                membership,
                reader,
                TradingCalendar(calendar_frame()),
                identity_statistics(FeatureLayout.stage1()),
            )
            current = dataset[0]
            self.assertEqual(current.security_codes, tuple(sorted(
                membership.loc[
                    membership["trade_date"].eq(dates[7]), "sec_code"
                ]
            )))
            del current, dataset, alpha, market
            gc.collect()

    def test_stage2_406_market63_and_missing_intraday_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = FeatureLayout.stage2()
            dataset, alpha, market, intraday = self._build(Path(directory), layout)
            sample = dataset[0]
            self.assertEqual(layout.stock_dim, 406)
            self.assertEqual(layout.market_dim, 63)
            self.assertEqual(layout.conceptual_input_dim, 469)
            self.assertEqual(sample.stock_features.shape, (4, 8, 406))
            intraday_mask = sample.component_masks["intraday_price_volume"]
            self.assertFalse(intraday_mask[:, 3].any())
            self.assertFalse(intraday_mask[3].any())
            self.assertTrue((sample.stock_features[:, 3, 158:] == 0.0).all())
            self.assertTrue((sample.stock_features[3, :, 158:] == 0.0).all())
            del dataset, sample, alpha, market, intraday
            gc.collect()

    def test_market84_secondary_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = FeatureLayout.stage2("market84")
            dataset, alpha, market, intraday = self._build(Path(directory), layout)
            sample = dataset[0]
            self.assertEqual(sample.stock_features.shape, (4, 8, 406))
            self.assertEqual(sample.market_features.shape, (8, 84))
            self.assertEqual(layout.market_dim, 84)
            del dataset, sample, alpha, market, intraday
            gc.collect()

    def test_exploratory_dual_frequency_remains_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.bdate_range("2024-01-02", periods=12)
            layout = FeatureLayout.dual_frequency()
            alpha = write_cross_section(root, "alpha158", dates, CODES_PHYSICAL)
            intraday = write_cross_section(
                root,
                "intraday_price_volume",
                dates,
                ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"),
            )
            market = write_market(root, "market84", dates)
            intraday_market = write_market(root, "intraday_market", dates)
            membership, reader = membership_and_labels(root, dates)
            dataset = SequenceDataset(
                alpha,
                market,
                membership,
                reader,
                TradingCalendar(calendar_frame()),
                identity_statistics(layout),
                layout=layout,
                intraday_price_volume=intraday,
                intraday_market=intraday_market,
            )
            sample = dataset[0]
            self.assertEqual(sample.stock_features.shape, (4, 8, 406))
            self.assertEqual(sample.market_features.shape, (8, 84))
            self.assertEqual(sample.intraday_market_features.shape, (8, 192))
            self.assertEqual(layout.total_market_dim, 276)
            self.assertEqual(layout.conceptual_input_dim, 682)
            del dataset, sample, alpha, intraday, market, intraday_market
            gc.collect()

    def test_train_only_statistics_fit_uses_membership_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.bdate_range("2024-01-02", periods=12)
            layout = FeatureLayout.stage2()
            alpha = write_cross_section(root, "alpha158", dates, CODES_PHYSICAL)
            intraday = write_cross_section(
                root,
                "intraday_price_volume",
                dates,
                ("000001.SZ", "000002.SZ", "000003.SZ"),
            )
            market = write_market(root, "market63", dates)
            membership, _reader = membership_and_labels(root, dates)
            statistics = fit_dataset_statistics(
                layout,
                alpha,
                market,
                membership,
                dates[7:10],
                intraday_price_volume=intraday,
            )
            self.assertEqual(
                set(statistics.by_family),
                {"alpha158", "intraday_price_volume", "market63"},
            )
            self.assertTrue(
                (statistics.by_family["alpha158"].finite_count > 0).all()
            )
            del alpha, intraday, market
            gc.collect()

    def test_missing_required_security_or_date_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AlignmentError, "missing required securities"):
                self._build(
                    Path(directory), FeatureLayout.stage1(), include_unknown=True
                )
        with tempfile.TemporaryDirectory() as directory:
            dates = pd.bdate_range("2024-01-02", periods=12)
            with self.assertRaisesRegex(AlignmentError, "missing required dates"):
                self._build(
                    Path(directory),
                    FeatureLayout.stage1(),
                    missing_alpha_date=dates[2],
                )
        with tempfile.TemporaryDirectory() as directory:
            dates = pd.bdate_range("2024-01-02", periods=12)
            with self.assertRaisesRegex(AlignmentError, "missing required dates"):
                self._build(
                    Path(directory),
                    FeatureLayout.stage1(),
                    missing_market_date=dates[4],
                )

    def test_dataset_retains_mmap_and_materializes_only_requested_window(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, alpha, market, _ = self._build(
                Path(directory), FeatureLayout.stage1()
            )
            self.assertIsInstance(alpha.values, np.memmap)
            self.assertIs(dataset.artifacts["alpha158"].values, alpha.values)
            sample = dataset[0]
            self.assertLess(sample.stock_features.nbytes, alpha.values.nbytes)
            del dataset, sample, alpha, market
            gc.collect()

    def test_formal_robust_and_daily_label_normalization(self):
        stats = fit_robust_statistics(
            np.asarray([[1.0], [3.0], [100.0]]), ("feature",)
        )
        self.assertAlmostEqual(float(stats.median[0]), 3.0)
        self.assertAlmostEqual(float(stats.mad[0]), 2.0)
        self.assertAlmostEqual(float(stats.scale[0]), 2.0 * 1.4826)
        transformed = stats.transform(np.asarray([[np.nan], [1000.0]]))
        self.assertEqual(float(transformed[0, 0]), 0.0)
        self.assertEqual(float(transformed[1, 0]), 3.0)

        labels = pd.DataFrame(
            {
                "trade_date": pd.Timestamp("2024-01-02"),
                "sec_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "label_raw": [0.0, 1.0, 100.0, np.nan],
                "label_mask": [True, True, True, False],
            }
        )
        output = normalize_cross_section_labels(labels)
        processed = output.loc[output["label_mask"], "label"].to_numpy()
        self.assertAlmostEqual(float(processed.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(processed.std(ddof=0)), 1.0, places=7)
        self.assertEqual(output["label_mask"].tolist(), [True, True, True, False])

    def test_trading_day_split_uses_label_t6_dependency(self):
        calendar = TradingCalendar(calendar_frame(40))
        dates = calendar.open_dates
        result = build_split_contract(
            calendar,
            dates[0], dates[14],
            dates[15], dates[24],
            dates[25], dates[32],
        )
        self.assertEqual(
            result.purge_dependency, PURGE_DEPENDENCY_OFFSET_TRADING_DAYS
        )
        self.assertTrue(
            all(calendar.next_trading_date(day, 6) < result.valid_dates[0]
                for day in result.train_dates)
        )
        self.assertTrue(
            all(calendar.next_trading_date(day, 6) < result.test_dates[0]
                for day in result.valid_dates)
        )
        self.assertGreater(len(result.purged_train_dates), 0)
        self.assertGreater(len(result.purged_valid_dates), 0)


class RobustFitBlockingTest(unittest.TestCase):
    def test_column_blocking_is_exact_for_float64_and_float32(self):
        import warnings
        from unittest import mock

        import market_guided_stock_alpha.dataset.normalization as normalization

        rng = np.random.default_rng(3)
        values = rng.standard_t(3, size=(503, 11))
        values[rng.random(values.shape) < 0.05] = np.nan
        values[rng.random(values.shape) < 0.01] = np.inf
        values[:, 4] = 2.0  # zero MAD
        names = ["f{}".format(index) for index in range(11)]
        for data in (values, values.astype("float32")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                whole = normalization.fit_robust_statistics(data, names)
                with mock.patch.object(normalization, "_FIT_BLOCK_ELEMENTS", 503 * 3):
                    blocked = normalization.fit_robust_statistics(data, names)
            for field in ("median", "mad", "scale", "finite_count", "zero_mad_mask"):
                with self.subTest(dtype=str(data.dtype), field=field):
                    np.testing.assert_array_equal(
                        getattr(whole, field), getattr(blocked, field)
                    )
            self.assertTrue(blocked.zero_mad_mask[4])


if __name__ == "__main__":
    unittest.main()
