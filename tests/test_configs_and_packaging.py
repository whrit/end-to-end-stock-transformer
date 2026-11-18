"""Experiment config, packaging, and LightGBM rolling contracts."""

import gc
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.data import TradingCalendar
from market_guided_stock_alpha.dataset import FeatureLayout, feature_names
from market_guided_stock_alpha.derived import (
    read_alpha158_artifact,
    read_market63_artifact,
    write_feature_artifact,
)
from market_guided_stock_alpha.labels import DEFAULT_VALUE_COLUMN, DerivedLabelReader
from market_guided_stock_alpha.rolling import (
    InitializationMode,
    RollingDataBundle,
    RollingRunConfig,
    RollingRunner,
    RollingTarget,
    load_experiment_config,
    validate_experiment_config,
)
import market_guided_stock_alpha.rolling.configuration as rolling_configuration
from market_guided_stock_alpha.training import LightGBMTrainingConfig


class ConfigAndPackagingTest(unittest.TestCase):
    def test_every_config_parses_and_has_unique_experiment_name(self):
        files = sorted((ROOT / "configs").rglob("*.json"))
        self.assertTrue(files)
        names = set()
        for path in files:
            with self.subTest(path=str(path.relative_to(ROOT))):
                document, _validated = load_experiment_config(path)
                self.assertNotIn(document["experiment_name"], names)
                names.add(document["experiment_name"])

    def test_shipped_config_paths_resolve_to_repository_data_and_runs(self):
        data_root = (ROOT / "data").resolve()
        expected_manifest = (
            data_root / "derived" / "relative_excess_return_5d.manifest.json"
        )
        for path in sorted((ROOT / "configs").rglob("*.json")):
            with self.subTest(path=str(path.relative_to(ROOT))):
                document, _validated = load_experiment_config(path)
                data = dict(document["data"])
                self.assertIn("label_manifest", data)
                for name, value in data.items():
                    if name == "label_column":
                        continue
                    relative = Path(str(value))
                    self.assertFalse(relative.is_absolute())
                    resolved = (path.parent / relative).resolve()
                    self.assertTrue(
                        resolved == data_root or data_root in resolved.parents
                    )
                self.assertEqual(
                    (path.parent / data["label_manifest"]).resolve(),
                    expected_manifest,
                )
                self.assertEqual(
                    (path.parent / document["output_root"]).resolve(),
                    (ROOT / "runs").resolve(),
                )

    def test_user_config_accepts_absolute_data_and_output_paths(self):
        path = ROOT / "configs" / "stage1" / "master_d128_market63.json"
        document, _validated = load_experiment_config(path)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = dict(document["data"])
            for name in data:
                if name != "label_column":
                    data[name] = str(root / name)
            document["data"] = data
            document["output_root"] = str(root / "runs")
            validated = validate_experiment_config(document)
        self.assertEqual(validated["layout"], FeatureLayout.stage1())

    def test_label_manifest_is_wired_into_derived_label_reader(self):
        path = ROOT / "configs" / "stage1" / "master_d128_market63.json"
        with mock.patch.object(
            rolling_configuration, "read_table", return_value=mock.sentinel.table
        ), mock.patch.object(
            rolling_configuration,
            "TradingCalendar",
            return_value=mock.sentinel.calendar,
        ), mock.patch.object(
            rolling_configuration,
            "DerivedLabelReader",
            return_value=mock.sentinel.label_reader,
        ) as label_reader, mock.patch.object(
            rolling_configuration,
            "read_alpha158_artifact",
            return_value=mock.sentinel.alpha158,
        ), mock.patch.object(
            rolling_configuration,
            "read_market63_artifact",
            return_value=mock.sentinel.market,
        ), mock.patch.object(
            rolling_configuration, "generate_rolling_targets", return_value=()
        ), mock.patch.object(
            rolling_configuration, "RollingRunner"
        ) as runner:
            runner.return_value.run.return_value = {}
            rolling_configuration.run_from_json_config(path)

        label_path, column, manifest_path = label_reader.call_args.args
        self.assertEqual(
            label_path.resolve(),
            (ROOT / "data" / "derived" / "relative_excess_return_5d.pkl").resolve(),
        )
        self.assertEqual(column, DEFAULT_VALUE_COLUMN)
        self.assertEqual(
            manifest_path.resolve(),
            (
                ROOT
                / "data"
                / "derived"
                / "relative_excess_return_5d.manifest.json"
            ).resolve(),
        )

    def test_python_38_syntax_and_packaging_metadata(self):
        import ast

        paths = list((ROOT / "src").rglob("*.py")) + list(
            (ROOT / "scripts").glob("*.py")
        )
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 8),
                )
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.14"', pyproject)
        self.assertIn('lightgbm = ["lightgbm"]', pyproject)
        self.assertIn('parquet = ["pyarrow"]', pyproject)
        for path in (ROOT / "scripts").glob("*.py"):
            self.assertNotIn("sys.path.insert", path.read_text(encoding="utf-8"))


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "optional lightgbm is not installed")
class LightGBMRollingTest(unittest.TestCase):
    def test_cold_start_target_to_prediction_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.bdate_range("2024-01-02", periods=40)
            calendar = TradingCalendar(pd.DataFrame({"calendar_date": dates, "is_open": 1}))
            codes = tuple("{:06d}.SZ".format(value) for value in range(1, 5))
            alpha_names = feature_names("alpha158")
            market_names = feature_names("market63")
            alpha_values = np.empty(
                (len(dates), len(codes), len(alpha_names)), dtype="float32"
            )
            for date_position in range(len(dates)):
                for code_position in range(len(codes)):
                    alpha_values[date_position, code_position] = (
                        date_position * 0.1 + code_position
                    )
            market_values = np.repeat(
                np.arange(len(dates), dtype="float32")[:, None],
                len(market_names),
                axis=1,
            )
            write_feature_artifact(
                root / "alpha158",
                "alpha158",
                alpha_values,
                dates,
                alpha_names,
                security_codes=codes,
            )
            write_feature_artifact(
                root / "market63", "market63", market_values, dates, market_names
            )
            alpha = read_alpha158_artifact(root / "alpha158")
            market = read_market63_artifact(root / "market63")
            membership_rows = []
            label_rows = []
            for day in dates[7:]:
                for position, code in enumerate(codes):
                    membership_rows.append({"trade_date": day, "sec_code": code})
                    label_rows.append(
                        {
                            "trade_date": day,
                            "sec_code": code,
                            DEFAULT_VALUE_COLUMN: float(position),
                        }
                    )
            membership = pd.DataFrame(membership_rows)
            label_path = root / "labels.pkl"
            pd.DataFrame(label_rows).to_pickle(label_path)
            target = RollingTarget(
                "T1",
                dates[7],
                dates[16],
                dates[18],
                dates[23],
                dates[25],
                dates[27],
                "semiannual",
                dates[7:17],
                dates[18:24],
                dates[25:28],
                pd.DatetimeIndex([]),
                pd.DatetimeIndex([]),
            )
            result = RollingRunner(root / "runs", "cpu").run_lightgbm(
                RollingRunConfig(
                    "lightgbm_smoke",
                    seeds=(42,),
                    initialization=InitializationMode.COLD,
                ),
                (target,),
                RollingDataBundle(
                    calendar,
                    membership,
                    DerivedLabelReader(label_path, DEFAULT_VALUE_COLUMN),
                    alpha,
                    market,
                ),
                FeatureLayout.stage1(),
                LightGBMTrainingConfig(
                    parameters={
                        "objective": "regression",
                        "metric": "None",
                        "learning_rate": 0.1,
                        "num_leaves": 4,
                        "min_data_in_leaf": 1,
                        "verbosity": -1,
                        "deterministic": True,
                        "force_col_wise": True,
                    },
                    num_boost_round=5,
                    early_stopping_rounds=2,
                ),
            )
            target_result = result["individual_seeds"][42]["targets"][0]
            self.assertEqual(target_result["initialization"], "cold")
            self.assertGreater(target_result["evaluation_observation_count"], 0)
            self.assertTrue(
                (root / "runs/lightgbm_smoke/seed_42/target_T1/model.txt").is_file()
            )
            self.assertTrue(
                (
                    root
                    / "runs/lightgbm_smoke/seed_42/target_T1/predictions.pkl"
                ).is_file()
            )
            del result, alpha, market
            gc.collect()


if __name__ == "__main__":
    unittest.main()
