"""Public experiment config, packaging, and LightGBM rolling contracts."""

import gc
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
)
from market_guided_stock_alpha.training import LightGBMTrainingConfig


class PublicConfigTest(unittest.TestCase):
    def test_every_public_config_parses_and_obeys_public_contract(self):
        files = sorted((ROOT / "configs").rglob("*.json"))
        self.assertEqual(len(files), 22)
        names = set()
        for path in files:
            with self.subTest(path=str(path.relative_to(ROOT))):
                document, validated = load_experiment_config(path)
                self.assertNotIn(document["experiment_name"], names)
                names.add(document["experiment_name"])
                self.assertEqual(document["universe"], "dynamic_all_a_shares")
                self.assertEqual(document["label_contract"]["benchmark"], "000905.SH")
                self.assertEqual(
                    document["label_contract"]["dependency_offset_trading_days"], 6
                )
                self.assertNotIn("purge", document)
                serialized = json.dumps(document).replace("\\", "/")
                self.assertNotIn("/mnt/", serialized)
                self.assertNotIn("provided_tag", serialized)
                self.assertNotRegex(document["experiment_name"], r"exp\d+|seed\d+|ramcache")
                self.assertIn(validated["frequency"], ("semiannual", "quarterly", "monthly"))

    def test_main_stage2_dimensions_defaults_and_secondary_modes(self):
        main, validated = load_experiment_config(
            ROOT / "configs/stage2/intraday_price_volume_master_d512_market63.json"
        )
        self.assertEqual(validated["layout"].stock_dim, 406)
        self.assertEqual(validated["layout"].market_dim, 63)
        self.assertEqual(validated["seeds"], (42, 43, 44))
        self.assertEqual(validated["frequency"], "semiannual")
        self.assertEqual(validated["initialization"], InitializationMode.WARM)
        self.assertEqual(main["model"]["d_model"], 512)

        _quarterly, quarterly = load_experiment_config(
            ROOT / "configs/secondary/master_d128_market63_quarterly.json"
        )
        _cold, cold = load_experiment_config(
            ROOT / "configs/secondary/master_d128_market63_cold_start.json"
        )
        self.assertEqual(quarterly["frequency"], "quarterly")
        self.assertEqual(cold["initialization"], InitializationMode.COLD)

    def test_lightgbm_public_config_matches_formal_reference(self):
        document, validated = load_experiment_config(
            ROOT / "configs/stage1/lightgbm_market63.json"
        )
        self.assertIsInstance(validated["training"], LightGBMTrainingConfig)
        self.assertEqual(validated["training"].num_boost_round, 2000)
        self.assertEqual(validated["training"].early_stopping_rounds, 100)
        self.assertEqual(validated["seeds"], (42, 43, 44))
        self.assertEqual(document["initialization"], "cold")

    def test_python_38_syntax_and_packaging_metadata(self):
        import ast

        for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.8"', pyproject)
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
            alpha_values = np.empty((len(dates), len(codes), len(alpha_names)), dtype="float32")
            for date_position in range(len(dates)):
                for code_position in range(len(codes)):
                    alpha_values[date_position, code_position] = date_position * 0.1 + code_position
            market_values = np.repeat(
                np.arange(len(dates), dtype="float32")[:, None], len(market_names), axis=1
            )
            write_feature_artifact(
                root / "alpha158", "alpha158", alpha_values, dates, alpha_names,
                security_codes=codes,
            )
            write_feature_artifact(root / "market63", "market63", market_values, dates, market_names)
            alpha = read_alpha158_artifact(root / "alpha158")
            market = read_market63_artifact(root / "market63")
            membership_rows = []
            label_rows = []
            for day in dates[7:]:
                for position, code in enumerate(codes):
                    membership_rows.append({"trade_date": day, "sec_code": code})
                    label_rows.append(
                        {"trade_date": day, "sec_code": code, DEFAULT_VALUE_COLUMN: float(position)}
                    )
            membership = pd.DataFrame(membership_rows)
            label_path = root / "labels.pkl"
            pd.DataFrame(label_rows).to_pickle(label_path)
            target = RollingTarget(
                "T1", dates[7], dates[16], dates[18], dates[23], dates[25], dates[27],
                "semiannual", dates[7:17], dates[18:24], dates[25:28],
                pd.DatetimeIndex([]), pd.DatetimeIndex([]),
            )
            result = RollingRunner(root / "runs", "cpu").run_lightgbm(
                RollingRunConfig(
                    "lightgbm_smoke", seeds=(42,), initialization=InitializationMode.COLD
                ),
                (target,),
                RollingDataBundle(
                    calendar, membership, DerivedLabelReader(label_path, DEFAULT_VALUE_COLUMN),
                    alpha, market,
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
            self.assertTrue((root / "runs/lightgbm_smoke/seed_42/target_T1/model.txt").is_file())
            self.assertTrue((root / "runs/lightgbm_smoke/seed_42/target_T1/predictions.pkl").is_file())
            del result, alpha, market
            gc.collect()


if __name__ == "__main__":
    unittest.main()
