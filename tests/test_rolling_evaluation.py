"""Canonical rolling, prediction, evaluation, and ensemble contracts."""

import json
import math
import gc
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


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
from market_guided_stock_alpha.evaluation import (
    aggregate_overall,
    evaluate_target,
    raw_equal_weight_ensemble,
    summarize_seed_robustness,
    validate_prediction_frame,
)
from market_guided_stock_alpha.labels import (
    DEFAULT_VALUE_COLUMN,
    PURGE_DEPENDENCY_OFFSET_TRADING_DAYS,
    DerivedLabelReader,
)
from market_guided_stock_alpha.rolling import (
    InitializationMode,
    RollingDataBundle,
    RollingRunConfig,
    RollingRunner,
    RollingTarget,
    generate_rolling_targets,
    initialize_target_model,
    read_prediction_artifact,
    write_prediction_artifact,
)
from market_guided_stock_alpha.training import (
    CheckpointContract,
    CheckpointManager,
    EarlyStoppingConfig,
    OptimizerConfig,
    TrainingConfig,
)


def calendar_frame(start="2020-01-01", end="2025-12-31"):
    return pd.DataFrame(
        {"calendar_date": pd.date_range(start, end, freq="D")}
    ).assign(is_open=lambda value: value["calendar_date"].dt.dayofweek.lt(5).astype(int))


def prediction_frame(target_id="2024H1", prediction=None):
    count = 20
    labels = np.arange(count, dtype="float64")
    return pd.DataFrame(
        {
            "target_id": target_id,
            "trade_date": pd.Timestamp("2024-01-02"),
            "sec_code": ["{:06d}.SZ".format(i + 1) for i in range(count)],
            "prediction": labels if prediction is None else prediction,
            "label_raw": labels,
            "label": labels,
            "label_mask": True,
        }
    )


class TargetGenerationTest(unittest.TestCase):
    def test_all_frequencies_use_trading_boundaries_and_t6_purge(self):
        calendar = TradingCalendar(calendar_frame())
        expected_counts = {"semiannual": 2, "quarterly": 4, "monthly": 12}
        for frequency, expected in expected_counts.items():
            with self.subTest(frequency=frequency):
                targets = generate_rolling_targets(
                    calendar, frequency, "2020-01-02", "2023-01-01", "2023-12-31"
                )
                self.assertEqual(len(targets), expected)
                for target in targets:
                    self.assertTrue(calendar.is_trading_day(target.train_start))
                    self.assertTrue(calendar.is_trading_day(target.valid_start))
                    self.assertTrue(calendar.is_trading_day(target.test_start))
                    self.assertEqual(
                        target.purge_dependency,
                        PURGE_DEPENDENCY_OFFSET_TRADING_DAYS,
                    )
                    self.assertTrue(
                        all(
                            calendar.next_trading_date(day, 6) < target.valid_start
                            for day in target.train_dates
                        )
                    )
                    self.assertTrue(
                        all(
                            calendar.next_trading_date(day, 6) < target.test_start
                            for day in target.valid_dates
                        )
                    )

    def test_partial_semiannual_identity_is_not_hard_coded(self):
        targets = generate_rolling_targets(
            TradingCalendar(calendar_frame()),
            "semiannual",
            "2020-01-02",
            "2024-01-01",
            "2024-03-15",
        )
        self.assertEqual(targets[0].target_id, "2024H1_partial")


class MetricAndEnsembleTest(unittest.TestCase):
    def test_exact_daily_metrics_and_percentile_portfolios(self):
        summary, daily = evaluate_target(prediction_frame())
        self.assertAlmostEqual(summary["ic"], 1.0)
        self.assertAlmostEqual(summary["rank_ic"], 1.0)
        self.assertAlmostEqual(summary["top5"], 19.0)
        self.assertAlmostEqual(summary["top10"], 18.5)
        self.assertAlmostEqual(summary["mse"], 0.0)
        self.assertAlmostEqual(summary["r2"], 1.0)
        self.assertEqual(summary["ic_valid_day_count"], 1)
        self.assertEqual(int(daily.iloc[0]["raw_valid_count"]), 20)

    def test_invalid_correlation_day_is_skipped_but_top_group_is_defined(self):
        frame = prediction_frame(prediction=np.zeros(20))
        summary, _daily = evaluate_target(frame)
        self.assertIsNone(summary["ic"])
        self.assertIsNone(summary["rank_ic"])
        self.assertEqual(summary["top5_valid_day_count"], 1)

    def test_overall_weights_valid_days_and_pools_mse_and_r2(self):
        rows = [
            {
                "ic_sum": 2.0,
                "ic_valid_day_count": 2,
                "rank_ic_sum": 1.0,
                "rank_ic_valid_day_count": 2,
                "top5_sum": 4.0,
                "top5_valid_day_count": 2,
                "top10_sum": 6.0,
                "top10_valid_day_count": 2,
                "evaluation_observation_count": 2,
                "evaluation_sse": 2.0,
                "evaluation_label_sum": 1.0,
                "evaluation_label_sum_sq": 1.0,
            },
            {
                "ic_sum": -1.0,
                "ic_valid_day_count": 1,
                "rank_ic_sum": -0.5,
                "rank_ic_valid_day_count": 1,
                "top5_sum": 10.0,
                "top5_valid_day_count": 1,
                "top10_sum": 12.0,
                "top10_valid_day_count": 1,
                "evaluation_observation_count": 3,
                "evaluation_sse": 3.0,
                "evaluation_label_sum": 5.0,
                "evaluation_label_sum_sq": 9.0,
            },
        ]
        overall = aggregate_overall(rows)
        self.assertAlmostEqual(overall["ic"], 1.0 / 3.0)
        self.assertAlmostEqual(overall["top5"], 14.0 / 3.0)
        self.assertAlmostEqual(overall["mse"], 1.0)
        pooled_sst = 10.0 - 36.0 / 5.0
        self.assertAlmostEqual(overall["r2"], 1.0 - 5.0 / pooled_sst)
        target_r2_mean = ((1.0 - 2.0 / 0.5) + (1.0 - 3.0 / (9.0 - 25.0 / 3.0))) / 2
        self.assertNotAlmostEqual(overall["r2"], target_r2_mean)

    def test_seed_robustness_uses_sample_standard_deviation(self):
        rows = {
            42: {name: 1.0 for name in ("ic", "rank_ic", "top5", "top10", "mse", "r2")},
            43: {name: 2.0 for name in ("ic", "rank_ic", "top5", "top10", "mse", "r2")},
            44: {name: 3.0 for name in ("ic", "rank_ic", "top5", "top10", "mse", "r2")},
        }
        result = summarize_seed_robustness(rows)
        self.assertEqual(result["standard_deviation_ddof"], 1)
        self.assertAlmostEqual(result["metrics"]["ic"]["mean"], 2.0)
        self.assertAlmostEqual(result["metrics"]["ic"]["sample_std"], 1.0)

    def test_strict_raw_prediction_ensemble_and_new_evaluation(self):
        base = prediction_frame()
        labels = base["label_raw"].to_numpy()
        frames = []
        for prediction in (labels, -labels, np.roll(labels, 1)):
            current = base.copy()
            current["prediction"] = prediction
            frames.append(current.sample(frac=1.0, random_state=len(frames)))
        ensemble = raw_equal_weight_ensemble(frames)
        expected = np.mean(
            np.stack([frame.sort_values("sec_code")["prediction"] for frame in frames]),
            axis=0,
        )
        np.testing.assert_allclose(ensemble["prediction"], expected)
        ensemble_metrics, _ = evaluate_target(ensemble)
        seed_ic = [evaluate_target(frame)[0]["ic"] for frame in frames]
        self.assertFalse(math.isclose(ensemble_metrics["ic"], float(np.mean(seed_ic))))

        missing = frames[0].iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "keys differ"):
            raw_equal_weight_ensemble([missing, frames[1], frames[2]])

    def test_prediction_schema_and_pickle_manifest(self):
        with self.assertRaisesRegex(ValueError, "not unique"):
            validate_prediction_frame(pd.concat([prediction_frame(), prediction_frame()]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.pkl"
            write_prediction_artifact(path, prediction_frame(), {"checkpoint_sha256": "abc"})
            loaded, manifest = read_prediction_artifact(path)
            self.assertEqual(len(loaded), 20)
            self.assertEqual(manifest["metadata"]["checkpoint_sha256"], "abc")


class TinyRegressor(nn.Module):
    input_kind = "stock"

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = int(input_dim)
        self.linear = nn.Linear(1, 1)

    def forward(self, stock_features):
        return self.linear(stock_features[:, -1, :1]).squeeze(-1)

    def config_dict(self):
        return {"input_dim": self.input_dim, "kind": "tiny_test_model"}


def _preprocessing_contract(marker):
    return {
        "format_version": 1,
        "fit_scope": "training_dates_only",
        "feature_statistics": {"marker": marker},
    }


class InitializationAndRollingSmokeTest(unittest.TestCase):
    def test_warm_loads_only_weights_and_cold_keeps_new_initialization(self):
        layout = FeatureLayout.stage1()
        config = TrainingConfig(
            epochs=1,
            early_stopping=EarlyStoppingConfig(enabled=False, patience=1, minimum_epochs=1),
            seed=42,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = TinyRegressor(layout.stock_dim)
            with torch.no_grad():
                first.linear.weight.fill_(7.0)
                first.linear.bias.fill_(3.0)
            first_manager = CheckpointManager(
                root / "first",
                CheckpointContract.from_components(first, config, layout, _preprocessing_contract(1)),
            )
            first_manager.save(1, 1.0, first, torch.optim.Adam(first.parameters()), [], True)

            warm = TinyRegressor(layout.stock_dim)
            warm_manager = CheckpointManager(
                root / "warm",
                CheckpointContract.from_components(warm, config, layout, _preprocessing_contract(2)),
            )
            status = initialize_target_model(
                warm, InitializationMode.WARM, warm_manager, root / "first" / "best.pt"
            )
            self.assertEqual(status, "previous_target_best_weights_only")
            self.assertAlmostEqual(float(warm.linear.weight.item()), 7.0)

            cold = TinyRegressor(layout.stock_dim)
            before = cold.linear.weight.detach().clone()
            status = initialize_target_model(
                cold, InitializationMode.COLD, warm_manager, root / "first" / "best.pt"
            )
            self.assertEqual(status, "new_model_initialization")
            self.assertTrue(torch.equal(before, cold.linear.weight))

    def test_two_target_two_seed_end_to_end_smoke_and_fresh_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.bdate_range("2023-01-02", periods=62)
            calendar = TradingCalendar(
                pd.DataFrame({"calendar_date": dates, "is_open": 1})
            )
            codes = tuple("{:06d}.SZ".format(value) for value in range(1, 5))
            alpha_names = feature_names("alpha158")
            market_names = feature_names("market63")
            alpha_values = np.empty((len(dates), len(codes), len(alpha_names)), dtype="float32")
            for d in range(len(dates)):
                for s in range(len(codes)):
                    alpha_values[d, s, :] = d * 0.1 + s
            market_values = np.repeat(
                np.arange(len(dates), dtype="float32")[:, None], len(market_names), axis=1
            )
            write_feature_artifact(
                root / "alpha158", "alpha158", alpha_values, dates, alpha_names,
                security_codes=codes,
            )
            write_feature_artifact(
                root / "market63", "market63", market_values, dates, market_names
            )
            alpha = read_alpha158_artifact(root / "alpha158")
            market = read_market63_artifact(root / "market63")
            member_rows = []
            label_rows = []
            for day in dates[7:]:
                for position, code in enumerate(codes):
                    member_rows.append({"trade_date": day, "sec_code": code})
                    label_rows.append(
                        {
                            "trade_date": day,
                            "sec_code": code,
                            DEFAULT_VALUE_COLUMN: float(position - 1.5) + float(day.day) * 0.001,
                        }
                    )
            membership = pd.DataFrame(member_rows)
            label_path = root / "labels.pkl"
            pd.DataFrame(label_rows).to_pickle(label_path)
            target_specs = (
                ("T1", dates[7:18], dates[20:26], dates[28:30]),
                ("T2", dates[7:28], dates[30:36], dates[38:40]),
            )
            targets = []
            for target_id, train, valid, test in target_specs:
                targets.append(
                    RollingTarget(
                        target_id,
                        train[0], train[-1], valid[0], valid[-1], test[0], test[-1],
                        "monthly", pd.DatetimeIndex(train), pd.DatetimeIndex(valid),
                        pd.DatetimeIndex(test), pd.DatetimeIndex([]), pd.DatetimeIndex([]),
                    )
                )
            training_factory = lambda seed: TrainingConfig(
                epochs=1,
                optimizer=OptimizerConfig(learning_rate=1e-2),
                early_stopping=EarlyStoppingConfig(enabled=False, patience=1, minimum_epochs=1),
                seed=seed,
            )
            finished = []
            on_target = lambda path: finished.append(
                (path.relative_to(root / "runs").as_posix(), (path / "metrics.json").is_file(),
                 (path / "checkpoints/best.pt").is_file())
            )
            result = RollingRunner(root / "runs", "cpu", on_target=on_target).run(
                RollingRunConfig("tiny", seeds=(42, 43), initialization=InitializationMode.WARM),
                targets,
                RollingDataBundle(
                    calendar, membership, DerivedLabelReader(label_path, DEFAULT_VALUE_COLUMN),
                    alpha, market,
                ),
                FeatureLayout.stage1(),
                lambda layout, _seed: TinyRegressor(layout.stock_dim),
                training_factory,
            )
            # Each target is handed over complete, as it finishes (what remote runs upload).
            self.assertEqual(
                finished,
                [("tiny/seed_{}/target_{}".format(seed, target), True, True) for seed in (42, 43) for target in ("T1", "T2")],
            )
            self.assertIsNotNone(result["ensemble_evaluation"])
            self.assertEqual(
                result["individual_seeds"][42]["targets"][1]["initialization"],
                "previous_target_best_weights_only",
            )
            first = json.loads(
                (root / "runs/tiny/seed_42/target_T1/preprocessing.json").read_text()
            )
            second = json.loads(
                (root / "runs/tiny/seed_42/target_T2/preprocessing.json").read_text()
            )
            self.assertNotEqual(
                first["feature_statistics"]["alpha158"]["median"],
                second["feature_statistics"]["alpha158"]["median"],
            )
            self.assertTrue((root / "runs/tiny/ensemble/metrics.json").is_file())
            del result, alpha, market
            gc.collect()


if __name__ == "__main__":
    unittest.main()
