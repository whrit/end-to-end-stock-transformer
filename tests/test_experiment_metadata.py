"""Historical experiment and result-source registry contracts."""

import csv
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_COLUMNS = (
    "public_family",
    "public_display_name",
    "historical_project",
    "historical_config",
    "historical_execution_id",
    "model_type",
    "feature_layout",
    "market_state",
    "d_model",
    "temporal_heads",
    "spatial_heads",
    "beta",
    "dropout",
    "learning_rate",
    "loss",
    "rolling_frequency",
    "initialization",
    "seed",
    "status",
    "result_location",
    "notes",
    "recovery_status",
    "result_availability",
    "canonical_result_location",
    "recovery_evidence",
)
RESULT_COLUMNS = (
    "public_display_name",
    "historical_execution_id",
    "project",
    "expected_result_type",
    "historical_local_or_cloud_path",
    "targets",
    "seeds",
    "metrics_available",
    "portfolio_backtest_available",
    "needs_cloud_export",
    "notes",
    "recovery_status",
    "result_availability",
    "canonical_result_location",
    "recovery_evidence",
)
STATUSES = {
    "main",
    "baseline",
    "ablation",
    "secondary",
    "exploratory",
    "screening",
    "deprecated",
}


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


class ExperimentMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historical_header, cls.historical = read_csv(
            ROOT / "results/metadata/historical_experiments.csv"
        )
        cls.result_header, cls.results = read_csv(
            ROOT / "results/metadata/result_sources.csv"
        )
        cls.public_configs = []
        for path in sorted((ROOT / "configs").rglob("*.json")):
            cls.public_configs.append(json.loads(path.read_text(encoding="utf-8")))

    def test_schema_status_and_sanitized_paths_are_stable(self):
        self.assertEqual(self.historical_header, HISTORICAL_COLUMNS)
        self.assertEqual(self.result_header, RESULT_COLUMNS)
        self.assertGreaterEqual(len(self.historical), 90)
        self.assertEqual(len(self.results), len(self.historical))
        self.assertEqual({row["status"] for row in self.historical} - STATUSES, set())
        serialized = json.dumps(
            (self.historical, self.results), ensure_ascii=False
        ).replace("\\", "/")
        self.assertNotIn("/mnt/data01", serialized)
        self.assertNotIn("intern001", serialized)
        self.assertNotIn("Dual-Frequency MASTER / Development Tuning", serialized)
        self.assertNotIn("Dual-Frequency Auxiliary Ablation /", serialized)

    def test_historical_paths_are_relative_and_row_key_is_explicit(self):
        for row in self.historical:
            config = row["historical_config"]
            if config:
                self.assertFalse(Path(config).is_absolute(), config)
                self.assertNotIn("..", Path(config).parts)
                self.assertTrue(
                    config.startswith(("03_master_replication/", "04/")), config
                )
        # historical_execution_id is provenance, not a primary key: plan matrix
        # rows may lack a materialized ID and multiple rows may share one seed.
        keys = [
            (
                row["historical_project"],
                row["historical_config"],
                row["historical_execution_id"],
                row["d_model"],
                row["temporal_heads"],
                row["spatial_heads"],
                row["beta"],
                row["dropout"],
                row["learning_rate"],
                row["loss"],
                row["rolling_frequency"],
                row["initialization"],
                row["seed"],
                row["notes"],
            )
            for row in self.historical
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_public_names_are_not_historical_execution_ids(self):
        forbidden = re.compile(r"exp\d+|seed\d+|ramcache|gsep|gjoint", re.I)
        for row in self.historical:
            self.assertTrue(row["public_display_name"])
            self.assertIsNone(forbidden.search(row["public_display_name"]))
            if row["historical_execution_id"]:
                self.assertNotEqual(
                    row["public_display_name"], row["historical_execution_id"]
                )

    def test_public_config_curation_contract(self):
        names = {config["experiment_name"] for config in self.public_configs}
        self.assertEqual(len(names), 22)
        self.assertIn("intraday_price_volume_master_d512_market63", names)
        self.assertTrue(
            {
                "master_d128_market63",
                "master_d512_market63",
                "lightgbm_market63",
                "gru_market63",
                "transformer_market63",
            }.issubset(names)
        )
        self.assertTrue(
            {
                "ablation_no_market",
                "ablation_no_intra",
                "ablation_no_inter",
                "ablation_no_temporal",
            }.issubset(names)
        )
        self.assertTrue({"loss_ccc", "loss_listfold", "loss_rpl", "loss_rsr"}.issubset(names))
        self.assertNotIn("loss_mse", names)
        self.assertEqual(
            {name for name in names if name.startswith("attention_")},
            {"attention_t8_s2"},
        )
        deprecated_auxiliary = {
            "daily_market_only",
            "fixed_lambda_one",
            "daily_branch_only",
            "intraday_branch_only",
            "uniform_gate",
        }
        self.assertTrue(names.isdisjoint(deprecated_auxiliary))
        self.assertNotIn("monthly", {c["rolling"]["frequency"] for c in self.public_configs})

    def test_result_sources_match_recovery_availability(self):
        historical_pairs = {
            (
                row["public_display_name"],
                row["historical_execution_id"],
                row["historical_project"],
            )
            for row in self.historical
        }
        for row in self.results:
            self.assertIn(
                (
                    row["public_display_name"],
                    row["historical_execution_id"],
                    row["project"],
                ),
                historical_pairs,
            )
            recovered = row["recovery_status"].startswith("recovered")
            self.assertEqual(row["result_availability"], "available" if recovered else "missing")
            self.assertEqual(row["needs_cloud_export"].lower(), "false" if recovered else "true")
            if recovered:
                self.assertTrue(row["canonical_result_location"])
                self.assertIn("recovered", row["metrics_available"])
            else:
                self.assertFalse(row["canonical_result_location"])
                self.assertIn("not recovered", row["metrics_available"])


if __name__ == "__main__":
    unittest.main()
