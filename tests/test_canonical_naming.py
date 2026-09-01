"""Pre-release contracts for canonical feature-family naming."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from market_guided_stock_alpha.dataset import FeatureLayout, feature_names


OLD_PUBLIC_NAMES = (
    "intraday_v3",
    "Intraday V3",
    "IntradayV3",
    "intraday_market192",
    "IntradayMarket192",
)


def _csv_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CanonicalNamingTest(unittest.TestCase):
    def test_public_runtime_configs_and_docs_do_not_expose_old_names(self):
        paths = [ROOT / "README.md", ROOT / "README.zh-CN.md"]
        paths.extend(sorted((ROOT / "docs").glob("*.md")))
        paths.extend(sorted((ROOT / "src").rglob("*.py")))
        paths.extend(sorted((ROOT / "scripts").glob("*.py")))
        paths.extend(sorted((ROOT / "configs").rglob("*.json")))
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                text = path.read_text(encoding="utf-8")
                for old_name in OLD_PUBLIC_NAMES:
                    self.assertNotIn(old_name, text)

        source_names = [path.name for path in (ROOT / "src").rglob("*.py")]
        config_names = [path.name for path in (ROOT / "configs").rglob("*.json")]
        self.assertFalse(any("_v3" in name for name in source_names))
        self.assertFalse(any("intraday_market192" in name for name in source_names))
        self.assertFalse(any("intraday_v3" in name for name in config_names))

    def test_builder_cli_exposes_only_canonical_family_keys(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/build_intraday_features.py"), "--help"],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("intraday_price_volume", completed.stdout)
        self.assertIn("intraday_market", completed.stdout)
        self.assertNotIn("intraday_v3", completed.stdout)
        self.assertNotIn("intraday_market192", completed.stdout)

    def test_public_configs_and_readme_quick_start_use_canonical_names(self):
        configs = sorted((ROOT / "configs").rglob("*.json"))
        self.assertTrue(configs)
        for path in configs:
            document = json.loads(path.read_text(encoding="utf-8"))
            serialized = json.dumps(document)
            self.assertNotIn("intraday_v3", serialized)
            self.assertNotIn("intraday_market192", serialized)
        main_config = (
            ROOT
            / "configs/stage2/intraday_price_volume_master_d512_market63.json"
        )
        self.assertTrue(main_config.is_file())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("build_intraday_features.py intraday_price_volume", readme)
        self.assertIn(main_config.relative_to(ROOT).as_posix(), readme)

    def test_feature_orders_and_dimensions_remain_canonical(self):
        intraday_names = feature_names("intraday_price_volume")
        market_names = feature_names("intraday_market")
        self.assertEqual(len(intraday_names), 248)
        self.assertEqual(len(set(intraday_names)), 248)
        self.assertEqual(len(market_names), 192)
        self.assertEqual(len(set(market_names)), 192)

        stage2 = FeatureLayout.stage2("market63")
        self.assertEqual(stage2.stock_dim, 406)
        self.assertEqual(stage2.market_dim, 63)
        self.assertEqual(stage2.conceptual_input_dim, 469)
        self.assertEqual(FeatureLayout.dual_frequency().intraday_market_dim, 192)

    def test_results_use_new_public_names_but_keep_historical_identity(self):
        historical_identity_found = False
        for path in sorted((ROOT / "results").rglob("*.csv")):
            rows = _csv_rows(path)
            for row in rows:
                for field in ("public_display_name", "feature_layout"):
                    value = row.get(field, "")
                    self.assertNotIn("intraday_v3", value)
                    self.assertNotIn("IntradayV3", value)
                historical = "|".join(
                    row.get(field, "")
                    for field in (
                        "historical_setting_id",
                        "historical_experiment_id",
                        "historical_execution_id",
                        "result_source",
                        "provenance_source",
                    )
                )
                historical_identity_found |= "intraday_v3" in historical
        self.assertTrue(historical_identity_found)

        metadata = _csv_rows(ROOT / "results/metadata/historical_experiments.csv")
        self.assertTrue(
            any("intraday_v3" in row["historical_execution_id"] for row in metadata)
        )
        self.assertFalse(
            any("Intraday V3" in row["public_display_name"] for row in metadata)
        )

        manifests = sorted(
            (ROOT / "results/provenance/project04_experiment_manifests").glob(
                "*intraday_v3*.json"
            )
        )
        self.assertTrue(manifests)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertIn("intraday_v3", manifest["namespace"])
        self.assertRegex(manifest["experiment_config_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
