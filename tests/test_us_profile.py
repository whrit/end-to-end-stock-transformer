"""US-equities profile: codes, US market-state families, configs, and R2 wiring."""

import contextlib
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

from market_guided_stock_alpha.data.contracts import (
    DataQualityError,
    normalize_security_code,
)
from market_guided_stock_alpha.dataset import FeatureLayout, feature_names
from market_guided_stock_alpha.derived import (
    build_us_market63_artifact,
    build_us_market84_artifact,
    read_us_market63_artifact,
    read_us_market84_artifact,
    us_market63_spec,
    us_market84_spec,
)
from market_guided_stock_alpha.features.us_market import (
    US_MARKET63_INDEX_CODES,
    US_MARKET84_INDEX_CODES,
    USMarketGenerator,
    USMarketSpecError,
)
from market_guided_stock_alpha.rolling import (
    load_experiment_config,
    validate_experiment_config,
)
import market_guided_stock_alpha.rolling.configuration as rolling_configuration


US_CONFIGS = ROOT / "configs" / "us" / "stage1"
US_MARKET63_CONFIG = US_CONFIGS / "master_d512_us_market63.json"
US_MARKET84_CONFIG = US_CONFIGS / "master_d512_us_market84.json"
REMOTE_RUNS = "r2://stock-transformer-data/us/v1/runs"


def daily_index(codes, periods=65):
    rows = []
    for block, code in enumerate(codes, start=1):
        for offset, day in enumerate(pd.bdate_range("2024-01-02", periods=periods)):
            close = 100.0 * block + offset
            rows.append(
                {
                    "trade_date": day,
                    "index_code": code,
                    "open": close - 0.5,
                    "close": close,
                    "amount": 1e9 * block + offset * 1e6,
                }
            )
    return pd.DataFrame(rows)


class SecurityCodeTest(unittest.TestCase):
    def test_accepted_formats_are_uppercased(self):
        cases = {
            "000001.sz": "000001.SZ",
            "600000.SH": "600000.SH",
            "830799.BJ": "830799.BJ",
            "bbg001s5s399": "BBG001S5S399",
            "spy.us": "SPY.US",
            "BRK.B.US": "BRK.B.US",
            "GOOGL.US": "GOOGL.US",
            "A.US": "A.US",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_security_code(raw), expected)

    def test_ambiguous_or_malformed_codes_are_rejected(self):
        for raw in (
            "AAPL",
            "000001",
            "BBG001S5S39",
            "BBG001S5S3990",
            "AAPL.UK",
            "1SPY.US",
            "TOOLONG.US",
            "BRK.BB.US",
            "SPY-US",
            "",
            None,
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(
                    DataQualityError, "is empty|exchange guessing is disabled"
                ):
                    normalize_security_code(raw)

    def test_label_reader_accepts_figi_keyed_artifacts(self):
        from market_guided_stock_alpha.labels import DerivedLabelReader

        frame = pd.DataFrame(
            {"relative_excess_return_5d": [0.01, -0.02]},
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2024-01-02"), "BBG001S5N8V8"),
                    (pd.Timestamp("2024-01-02"), "bbg001s5s399"),
                ],
                names=["trade_date", "sec_code"],
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.pkl"
            frame.to_pickle(path)
            labels, _metadata = DerivedLabelReader(path, "relative_excess_return_5d").load()
        codes = sorted(labels["sec_code"])
        self.assertEqual(codes, ["BBG001S5N8V8", "BBG001S5S399"])


class USMarketFeatureTest(unittest.TestCase):
    def test_us_market_feature_counts_order_and_prefixes(self):
        us63 = feature_names("us_market63")
        us84 = feature_names("us_market84")
        self.assertEqual((len(us63), len(set(us63))), (63, 63))
        self.assertEqual((len(us84), len(set(us84))), (84, 84))
        self.assertEqual(us63[0], "MKT_SPY_RET1")
        self.assertEqual(us63[21], "MKT_IJH_RET1")
        self.assertEqual(us63[42], "MKT_IJR_RET1")
        self.assertEqual(us63[-1], "MKT_IJR_AMOUNT_STD_RATIO60")
        self.assertEqual(us84[:63], us63)
        self.assertEqual(us84[63], "MKT_IWM_RET1")
        self.assertEqual(us84[-1], "MKT_IWM_AMOUNT_STD_RATIO60")
        for block, prefix in enumerate(
            ("MKT_SPY_", "MKT_IJH_", "MKT_IJR_", "MKT_IWM_")
        ):
            names = us84[block * 21:(block + 1) * 21]
            self.assertTrue(all(name.startswith(prefix) for name in names))
        china = feature_names("market63")
        self.assertEqual(
            [name.split("_", 2)[2] for name in us63],
            [name.split("_", 2)[2] for name in china],
        )

    def test_us_market_specs_are_strict(self):
        self.assertEqual(us_market63_spec()["index_codes"], list(US_MARKET63_INDEX_CODES))
        self.assertEqual(us_market84_spec()["index_codes"], list(US_MARKET84_INDEX_CODES))
        self.assertEqual(us_market84_spec()["total_feature_count"], 84)
        broken = dict(us_market63_spec())
        broken["universes"] = ["ijh", "spy", "ijr"]
        with self.assertRaisesRegex(USMarketSpecError, "universe order"):
            USMarketGenerator(broken)
        broken = dict(us_market84_spec())
        broken["total_feature_count"] = 63
        with self.assertRaisesRegex(USMarketSpecError, "feature counts"):
            USMarketGenerator(broken)
        broken = dict(us_market63_spec())
        broken["name"] = "market63"
        with self.assertRaisesRegex(USMarketSpecError, "name"):
            USMarketGenerator(broken)

    def test_us_market_build_read_round_trips(self):
        cases = (
            ("us_market63", US_MARKET63_INDEX_CODES,
             build_us_market63_artifact, read_us_market63_artifact, 63),
            ("us_market84", US_MARKET84_INDEX_CODES,
             build_us_market84_artifact, read_us_market84_artifact, 84),
        )
        with tempfile.TemporaryDirectory() as directory:
            for family, codes, builder, reader, count in cases:
                with self.subTest(family=family):
                    path = Path(directory) / family
                    builder(daily_index(codes), path, {"test": "synthetic"})
                    artifact = reader(path, mmap_mode=None)
                    self.assertEqual(artifact.values.shape, (65, count))
                    self.assertEqual(
                        tuple(artifact.feature_names), feature_names(family)
                    )
                    self.assertTrue(artifact.source_observed.all())
                    self.assertFalse(np.isinf(artifact.values).any())
                    self.assertEqual(artifact.manifest["family"], family)
                    self.assertEqual(
                        artifact.manifest["provenance"]["test"], "synthetic"
                    )
            with self.assertRaisesRegex(ValueError, "artifact family does not match"):
                read_us_market63_artifact(Path(directory) / "us_market84")

    def test_us_market_rejects_unexpected_index_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unexpected index codes: IWM.US"):
                build_us_market63_artifact(
                    daily_index(US_MARKET84_INDEX_CODES),
                    Path(directory) / "us_market63",
                )


class USConfigTest(unittest.TestCase):
    def test_us_configs_validate_with_us_layouts(self):
        for path, family in (
            (US_MARKET63_CONFIG, "us_market63"),
            (US_MARKET84_CONFIG, "us_market84"),
        ):
            with self.subTest(path=path.name):
                document, validated = load_experiment_config(path)
                self.assertEqual(document["universe"], "dynamic_us_common_stocks")
                self.assertEqual(document["label_contract"]["benchmark"], "SPY.US")
                layout = validated["layout"]
                self.assertEqual(layout, FeatureLayout.stage1(family))
                self.assertEqual(layout.market_family, family)
                self.assertEqual(layout.stock_dim, 158)
                self.assertEqual(layout.market_dim, 21 * (4 if family.endswith("84") else 3))
        self.assertEqual(FeatureLayout.stage1().market_family, "market63")

    def test_universe_layout_and_label_mismatches_are_rejected(self):
        us_document, _validated = load_experiment_config(US_MARKET63_CONFIG)
        china_document, _validated = load_experiment_config(
            ROOT / "configs" / "stage1" / "master_d512_market63.json"
        )
        china_with_us_layout = dict(china_document)
        china_with_us_layout["feature_layout"] = "us_stage1_market63"
        with self.assertRaisesRegex(ValueError, "does not match universe dynamic_all_a_shares"):
            validate_experiment_config(china_with_us_layout)
        us_with_china_layout = dict(us_document)
        us_with_china_layout["feature_layout"] = "stage1_market63"
        with self.assertRaisesRegex(ValueError, "does not match universe dynamic_us_common_stocks"):
            validate_experiment_config(us_with_china_layout)
        us_with_china_label = dict(us_document)
        us_with_china_label["label_contract"] = china_document["label_contract"]
        with self.assertRaisesRegex(ValueError, "SPY-relative"):
            validate_experiment_config(us_with_china_label)
        unknown_universe = dict(us_document)
        unknown_universe["universe"] = "dynamic_everything"
        with self.assertRaisesRegex(ValueError, "universe must be one of"):
            validate_experiment_config(unknown_universe)


class USRemoteRunTest(unittest.TestCase):
    def _run(self, resume=False, materialize_run=None, run_effect=None):
        """Run the US config with storage, readers, and runner mocked out."""
        state = {"materialized": [], "cache": None}

        def materialize(value, base, cache):
            state["cache"] = Path(cache)
            state["materialized"].append(str(value))
            if str(value).startswith(REMOTE_RUNS + "/"):
                if materialize_run is None:
                    raise FileNotFoundError(value)
                return materialize_run(Path(cache))
            return Path(cache) / "objects" / str(value)[len("r2://"):]

        with contextlib.ExitStack() as stack:
            patch = lambda name, **kwargs: stack.enter_context(
                mock.patch.object(rolling_configuration, name, **kwargs)
            )
            storage = patch("storage")
            storage.is_remote.side_effect = lambda value: str(value).startswith("r2://")
            storage.materialize.side_effect = materialize
            storage.publish.side_effect = lambda local, remote, overwrite: self.assertTrue(
                Path(local).is_dir()
            )
            patch("read_table", return_value=mock.sentinel.table)
            patch("TradingCalendar", return_value=mock.sentinel.calendar)
            label_reader = patch("DerivedLabelReader", return_value=mock.sentinel.label)
            patch("read_alpha158_artifact", return_value=mock.sentinel.alpha158)
            market_reader = patch(
                "read_us_market63_artifact", return_value=mock.sentinel.market
            )
            patch("generate_rolling_targets", return_value=())
            runner = patch("RollingRunner")

            def run(run_config, *_args):
                root = Path(runner.call_args.args[0])
                (root / run_config.experiment_name).mkdir(parents=True, exist_ok=True)
                if run_effect is not None:
                    raise run_effect
                return {"ok": True}

            runner.return_value.run.side_effect = run
            error = None
            result = None
            try:
                result = rolling_configuration.run_from_json_config(
                    US_MARKET63_CONFIG, resume_override=resume
                )
            except Exception as caught:
                error = caught
        return {
            "result": result,
            "error": error,
            "state": state,
            "storage": storage,
            "runner": runner,
            "label_reader": label_reader,
            "market_reader": market_reader,
        }

    def test_r2_data_paths_are_materialized_and_output_is_published(self):
        outcome = self._run()
        self.assertIsNone(outcome["error"])
        self.assertEqual(
            outcome["result"],
            {"ok": True, "run_directory": REMOTE_RUNS + "/us_master_d512_us_market63"},
        )
        document, _validated = load_experiment_config(US_MARKET63_CONFIG)
        expected = sorted(
            value for name, value in document["data"].items() if name != "label_column"
        )
        state = outcome["state"]
        self.assertEqual(sorted(state["materialized"]), expected)
        cache = state["cache"]
        for call in outcome["storage"].materialize.call_args_list:
            self.assertEqual(call.args[1], US_MARKET63_CONFIG.parent)
        objects = cache / "objects" / "stock-transformer-data" / "us" / "v1" / "derived"
        label_path, column, manifest_path = outcome["label_reader"].call_args.args
        self.assertEqual(label_path, objects / "relative_excess_return_5d.pkl")
        self.assertEqual(column, "relative_excess_return_5d")
        self.assertEqual(manifest_path, objects / "relative_excess_return_5d.manifest.json")
        outcome["market_reader"].assert_called_once_with(objects / "us_market63")
        local_root = Path(outcome["runner"].call_args.args[0])
        self.assertTrue(cache in local_root.parents)
        outcome["storage"].publish.assert_called_once_with(
            local_root, REMOTE_RUNS, overwrite=True
        )
        self.assertFalse(cache.exists())

    def test_resume_downloads_existing_remote_run_first(self):
        def existing(cache):
            path = cache / "runs_download" / "us_master_d512_us_market63"
            path.mkdir(parents=True)
            return path

        outcome = self._run(resume=True, materialize_run=existing)
        self.assertIsNone(outcome["error"])
        cache = outcome["state"]["cache"]
        self.assertIn(
            REMOTE_RUNS + "/us_master_d512_us_market63", outcome["state"]["materialized"]
        )
        local_root = cache / "runs_download"
        self.assertEqual(Path(outcome["runner"].call_args.args[0]), local_root)
        self.assertTrue(outcome["runner"].return_value.run.call_args.args[0].resume)
        outcome["storage"].publish.assert_called_once_with(
            local_root, REMOTE_RUNS, overwrite=True
        )

    def test_resume_without_remote_run_starts_empty(self):
        outcome = self._run(resume=True)
        self.assertIsNone(outcome["error"])
        local_root = Path(outcome["runner"].call_args.args[0])
        self.assertEqual(local_root, outcome["state"]["cache"] / "output")
        outcome["storage"].publish.assert_called_once_with(
            local_root, REMOTE_RUNS, overwrite=True
        )

    def test_failed_run_still_publishes_partial_output(self):
        outcome = self._run(run_effect=RuntimeError("target failed"))
        self.assertIsInstance(outcome["error"], RuntimeError)
        local_root = Path(outcome["runner"].call_args.args[0])
        outcome["storage"].publish.assert_called_once_with(
            local_root, REMOTE_RUNS, overwrite=True
        )


if __name__ == "__main__":
    unittest.main()
