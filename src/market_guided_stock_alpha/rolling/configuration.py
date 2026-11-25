"""JSON configuration validation and rolling runtime wiring."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

from market_guided_stock_alpha import storage
from market_guided_stock_alpha.progress import note, status_to
from market_guided_stock_alpha.data.calendar import TradingCalendar
from market_guided_stock_alpha.data.io import read_table
from market_guided_stock_alpha.dataset.layout import FeatureLayout
from market_guided_stock_alpha.derived.readers import (
    read_alpha158_artifact,
    read_intraday_market_artifact,
    read_intraday_price_volume_artifact,
    read_market63_artifact,
    read_market84_artifact,
    read_us_market63_artifact,
    read_us_market84_artifact,
)
from market_guided_stock_alpha.labels.derived_label import DerivedLabelReader
from market_guided_stock_alpha.labels.relative_excess_return import DEFAULT_VALUE_COLUMN
from market_guided_stock_alpha.modeling.baselines import (
    GRURegressor,
    VanillaTransformerRegressor,
)
from market_guided_stock_alpha.modeling.master import MasterModel
from market_guided_stock_alpha.modeling.experimental.dual_frequency import (
    DualFrequencyMasterConfig,
    DualFrequencyMasterModel,
)
from market_guided_stock_alpha.rolling.runner import (
    RollingDataBundle,
    RollingRunConfig,
    RollingRunner,
)
from market_guided_stock_alpha.rolling.schedule import InitializationMode
from market_guided_stock_alpha.rolling.targets import generate_rolling_targets
from market_guided_stock_alpha.training.config import (
    EarlyStoppingConfig,
    OptimizerConfig,
    TrainingConfig,
    LightGBMTrainingConfig,
)
from market_guided_stock_alpha.training.losses import LossConfig


_MARKET_PROFILES = {
    "dynamic_all_a_shares": (
        "CSI500",
        {
            "name": "csi500_relative_excess_return_t1_t6",
            "benchmark": "000905.SH",
            "holding_period_trading_days": 5,
            "dependency_offset_trading_days": 6,
        },
        (
            "stage1_market63",
            "stage2_market63",
            "stage2_market84",
            "dual_frequency_exploratory",
        ),
    ),
    "dynamic_us_common_stocks": (
        "SPY",
        {
            "name": "spy_relative_excess_return_t1_t6",
            "benchmark": "SPY.US",
            "holding_period_trading_days": 5,
            "dependency_offset_trading_days": 6,
        },
        ("us_stage1_market63", "us_stage1_market84"),
    ),
}


def _layout(name: str) -> FeatureLayout:
    if name == "stage1_market63":
        return FeatureLayout.stage1()
    if name == "stage2_market63":
        return FeatureLayout.stage2("market63")
    if name == "stage2_market84":
        return FeatureLayout.stage2("market84")
    if name == "dual_frequency_exploratory":
        return FeatureLayout.dual_frequency()
    if name == "us_stage1_market63":
        return FeatureLayout.stage1("us_market63")
    if name == "us_stage1_market84":
        return FeatureLayout.stage1("us_market84")
    raise ValueError("unknown feature layout")


def _read_market(family: str, path: Path) -> Any:
    # Built per call so module-level reader names stay patchable.
    readers = {
        "market63": read_market63_artifact,
        "market84": read_market84_artifact,
        "us_market63": read_us_market63_artifact,
        "us_market84": read_us_market84_artifact,
    }
    return readers[family](path)


def _training_factory(document: Mapping[str, Any]):
    values = dict(document)
    optimizer = OptimizerConfig(**dict(values.pop("optimizer", {})))
    loss = LossConfig(**dict(values.pop("loss", {})))
    early_stopping = EarlyStoppingConfig(**dict(values.pop("early_stopping", {})))
    values.pop("seed", None)
    template = TrainingConfig(
        optimizer=optimizer,
        loss=loss,
        early_stopping=early_stopping,
        seed=0,
        **values,
    )
    return lambda seed: replace(template, seed=int(seed))


def _model_factory(document: Mapping[str, Any]):
    values = dict(document)
    family = str(values.pop("family"))
    if family == "master":
        return lambda layout, _seed: MasterModel.from_layout(layout, **values)
    if family == "gru":
        return lambda layout, _seed: GRURegressor.from_layout(layout, **values)
    if family == "transformer":
        return lambda layout, _seed: VanillaTransformerRegressor.from_layout(layout, **values)
    if family == "dual_frequency_master":
        return lambda layout, _seed: DualFrequencyMasterModel(
            DualFrequencyMasterConfig.from_layout(layout, **values)
        )
    raise ValueError(
        "rolling neural model family must be master, gru, transformer, or dual_frequency_master"
    )


def _lightgbm_training(document: Mapping[str, Any]) -> LightGBMTrainingConfig:
    model = dict(document["model"])
    if model.pop("family", None) != "lightgbm":
        raise ValueError("LightGBM config model family changed")
    parameters = dict(model.pop("parameters", {}))
    if model:
        raise ValueError("unexpected LightGBM model keys: {}".format(sorted(model)))
    return LightGBMTrainingConfig(
        parameters=parameters,
        **dict(document.get("training", {})),
    )


def validate_experiment_config(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one experiment config without opening data or output paths."""
    config = dict(document)
    required = {
        "schema_version",
        "experiment_name",
        "research_position",
        "universe",
        "label_contract",
        "feature_layout",
        "model",
        "training",
        "rolling",
        "initialization",
        "seeds",
        "data",
        "output_root",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError("experiment config fields missing: {}".format(sorted(missing)))
    if config["schema_version"] != 1:
        raise ValueError("unsupported experiment config schema")
    if config["research_position"] not in ("stage1", "stage2", "secondary", "exploratory"):
        raise ValueError("invalid research_position")
    profile = _MARKET_PROFILES.get(config["universe"])
    if profile is None:
        raise ValueError(
            "universe must be one of {}".format(",".join(sorted(_MARKET_PROFILES)))
        )
    benchmark_name, expected_label, layout_names = profile
    if config["label_contract"] != expected_label:
        raise ValueError(
            "label contract must be the canonical {}-relative target".format(
                benchmark_name
            )
        )
    layout_name = str(config["feature_layout"])
    layout = _layout(layout_name)
    if layout_name not in layout_names:
        raise ValueError(
            "feature_layout {} does not match universe {}".format(
                layout_name, config["universe"]
            )
        )
    rolling = dict(config["rolling"])
    if rolling.get("frequency") not in ("semiannual", "quarterly", "monthly"):
        raise ValueError("invalid rolling frequency")
    for name in ("train_start", "test_start", "test_end"):
        if name not in rolling:
            raise ValueError("rolling config missing {}".format(name))
    initialization = InitializationMode(str(config["initialization"]))
    seeds = tuple(config["seeds"])
    if not seeds or len(seeds) != len(set(seeds)) or any(type(value) is not int for value in seeds):
        raise ValueError("config seeds must be unique integers")
    data = dict(config["data"])
    required_data = {"calendar", "membership", "label", "label_column", "alpha158", "market"}
    if "intraday_price_volume" in layout.stock_families:
        required_data.add("intraday_price_volume")
    if layout.intraday_market_family:
        required_data.add("intraday_market")
    missing_data = required_data.difference(data)
    if missing_data:
        raise ValueError("data locations missing: {}".format(sorted(missing_data)))
    model_document = dict(config["model"])
    family = model_document.get("family")
    if family == "lightgbm":
        if initialization != InitializationMode.COLD:
            raise ValueError("LightGBM config must be cold-start")
        lightgbm_training = _lightgbm_training(config)
        model = None
        training = lightgbm_training
    else:
        model = _model_factory(model_document)(layout, seeds[0])
        training = _training_factory(config["training"])(seeds[0])
    return {
        "layout": layout,
        "model": model,
        "training": training,
        "frequency": rolling["frequency"],
        "initialization": initialization,
        "seeds": seeds,
    }


def load_experiment_config(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return document, validate_experiment_config(document)


class _ReferenceRun(object):
    """Lazy ``metrics.json`` lookups in an earlier run (local dir or r2://).

    Feeds the blotter's Δ column only; a missing reference never fails a run.
    """

    def __init__(self, location: str, cache: Path) -> None:
        self.root = str(location).rstrip("/")
        self.name = self.root.rsplit("/", 1)[-1]
        self.cache = cache
        self.found: Dict[Tuple[int, str], Optional[Dict[str, Any]]] = {}
        config = self._read("config.json")
        self.seeds = [] if config is None else [int(seed) for seed in config.get("seeds", [])]
        if config is None:
            note(("⚠ reference run not found at {}; Δ column stays empty".format(self.root), "yellow"))

    def _read(self, relative: str) -> Optional[Dict[str, Any]]:
        try:
            path = storage.materialize(
                "{}/{}".format(self.root, relative), Path.cwd(), self.cache, quiet=True
            )
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            return None

    def __call__(self, seed: int, target_id: str) -> Optional[Dict[str, Any]]:
        key = (int(seed), str(target_id))
        if key not in self.found:
            use = key[0] if key[0] in self.seeds or not self.seeds else self.seeds[0]
            self.found[key] = self._read("seed_{}/target_{}/metrics.json".format(use, key[1]))
        return self.found[key]


def run_from_json_config(
    config_path: Path,
    resume_override: bool = False,
    device: Any = None,
    compare: Optional[str] = None,
) -> Dict[str, Any]:
    source = Path(config_path)
    document, validated = load_experiment_config(source)
    with tempfile.TemporaryDirectory() as directory:
        return _run_document(
            document, validated, source.parent, Path(directory), resume_override, device, compare
        )


def _run_document(
    document: Mapping[str, Any],
    validated: Mapping[str, Any],
    base: Path,
    cache: Path,
    resume_override: bool,
    device: Any,
    compare: Optional[str] = None,
) -> Dict[str, Any]:
    def resolve(value: Any) -> Path:
        return storage.materialize(value, base, cache)

    layout = validated["layout"]
    data_config = dict(document["data"])
    calendar = TradingCalendar(read_table(resolve(data_config["calendar"])))
    membership = read_table(resolve(data_config["membership"]))
    label_path = resolve(data_config["label"])
    manifest_value = data_config.get("label_manifest")
    label_reader = DerivedLabelReader(
        label_path,
        str(data_config.get("label_column", DEFAULT_VALUE_COLUMN)),
        None if manifest_value is None else resolve(manifest_value),
    )
    alpha158 = read_alpha158_artifact(resolve(data_config["alpha158"]))
    market = _read_market(layout.market_family, resolve(data_config["market"]))
    intraday = (
        read_intraday_price_volume_artifact(resolve(data_config["intraday_price_volume"]))
        if "intraday_price_volume" in layout.stock_families
        else None
    )
    intraday_market = (
        read_intraday_market_artifact(resolve(data_config["intraday_market"]))
        if layout.intraday_market_family
        else None
    )
    target_config = dict(document["rolling"])
    targets = generate_rolling_targets(
        calendar,
        str(target_config["frequency"]),
        target_config["train_start"],
        target_config["test_start"],
        target_config["test_end"],
    )
    initialization = validated["initialization"]
    run_config = RollingRunConfig(
        experiment_name=str(document["experiment_name"]),
        seeds=tuple(int(value) for value in document.get("seeds", [42, 43, 44])),
        initialization=initialization,
        resume=bool(resume_override or document.get("resume", False)),
        minimum_security_count=int(document.get("minimum_security_count", 2)),
    )
    bundle = RollingDataBundle(
            calendar=calendar,
            membership=membership,
            label_reader=label_reader,
            alpha158=alpha158,
            market=market,
            intraday_price_volume=intraday,
            intraday_market=intraday_market,
        )

    reference = None if compare is None else _ReferenceRun(compare, cache / "reference")

    def execute(output_root: Path, on_target: Any = None) -> Dict[str, Any]:
        runner = RollingRunner(output_root, device, reference, on_target)
        if document["model"]["family"] == "lightgbm":
            return runner.run_lightgbm(
                run_config, targets, bundle, layout, validated["training"]
            )
        return runner.run(
            run_config,
            targets,
            bundle,
            layout,
            _model_factory(document["model"]),
            _training_factory(document["training"]),
        )

    output_value = document.get("output_root", "runs")
    if not storage.is_remote(output_value):
        local_output = resolve(output_value)
        status_to(local_output / run_config.experiment_name / "dashboard_status.json")
        return execute(local_output)
    remote_root = str(output_value).rstrip("/")
    status_to("{}/{}/dashboard_status.json".format(remote_root, run_config.experiment_name))
    local_root = cache / "output"
    if run_config.resume:
        try:
            local_root = resolve(
                "{}/{}".format(remote_root, run_config.experiment_name)
            ).parent
        except FileNotFoundError:
            pass

    def publish_target(target_root: Path) -> None:
        # Each target on R2 as soon as it finishes: a lost machine then costs one
        # target, and --resume continues from the rest. A failed upload is caught
        # up by the final publish of the whole run.
        relative = target_root.relative_to(local_root).as_posix()
        try:
            storage.publish(target_root, "{}/{}".format(remote_root, relative), overwrite=True, quiet=True)
        except Exception as error:
            note(("target upload failed ({}); continuing".format(error), "yellow"))

    try:
        result = dict(execute(local_root, publish_target))
        # The local cache is deleted on return; report where the run lives.
        result["run_directory"] = "{}/{}".format(remote_root, run_config.experiment_name)
        return result
    finally:
        # Publish partial targets too, so a failed run can --resume remotely.
        if local_root.is_dir():
            storage.publish(local_root, remote_root, overwrite=True)
