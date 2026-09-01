"""Transparent JSON-to-canonical rolling wiring; no legacy execution IDs."""

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from market_guided_stock_alpha.data.calendar import TradingCalendar
from market_guided_stock_alpha.data.io import read_table
from market_guided_stock_alpha.dataset.layout import FeatureLayout
from market_guided_stock_alpha.derived.readers import (
    read_alpha158_artifact,
    read_intraday_market_artifact,
    read_intraday_price_volume_artifact,
    read_market63_artifact,
    read_market84_artifact,
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


def _path(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _layout(name: str) -> FeatureLayout:
    if name == "stage1_market63":
        return FeatureLayout.stage1()
    if name == "stage2_market63":
        return FeatureLayout.stage2("market63")
    if name == "stage2_market84":
        return FeatureLayout.stage2("market84")
    if name == "dual_frequency_exploratory":
        return FeatureLayout.dual_frequency()
    raise ValueError("unknown feature layout")


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


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            for nested in _walk_keys(item):
                yield nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            for nested in _walk_keys(item):
                yield nested


def validate_experiment_config(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate one public config without opening user data or output paths."""
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
        raise ValueError("public config fields missing: {}".format(sorted(missing)))
    if config["schema_version"] != 1:
        raise ValueError("unsupported public config schema")
    if config["research_position"] not in ("stage1", "stage2", "secondary", "exploratory"):
        raise ValueError("invalid research_position")
    if config["universe"] != "dynamic_all_a_shares":
        raise ValueError("public universe must be dynamic_all_a_shares")
    expected_label = {
        "name": "csi500_relative_excess_return_t1_t6",
        "benchmark": "000905.SH",
        "holding_period_trading_days": 5,
        "dependency_offset_trading_days": 6,
    }
    if config["label_contract"] != expected_label:
        raise ValueError("public label contract must be the canonical CSI500-relative target")
    forbidden = {
        "execution_id",
        "ensemble_method",
        "beta_csi300",
        "beta_csi800",
        "provided_tag_v1",
        "server_namespace",
        "ppu",
    }
    present = forbidden.intersection(set(_walk_keys(config)))
    if present:
        raise ValueError("forbidden historical config keys: {}".format(sorted(present)))
    layout = _layout(str(config["feature_layout"]))
    rolling = dict(config["rolling"])
    if rolling.get("frequency") not in ("semiannual", "quarterly", "monthly"):
        raise ValueError("invalid rolling frequency")
    for name in ("train_start", "test_start", "test_end"):
        if name not in rolling:
            raise ValueError("rolling config missing {}".format(name))
    initialization = InitializationMode(str(config["initialization"]))
    seeds = tuple(config["seeds"])
    if not seeds or len(seeds) != len(set(seeds)) or any(type(value) is not int for value in seeds):
        raise ValueError("public config seeds must be unique integers")
    data = dict(config["data"])
    required_data = {"calendar", "membership", "label", "label_column", "alpha158", "market"}
    if "intraday_price_volume" in layout.stock_families:
        required_data.add("intraday_price_volume")
    if layout.intraday_market_family:
        required_data.add("intraday_market")
    missing_data = required_data.difference(data)
    if missing_data:
        raise ValueError("public data locations missing: {}".format(sorted(missing_data)))
    for value in list(data.values()) + [config["output_root"]]:
        text = str(value)
        if Path(text).is_absolute() or (len(text) >= 3 and text[1:3] in (":\\", ":/")):
            raise ValueError("public config paths must be relative")
        if "/mnt/" in text.replace("\\", "/"):
            raise ValueError("public config contains a company absolute path")
    model_document = dict(config["model"])
    family = model_document.get("family")
    if family == "lightgbm":
        if initialization != InitializationMode.COLD:
            raise ValueError("formal LightGBM config must be cold-start")
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


def run_from_json_config(
    config_path: Path,
    resume_override: bool = False,
    device: Any = None,
) -> Dict[str, Any]:
    source = Path(config_path)
    document, validated = load_experiment_config(source)
    base = source.parent
    layout = validated["layout"]
    data_config = dict(document["data"])
    calendar = TradingCalendar(read_table(_path(base, data_config["calendar"])))
    membership = read_table(_path(base, data_config["membership"]))
    label_path = _path(base, data_config["label"])
    manifest_value = data_config.get("label_manifest")
    label_reader = DerivedLabelReader(
        label_path,
        str(data_config.get("label_column", DEFAULT_VALUE_COLUMN)),
        None if manifest_value is None else _path(base, manifest_value),
    )
    alpha158 = read_alpha158_artifact(_path(base, data_config["alpha158"]))
    market = (
        read_market63_artifact(_path(base, data_config["market"]))
        if layout.market_family == "market63"
        else read_market84_artifact(_path(base, data_config["market"]))
    )
    intraday = (
        read_intraday_price_volume_artifact(_path(base, data_config["intraday_price_volume"]))
        if "intraday_price_volume" in layout.stock_families
        else None
    )
    intraday_market = (
        read_intraday_market_artifact(_path(base, data_config["intraday_market"]))
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
    output_root = _path(base, document.get("output_root", "runs"))
    bundle = RollingDataBundle(
            calendar=calendar,
            membership=membership,
            label_reader=label_reader,
            alpha158=alpha158,
            market=market,
            intraday_price_volume=intraday,
            intraday_market=intraday_market,
        )
    runner = RollingRunner(output_root, device)
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
