"""Small end-to-end rolling orchestration over canonical components."""

from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rich import box
from rich.console import Group
from rich.table import Table
from rich.text import Text

from market_guided_stock_alpha.artifacts import atomic_write_json, sha256_mapping, sha256_path
from market_guided_stock_alpha.data.calendar import TradingCalendar
from market_guided_stock_alpha.dataset.dataset import SequenceDataset
from market_guided_stock_alpha.dataset.layout import FeatureLayout
from market_guided_stock_alpha.dataset.normalization import fit_dataset_statistics
from market_guided_stock_alpha.derived.artifact import FeatureArtifact
from market_guided_stock_alpha.evaluation import (
    aggregate_overall,
    evaluate_target,
    raw_equal_weight_ensemble,
    summarize_seed_robustness,
)
from market_guided_stock_alpha.labels.derived_label import DerivedLabelReader
from market_guided_stock_alpha.modeling.transport import WholeDateLoader
from market_guided_stock_alpha.modeling.lightgbm import LightGBMRegressor
from market_guided_stock_alpha.rolling.artifacts import (
    read_prediction_artifact,
    write_prediction_artifact,
)
from market_guided_stock_alpha.progress import (
    StopRequested,
    allow_stop,
    board,
    curves,
    eta,
    header,
    log_to,
    note,
    progress,
    stop_requested,
    summary as summarize_job,
    ticker,
)
from market_guided_stock_alpha.rolling.prediction import (
    predict_lightgbm_model,
    predict_neural_model,
)
from market_guided_stock_alpha.rolling.schedule import (
    InitializationMode,
    initialize_target_model,
)
from market_guided_stock_alpha.rolling.targets import RollingTarget
from market_guided_stock_alpha.training.seed import resolve_device
from market_guided_stock_alpha.training.checkpoint import (
    CheckpointContract,
    CheckpointManager,
    feature_layout_identity,
)
from market_guided_stock_alpha.training.config import (
    LightGBMTrainingConfig,
    TrainingConfig,
)
from market_guided_stock_alpha.training.preprocessing import (
    preprocessing_to_dict,
    save_preprocessing,
)
from market_guided_stock_alpha.training.seed import seed_everything
from market_guided_stock_alpha.training.trainer import Trainer


ModelFactory = Callable[[FeatureLayout, int], torch.nn.Module]
TrainingConfigFactory = Callable[[int], TrainingConfig]


@dataclass(frozen=True)
class RollingDataBundle:
    calendar: TradingCalendar
    membership: pd.DataFrame
    label_reader: DerivedLabelReader
    alpha158: FeatureArtifact
    market: FeatureArtifact
    intraday_price_volume: Optional[FeatureArtifact] = None
    intraday_market: Optional[FeatureArtifact] = None


@dataclass(frozen=True)
class RollingRunConfig:
    experiment_name: str
    seeds: Sequence[int] = (42, 43, 44)
    initialization: InitializationMode = InitializationMode.WARM
    resume: bool = False
    minimum_security_count: int = 2

    def __post_init__(self) -> None:
        name = str(self.experiment_name).strip()
        if not name or any(value in name for value in ("/", "\\", "..")):
            raise ValueError("experiment_name must be a simple directory name")
        seeds = tuple(self.seeds)
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be a non-empty unique sequence")
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("seeds must contain non-negative integers")
        if type(self.minimum_security_count) is not int or self.minimum_security_count < 2:
            raise ValueError("minimum_security_count must be at least two")
        if not isinstance(self.initialization, InitializationMode):
            object.__setattr__(self, "initialization", InitializationMode(str(self.initialization)))


def _metric(value: Any) -> Tuple[str, str]:
    """Formatted metric with a direction arrow and sign colour."""
    if value is None:
        return "n/a", "dim"
    value = float(value)
    if value > 0:
        return "▲{:+.4f}".format(value), "green"
    if value < 0:
        return "▼{:+.4f}".format(value), "red"
    return "■{:+.4f}".format(value), "white"


def _selection(summary: Mapping[str, Any]) -> str:
    training = summary.get("training") or {}
    if training.get("best_epoch") is not None:
        return "ep {} · {:.4f}".format(
            training["best_epoch"], float(training.get("best_validation_metric", float("nan")))
        )
    if training.get("best_iteration") is not None:
        rank_ic = (training.get("best_validation_metrics") or {}).get("mean_rankic")
        return "rd {} · {:.4f}".format(
            training["best_iteration"], float("nan") if rank_ic is None else float(rank_ic)
        )
    return "n/a"


def _target_parts(
    seed: int,
    target_id: str,
    summary: Mapping[str, Any],
    resumed: bool,
    delta: Optional[float] = None,
) -> Tuple[Any, ...]:
    """One persistent log line per finished target (bars are transient)."""
    return (
        ("✓ ", "bold green"),
        ("seed {} ".format(seed), "bold"),
        ("target {}".format(target_id), "bold cyan"),
        (" (resumed)" if resumed else "", "dim"),
        ("  best ", "dim"),
        _selection(summary),
        ("  test rank IC ", "dim"),
        _metric(summary.get("rank_ic")),
        ("  IC ", "dim"),
        _metric(summary.get("ic")),
        ("  top10 ", "dim"),
        _metric(summary.get("top10")),
        ("  Δ vs ref " if delta is not None else "", "dim"),
        _metric(delta) if delta is not None else ("", ""),
    )


# (seed, target_id) -> that target's metrics.json in a reference run, or None.
Reference = Callable[[int, str], Optional[Mapping[str, Any]]]


def _delta(reference: Optional[Reference], seed: int, target_id: str, summary: Mapping[str, Any]) -> Optional[float]:
    if reference is None or summary.get("rank_ic") is None:
        return None
    other = reference(seed, target_id)
    if other is None or other.get("rank_ic") is None:
        return None
    return float(summary["rank_ic"]) - float(other["rank_ic"])


def _blotter(
    seed: int,
    targets: Sequence[RollingTarget],
    done: Mapping[str, Mapping[str, Any]],
    current: Optional[str],
    overall: Optional[Mapping[str, Any]] = None,
    reference: Optional[Reference] = None,
    compact: bool = False,
) -> Table:
    """Target blotter: status and test metrics for every target of one seed.

    ``compact`` (the live board) keeps rank IC and top10 only, so it fits half
    the screen; the exit summary shows every metric.
    """
    table = Table(
        title=None if overall is None else "seed {} · targets".format(seed),
        title_style="bold dodger_blue1",
        header_style="bold dodger_blue1",
        border_style="grey37",
        box=box.SIMPLE_HEAD,
    )
    names = ("rank_ic", "top10") if compact else ("rank_ic", "ic", "top5", "top10")
    labels = {"rank_ic": "rank IC", "ic": "IC", "top5": "top5", "top10": "top10"}
    table.add_column("target", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("best (valid)", no_wrap=True)
    for name in names:
        table.add_column(labels[name], justify="right", no_wrap=True)
    if reference is not None:
        table.add_column("Δ rIC ref", justify="right", no_wrap=True)

    def cells(row: Mapping[str, Any]) -> List[Text]:
        return [Text(*_metric(row.get(name))) for name in names]

    for target in targets:
        summary = done.get(target.target_id)
        if summary is not None:
            row = [target.target_id, Text("DONE", style="green"), _selection(summary)] + cells(summary)
            if reference is not None:
                delta = _delta(reference, seed, target.target_id, summary)
                row.append(Text(*_metric(delta)) if delta is not None else Text("—", style="dim"))
            table.add_row(*row)
        elif target.target_id == current:
            table.add_row(
                target.target_id, Text("TRAIN", style="bold yellow"), "", *[Text("···", style="dim")] * len(names)
            )
        else:
            table.add_row(Text(target.target_id, style="dim"), Text("QUEUED", style="dim"))
    if overall is not None:
        table.add_section()
        table.add_row(
            Text("OVERALL", style="bold"), Text("{} targets".format(overall.get("target_count")), style="dim"), "", *cells(overall)
        )
    return table


def _seed_matrix(
    targets: Sequence[RollingTarget],
    done: Mapping[int, Mapping[str, Mapping[str, Any]]],
    seeds: Sequence[int],
    ensemble: Mapping[str, Mapping[str, Any]],
    ensemble_overall: Optional[Mapping[str, Any]] = None,
) -> Table:
    """Test rank IC per target across seeds: spread, and the ensemble once run."""
    table = Table(
        title="rank IC across seeds",
        title_style="bold dodger_blue1",
        header_style="bold dodger_blue1",
        border_style="grey37",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("target", no_wrap=True)
    for seed in seeds:
        table.add_column("s{}".format(seed), justify="right", no_wrap=True)
    for name in ("mean", "sd", "ensemble"):
        table.add_column(name, justify="right", no_wrap=True)
    for target in targets:
        values = [done.get(int(seed), {}).get(target.target_id, {}).get("rank_ic") for seed in seeds]
        finite = [float(value) for value in values if value is not None]
        row: List[Any] = [target.target_id]
        row += [Text(*_metric(value)) if value is not None else Text("·", style="dim") for value in values]
        row.append(Text(*_metric(sum(finite) / len(finite))) if finite else Text("·", style="dim"))
        row.append(
            Text("{:.4f}".format(float(np.std(finite, ddof=1))), style="cyan") if len(finite) > 1 else Text("·", style="dim")
        )
        member = ensemble.get(target.target_id)
        row.append(Text(*_metric(member.get("rank_ic"))) if member else Text("·", style="dim"))
        table.add_row(*row)
    if ensemble_overall is not None:
        table.add_section()
        table.add_row(
            Text("OVERALL", style="bold"), *[""] * (len(seeds) + 2), Text(*_metric(ensemble_overall.get("rank_ic")))
        )
    return table


def _long_short_spread(frame: pd.DataFrame, fraction: float = 0.1) -> Dict[pd.Timestamp, float]:
    """Daily top-minus-bottom decile mean raw label (dashboard display only)."""
    spreads: Dict[pd.Timestamp, float] = {}
    for day, group in frame.groupby("trade_date", sort=True):
        prediction = group["prediction"].to_numpy(dtype="float64")
        raw = group["label_raw"].to_numpy(dtype="float64")
        mask = group["label_mask"].to_numpy(dtype="bool") & np.isfinite(prediction) & np.isfinite(raw)
        if int(mask.sum()) < 10:
            continue
        ordered = raw[mask][np.argsort(prediction[mask], kind="mergesort")]
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        spreads[pd.Timestamp(day)] = float(ordered[-count:].mean() - ordered[:count].mean())
    return spreads


class _RunMonitor(object):
    """Dashboard side of a rolling run: blotter, equity, ETA, ticker, stop.

    Display only: nothing here feeds back into training, predictions or the
    metric files.
    """

    def __init__(
        self,
        config: RollingRunConfig,
        targets: Sequence[RollingTarget],
        run_root: Path,
        reference: Optional[Reference],
        fields: Sequence[Tuple[str, Any]],
    ) -> None:
        log_to(run_root / "dashboard.log")
        allow_stop(True)
        for key, value in fields:
            header(key, value)
        if reference is not None:
            header("vs", getattr(reference, "name", "reference"))
        self.config = config
        self.targets = list(targets)
        self.reference = reference
        self.done: Dict[int, Dict[str, Mapping[str, Any]]] = {int(seed): {} for seed in config.seeds}
        self.ensemble: Dict[str, Mapping[str, Any]] = {}
        self.timings: List[Tuple[float, int]] = []
        self.curve: List[Tuple[float, float]] = []
        self.marks: List[int] = []
        self.started = time.monotonic()
        self.bar = progress(
            total=len(config.seeds) * len(targets), desc=config.experiment_name, unit="target"
        )

    def start(self, seed: int, target: RollingTarget) -> None:
        self.started = time.monotonic()
        if not self.done[int(seed)]:
            self.curve, self.marks = [], []
        self.bar.set_postfix_str("seed {} · target {}".format(seed, target.target_id))
        self._estimate(int(seed))
        self._show(int(seed), target.target_id)

    def finish(
        self,
        seed: int,
        target: RollingTarget,
        summary: Mapping[str, Any],
        frame: pd.DataFrame,
        resumed: bool,
    ) -> None:
        seed = int(seed)
        if not resumed:
            self.timings.append((time.monotonic() - self.started, max(len(target.train_dates), 1)))
        self.done[seed][target.target_id] = summary
        self.bar.update(1)
        note(*_target_parts(seed, target.target_id, summary, resumed, _delta(self.reference, seed, target.target_id, summary)))
        _summary, daily = evaluate_target(frame)
        spreads = _long_short_spread(frame)
        if self.curve:
            self.marks.append(len(self.curve))
        for day, value in zip(daily["trade_date"], daily["rank_ic"]):
            spread = spreads.get(pd.Timestamp(day), 0.0)
            self.curve.append((float(value) if np.isfinite(value) else 0.0, spread))
        curves(
            "EQUITY · seed {} · {} test days".format(seed, len(self.curve)),
            [
                ("Σ daily rank IC", np.cumsum([point[0] for point in self.curve]).tolist()),
                ("Σ daily L/S decile spread (raw label)", np.cumsum([point[1] for point in self.curve]).tolist()),
            ],
            self.marks,
        )
        ticker(
            "{}/{}".format(seed, target.target_id),
            ("s{} {} ".format(seed, target.target_id), "bold"),
            ("rIC ", "dim"),
            _metric(summary.get("rank_ic")),
            ("  top10 ", "dim"),
            _metric(summary.get("top10")),
        )
        self._show(seed, None)
        if stop_requested():
            raise StopRequested(
                "stopped after seed {} target {}; rerun with --resume to continue".format(seed, target.target_id)
            )

    def seed_done(self, seed: int, overall: Mapping[str, Any]) -> None:
        note(_blotter(seed, self.targets, self.done[int(seed)], None, overall, self.reference))
        summarize_job("seed {} rank IC {}".format(seed, _metric(overall.get("rank_ic"))[0]))

    def ensemble_done(self, summaries: Sequence[Mapping[str, Any]], overall: Mapping[str, Any]) -> None:
        self.ensemble = {str(item["target_id"]): item for item in summaries}
        table = _seed_matrix(self.targets, self.done, self.config.seeds, self.ensemble, overall)
        board(table, "SEEDS · ensemble")
        note(table)
        summarize_job("ensemble rank IC {}".format(_metric(overall.get("rank_ic"))[0]))

    def close(self) -> None:
        self.bar.close()
        eta(None)

    def _estimate(self, seed: int) -> None:
        """ETA from seconds per training date of targets trained this session."""
        if not self.timings:
            eta(None)
            return
        rate = sum(seconds for seconds, _ in self.timings) / sum(dates for _, dates in self.timings)
        seeds = [int(value) for value in self.config.seeds]
        remaining = sum(
            len(target.train_dates)
            for later in seeds[seeds.index(seed):]
            for target in self.targets
            if target.target_id not in self.done[later]
        )
        eta(rate * remaining)

    def _show(self, seed: int, current: Optional[str]) -> None:
        table = _blotter(seed, self.targets, self.done[seed], current, None, self.reference, compact=True)
        if len(self.config.seeds) > 1:
            board(
                Group(table, _seed_matrix(self.targets, self.done, self.config.seeds, self.ensemble)),
                "TARGETS · seed {}".format(seed),
            )
        else:
            board(table, "TARGETS · seed {}".format(seed))


def _build_dataset(
    data: RollingDataBundle,
    layout: FeatureLayout,
    statistics: Any,
    dates: Sequence[Any],
    minimum_security_count: int,
) -> SequenceDataset:
    return SequenceDataset(
        alpha158=data.alpha158,
        market=data.market,
        membership=data.membership,
        label_reader=data.label_reader,
        calendar=data.calendar,
        statistics=statistics,
        layout=layout,
        intraday_price_volume=data.intraday_price_volume,
        intraday_market=data.intraday_market,
        prediction_dates=dates,
        minimum_security_count=minimum_security_count,
    )


class RollingRunner:
    """Train, predict, evaluate, aggregate, and ensemble one experiment."""

    def __init__(self, output_root: Path, device: Any = None, reference: Optional[Reference] = None) -> None:
        self.output_root = Path(output_root)
        self.device = device
        self.reference = reference

    def run(
        self,
        config: RollingRunConfig,
        targets: Sequence[RollingTarget],
        data: RollingDataBundle,
        layout: FeatureLayout,
        model_factory: ModelFactory,
        training_config_factory: TrainingConfigFactory,
    ) -> Dict[str, Any]:
        if not targets:
            raise ValueError("rolling run requires at least one target")
        if len({target.target_id for target in targets}) != len(targets):
            raise ValueError("rolling target IDs must be unique")
        run_root = self.output_root / config.experiment_name
        run_root.mkdir(parents=True, exist_ok=True)
        public_identity = {
            "format_version": 1,
            "experiment_name": config.experiment_name,
            "seeds": list(config.seeds),
            "initialization": config.initialization.value,
            "feature_layout": feature_layout_identity(layout),
            "targets": [target.to_dict() for target in targets],
        }
        atomic_write_json(run_root / "config.json", public_identity)

        seed_results: Dict[int, Dict[str, Any]] = {}
        prediction_by_seed: Dict[int, Dict[str, pd.DataFrame]] = {}
        monitor = _RunMonitor(
            config,
            targets,
            run_root,
            self.reference,
            [
                ("device", resolve_device(self.device)),
                ("plan", "{} seed × {} targets".format(len(config.seeds), len(targets))),
            ],
        )
        for seed in config.seeds:
            previous_best: Optional[Path] = None
            target_summaries = []
            prediction_by_seed[int(seed)] = {}
            for target in targets:
                monitor.start(seed, target)
                target_root = run_root / "seed_{}".format(seed) / "target_{}".format(target.target_id)
                checkpoint_root = target_root / "checkpoints"
                preprocessing_path = target_root / "preprocessing.json"
                prediction_path = target_root / "predictions.pkl"
                metrics_path = target_root / "metrics.json"

                statistics = fit_dataset_statistics(
                    layout,
                    data.alpha158,
                    data.market,
                    data.membership,
                    target.train_dates,
                    data.intraday_price_volume,
                    data.intraday_market,
                )
                preprocessing_state = preprocessing_to_dict(statistics)
                seed_everything(int(seed), True)
                model = model_factory(layout, int(seed))
                training_config = training_config_factory(int(seed))
                if training_config.seed != seed:
                    raise ValueError("training config factory returned the wrong seed")
                contract = CheckpointContract.from_components(
                    model, training_config, layout, preprocessing_state
                )
                checkpoint_manager = CheckpointManager(checkpoint_root, contract)
                target_identity = {
                    "run": public_identity,
                    "seed": int(seed),
                    "target": target.to_dict(),
                    "model_type": contract.model_type,
                    "model_config": dict(contract.model_config),
                    "training_config": dict(contract.training_config),
                    "feature_order_sha256": contract.feature_order_sha256,
                    "preprocessing_sha256": sha256_mapping(preprocessing_state),
                }
                identity_sha256 = sha256_mapping(target_identity)
                best_path = checkpoint_root / "best.pt"

                resumed = False
                if config.resume and best_path.is_file() and prediction_path.is_file() and metrics_path.is_file():
                    frame, manifest = read_prediction_artifact(prediction_path)
                    if manifest.get("metadata", {}).get("experiment_config_sha256") == identity_sha256:
                        checkpoint_manager.load(best_path, model, restore_rng=False)
                        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                        resumed = True
                if not resumed:
                    target_root.mkdir(parents=True, exist_ok=True)
                    save_preprocessing(preprocessing_path, statistics)
                    initialization = initialize_target_model(
                        model,
                        config.initialization,
                        checkpoint_manager,
                        previous_best,
                    )
                    train_dataset = _build_dataset(
                        data, layout, statistics, target.train_dates, config.minimum_security_count
                    )
                    valid_dataset = _build_dataset(
                        data, layout, statistics, target.valid_dates, config.minimum_security_count
                    )
                    test_dataset = _build_dataset(
                        data, layout, statistics, target.test_dates, config.minimum_security_count
                    )
                    trainer = Trainer(
                        model,
                        training_config,
                        self.device,
                        checkpoint_manager,
                    )
                    training_result = trainer.fit(
                        WholeDateLoader(train_dataset, shuffle=True, seed=seed),
                        WholeDateLoader(valid_dataset, shuffle=False, seed=seed),
                    )
                    if not best_path.is_file():
                        raise RuntimeError("training did not publish a best checkpoint")
                    checkpoint_manager.load(best_path, model, restore_rng=False)
                    frame = predict_neural_model(
                        model,
                        WholeDateLoader(test_dataset, shuffle=False, seed=seed),
                        target.target_id,
                        self.device,
                    )
                    metadata = {
                        "experiment_name": config.experiment_name,
                        "experiment_config_sha256": identity_sha256,
                        "seed": int(seed),
                        "target_id": target.target_id,
                        "checkpoint_file": "checkpoints/best.pt",
                        "checkpoint_sha256": sha256_path(best_path),
                        "feature_layout": feature_layout_identity(layout),
                        "feature_order_sha256": contract.feature_order_sha256,
                        "model_type": contract.model_type,
                        "date_range": {
                            "start": str(target.test_start.date()),
                            "end": str(target.test_end.date()),
                        },
                        "initialization": initialization,
                        "preprocessing": "fit_on_current_target_training_dates",
                    }
                    write_prediction_artifact(prediction_path, frame, metadata)
                    summary, _daily = evaluate_target(frame)
                    summary.update(
                        {
                            "seed": int(seed),
                            "experiment_config_sha256": identity_sha256,
                            "training": training_result,
                            "initialization": initialization,
                        }
                    )
                    atomic_write_json(metrics_path, summary)
                previous_best = best_path
                prediction_by_seed[int(seed)][target.target_id] = frame
                target_summaries.append(summary)
                monitor.finish(seed, target, summary, frame, resumed)

            overall = aggregate_overall(target_summaries)
            overall.update({"seed": int(seed), "kind": "individual_seed_overall"})
            atomic_write_json(run_root / "seed_{}".format(seed) / "overall.json", overall)
            monitor.seed_done(seed, overall)
            seed_results[int(seed)] = {"targets": target_summaries, "overall": overall}

        monitor.close()
        ensemble_result: Optional[Dict[str, Any]] = None
        robustness: Optional[Dict[str, Any]] = None
        if len(config.seeds) >= 2:
            robustness = summarize_seed_robustness(
                {seed: result["overall"] for seed, result in seed_results.items()}
            )
            atomic_write_json(run_root / "individual_seed_summary.json", robustness)
            ensemble_summaries = []
            for target in targets:
                ensemble = raw_equal_weight_ensemble(
                    [prediction_by_seed[int(seed)][target.target_id] for seed in config.seeds]
                )
                ensemble_path = run_root / "ensemble" / "predictions" / "target_{}.pkl".format(target.target_id)
                write_prediction_artifact(
                    ensemble_path,
                    ensemble,
                    {
                        "experiment_name": config.experiment_name,
                        "target_id": target.target_id,
                        "ensemble": "raw_equal_weight_prediction_mean",
                        "seeds": list(config.seeds),
                    },
                )
                summary, _daily = evaluate_target(ensemble)
                ensemble_summaries.append(summary)
                atomic_write_json(
                    run_root / "ensemble" / "target_{}_metrics.json".format(target.target_id),
                    summary,
                )
            ensemble_overall = aggregate_overall(ensemble_summaries)
            ensemble_overall["kind"] = "raw_equal_weight_prediction_ensemble"
            ensemble_overall["seeds"] = list(config.seeds)
            atomic_write_json(run_root / "ensemble" / "metrics.json", ensemble_overall)
            ensemble_result = {"targets": ensemble_summaries, "overall": ensemble_overall}
            monitor.ensemble_done(ensemble_summaries, ensemble_overall)
        return {
            "run_directory": str(run_root),
            "individual_seeds": seed_results,
            "individual_seed_summary": robustness,
            "ensemble_evaluation": ensemble_result,
        }

    def run_lightgbm(
        self,
        config: RollingRunConfig,
        targets: Sequence[RollingTarget],
        data: RollingDataBundle,
        layout: FeatureLayout,
        training_config: LightGBMTrainingConfig,
    ) -> Dict[str, Any]:
        """Run the formal cold-start LightGBM baseline through the same outputs."""
        if config.initialization != InitializationMode.COLD:
            raise ValueError("formal LightGBM rolling is cold-start only")
        if not targets:
            raise ValueError("rolling run requires at least one target")
        if layout.intraday_market_family:
            raise ValueError("LightGBM baseline does not consume intraday market features")
        run_root = self.output_root / config.experiment_name
        run_root.mkdir(parents=True, exist_ok=True)
        public_identity = {
            "format_version": 1,
            "experiment_name": config.experiment_name,
            "model_family": "lightgbm",
            "seeds": list(config.seeds),
            "initialization": "cold",
            "feature_layout": feature_layout_identity(layout),
            "lightgbm_training": training_config.to_dict(),
            "targets": [target.to_dict() for target in targets],
        }
        atomic_write_json(run_root / "config.json", public_identity)
        seed_results: Dict[int, Dict[str, Any]] = {}
        prediction_by_seed: Dict[int, Dict[str, pd.DataFrame]] = {}
        monitor = _RunMonitor(
            config,
            targets,
            run_root,
            self.reference,
            [("model", "LightGBM"), ("plan", "{} seed × {} targets".format(len(config.seeds), len(targets)))],
        )
        for seed in config.seeds:
            target_summaries = []
            prediction_by_seed[int(seed)] = {}
            for target in targets:
                monitor.start(seed, target)
                target_root = run_root / "seed_{}".format(seed) / "target_{}".format(target.target_id)
                preprocessing_path = target_root / "preprocessing.json"
                model_path = target_root / "model.txt"
                training_record_path = target_root / "training.json"
                prediction_path = target_root / "predictions.pkl"
                metrics_path = target_root / "metrics.json"
                statistics = fit_dataset_statistics(
                    layout,
                    data.alpha158,
                    data.market,
                    data.membership,
                    target.train_dates,
                    data.intraday_price_volume,
                    None,
                )
                preprocessing_state = preprocessing_to_dict(statistics)
                identity = {
                    "run": public_identity,
                    "seed": int(seed),
                    "target": target.to_dict(),
                    "preprocessing_sha256": sha256_mapping(preprocessing_state),
                }
                identity_sha256 = sha256_mapping(identity)
                model = LightGBMRegressor()
                resumed = False
                if (
                    config.resume
                    and model_path.is_file()
                    and training_record_path.is_file()
                    and prediction_path.is_file()
                    and metrics_path.is_file()
                ):
                    frame, manifest = read_prediction_artifact(prediction_path)
                    record = json.loads(training_record_path.read_text(encoding="utf-8"))
                    if (
                        manifest.get("metadata", {}).get("experiment_config_sha256")
                        == identity_sha256
                        and record.get("model_sha256") == sha256_path(model_path)
                    ):
                        model.load_model(model_path, int(record["best_iteration"]))
                        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                        resumed = True
                if not resumed:
                    target_root.mkdir(parents=True, exist_ok=True)
                    save_preprocessing(preprocessing_path, statistics)
                    train_dataset = _build_dataset(
                        data, layout, statistics, target.train_dates, config.minimum_security_count
                    )
                    valid_dataset = _build_dataset(
                        data, layout, statistics, target.valid_dates, config.minimum_security_count
                    )
                    test_dataset = _build_dataset(
                        data, layout, statistics, target.test_dates, config.minimum_security_count
                    )
                    model.fit_rolling(train_dataset, valid_dataset, training_config, int(seed))
                    model.save_model(model_path)
                    record = dict(model.training_record or {})
                    record.update(
                        {
                            "experiment_config_sha256": identity_sha256,
                            "model_file": model_path.name,
                            "model_sha256": sha256_path(model_path),
                            "initialization": "cold",
                            "feature_layout": feature_layout_identity(layout),
                        }
                    )
                    atomic_write_json(training_record_path, record)
                    frame = predict_lightgbm_model(model, test_dataset, target.target_id)
                    write_prediction_artifact(
                        prediction_path,
                        frame,
                        {
                            "experiment_name": config.experiment_name,
                            "experiment_config_sha256": identity_sha256,
                            "seed": int(seed),
                            "target_id": target.target_id,
                            "checkpoint_file": model_path.name,
                            "checkpoint_sha256": record["model_sha256"],
                            "feature_layout": feature_layout_identity(layout),
                            "model_type": "lightgbm.Booster",
                            "date_range": {
                                "start": str(target.test_start.date()),
                                "end": str(target.test_end.date()),
                            },
                            "initialization": "cold",
                            "preprocessing": "fit_on_current_target_training_dates",
                            "input_representation": "eight_day_C_order_day_then_feature",
                        },
                    )
                    summary, _daily = evaluate_target(frame)
                    summary.update(
                        {
                            "seed": int(seed),
                            "experiment_config_sha256": identity_sha256,
                            "training": record,
                            "initialization": "cold",
                        }
                    )
                    atomic_write_json(metrics_path, summary)
                prediction_by_seed[int(seed)][target.target_id] = frame
                target_summaries.append(summary)
                monitor.finish(seed, target, summary, frame, resumed)
            overall = aggregate_overall(target_summaries)
            overall.update({"seed": int(seed), "kind": "individual_seed_overall"})
            atomic_write_json(run_root / "seed_{}".format(seed) / "overall.json", overall)
            monitor.seed_done(seed, overall)
            seed_results[int(seed)] = {"targets": target_summaries, "overall": overall}

        monitor.close()
        robustness = None
        ensemble_result = None
        if len(config.seeds) >= 2:
            robustness = summarize_seed_robustness(
                {seed: result["overall"] for seed, result in seed_results.items()}
            )
            atomic_write_json(run_root / "individual_seed_summary.json", robustness)
            ensemble_summaries = []
            for target in targets:
                ensemble = raw_equal_weight_ensemble(
                    [prediction_by_seed[int(seed)][target.target_id] for seed in config.seeds]
                )
                ensemble_path = run_root / "ensemble" / "predictions" / "target_{}.pkl".format(target.target_id)
                write_prediction_artifact(
                    ensemble_path,
                    ensemble,
                    {
                        "experiment_name": config.experiment_name,
                        "target_id": target.target_id,
                        "ensemble": "raw_equal_weight_prediction_mean",
                        "seeds": list(config.seeds),
                    },
                )
                summary, _daily = evaluate_target(ensemble)
                ensemble_summaries.append(summary)
                atomic_write_json(
                    run_root / "ensemble" / "target_{}_metrics.json".format(target.target_id),
                    summary,
                )
            ensemble_overall = aggregate_overall(ensemble_summaries)
            ensemble_overall.update(
                {
                    "kind": "raw_equal_weight_prediction_ensemble",
                    "seeds": list(config.seeds),
                }
            )
            atomic_write_json(run_root / "ensemble" / "metrics.json", ensemble_overall)
            ensemble_result = {"targets": ensemble_summaries, "overall": ensemble_overall}
            monitor.ensemble_done(ensemble_summaries, ensemble_overall)
        return {
            "run_directory": str(run_root),
            "individual_seeds": seed_results,
            "individual_seed_summary": robustness,
            "ensemble_evaluation": ensemble_result,
        }
