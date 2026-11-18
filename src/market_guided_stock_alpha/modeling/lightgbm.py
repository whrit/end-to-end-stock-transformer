"""Historical 8-day flattened LightGBM baseline representation and trainer."""

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from market_guided_stock_alpha.dataset.dataset import SequenceSample
from market_guided_stock_alpha.dataset.layout import FeatureLayout
from market_guided_stock_alpha.modeling.transport import usable_cpu_count
from market_guided_stock_alpha.progress import metric as show_metric, progress, reset_metrics
from market_guided_stock_alpha.training.config import LightGBMTrainingConfig
from market_guided_stock_alpha.training.metrics import daily_rank_ic


def flatten_sequence_sample(sample: SequenceSample) -> np.ndarray:
    """Flatten [N,T,stock+market] in C-order, day then feature.

    The 03 formal baseline used every one of the eight history days.  Market
    state is broadcast here, not in dataset storage.
    """
    stock = np.asarray(sample.stock_features, dtype="float32")
    market = np.asarray(sample.market_features, dtype="float32")
    if stock.ndim != 3 or market.ndim != 2 or stock.shape[1] != market.shape[0]:
        raise ValueError("LightGBM sample requires stock [N,T,F] and market [T,M]")
    repeated = np.broadcast_to(
        market[None, :, :], (stock.shape[0], market.shape[0], market.shape[1])
    )
    combined = np.concatenate((stock, repeated), axis=2)
    return np.ascontiguousarray(combined).reshape(stock.shape[0], -1)


def lightgbm_feature_order(layout: FeatureLayout) -> Tuple[str, ...]:
    combined = layout.stock_feature_names + layout.market_feature_names
    return tuple(
        "t{}_{}".format(position - layout.lookback + 1, name)
        for position in range(layout.lookback)
        for name in combined
    )


class LightGBMRegressor:
    """Small lazy-import wrapper; importing canonical modeling needs no LightGBM."""

    def __init__(self) -> None:
        self.booster = None
        self.best_iteration = None
        self.training_record = None

    @staticmethod
    def _library() -> Any:
        try:
            import lightgbm
        except ImportError as exc:
            raise ImportError(
                "LightGBMRegressor requires the optional 'lightgbm' package"
            ) from exc
        return lightgbm

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        parameters: Mapping[str, Any],
        num_boost_round: int,
    ) -> "LightGBMRegressor":
        lightgbm = self._library()
        matrix = np.asarray(features, dtype="float32")
        target = np.asarray(labels, dtype="float32")
        if matrix.ndim != 2 or target.shape != (matrix.shape[0],):
            raise ValueError("LightGBM fit shapes are invalid")
        if type(num_boost_round) is not int or num_boost_round <= 0:
            raise ValueError("num_boost_round must be a positive integer")
        dataset = lightgbm.Dataset(matrix, label=target)
        self.booster = lightgbm.train(
            dict(parameters),
            dataset,
            num_boost_round=num_boost_round,
        )
        return self

    @staticmethod
    def _effective_parameters(
        parameters: Mapping[str, Any], seed: int
    ) -> Dict[str, Any]:
        if type(seed) is not int or seed < 0:
            raise ValueError("LightGBM seed must be a non-negative integer")
        result = dict(parameters)
        if result.get("num_threads") == 0:
            # 0 = every CPU this job may use (container quota aware), not
            # OpenMP's host-wide default.
            result["num_threads"] = usable_cpu_count()
        result.update(
            {
                "seed": seed,
                "feature_fraction_seed": seed,
                "bagging_seed": seed,
                "data_random_seed": seed,
            }
        )
        return result

    @staticmethod
    def _validation_metrics(
        predictions: np.ndarray,
        labels_raw: np.ndarray,
        labels_processed: np.ndarray,
        row_ptr: np.ndarray,
        minimum_count: int,
    ) -> Dict[str, float]:
        ic_values = []
        rank_values = []
        squared_error = 0.0
        observation_count = 0
        for position in range(len(row_ptr) - 1):
            left, right = int(row_ptr[position]), int(row_ptr[position + 1])
            prediction = predictions[left:right]
            raw = labels_raw[left:right]
            processed = labels_processed[left:right]
            mask = np.isfinite(prediction) & np.isfinite(raw) & np.isfinite(processed)
            if int(mask.sum()) >= minimum_count:
                current_prediction = prediction[mask]
                current_raw = raw[mask]
                if current_prediction.std(ddof=0) > 0.0 and current_raw.std(ddof=0) > 0.0:
                    ic_values.append(float(np.corrcoef(current_prediction, current_raw)[0, 1]))
                    rank_values.append(
                        daily_rank_ic(
                            current_prediction,
                            current_raw,
                            np.ones(len(current_prediction), dtype="bool"),
                        )
                    )
            processed_mask = np.isfinite(prediction) & np.isfinite(processed)
            residual = prediction[processed_mask] - processed[processed_mask]
            squared_error += float(np.dot(residual, residual))
            observation_count += int(processed_mask.sum())
        finite_ic = [value for value in ic_values if math.isfinite(value)]
        finite_rank = [value for value in rank_values if math.isfinite(value)]
        return {
            "mean_rankic": float(np.mean(finite_rank)) if finite_rank else float("nan"),
            "mean_ic": float(np.mean(finite_ic)) if finite_ic else float("nan"),
            "daily_zscore_mse": (
                float(squared_error / observation_count)
                if observation_count
                else float("nan")
            ),
        }

    def fit_rolling(
        self,
        train_dataset: Any,
        valid_dataset: Any,
        config: LightGBMTrainingConfig,
        seed: int,
    ) -> "LightGBMRegressor":
        """Cold-start formal baseline with MASTER-equivalent validation selection."""
        lightgbm = self._library()
        train = assemble_supervised_matrix(train_dataset)
        valid = assemble_supervised_matrix(valid_dataset)
        parameters = self._effective_parameters(config.parameters, seed)
        train_set = lightgbm.Dataset(
            train["features"], label=train["labels_processed"], params=parameters
        ).construct()
        valid_set = lightgbm.Dataset(
            valid["features"],
            label=valid["labels_processed"],
            reference=train_set,
            params=parameters,
        ).construct()
        # LightGBM now holds binned copies; the float32 matrices (tens of GB for
        # 8 days x 1,768 columns on US data) are not needed while boosting.
        del train["features"], valid["features"]

        def validation_feval(predictions: Any, _dataset: Any):
            metrics = self._validation_metrics(
                np.asarray(predictions, dtype="float64"),
                valid["labels_raw"],
                valid["labels_processed"],
                valid["row_ptr"],
                config.minimum_valid_labels,
            )
            if not math.isfinite(metrics["mean_rankic"]):
                raise RuntimeError("LightGBM validation produced no finite daily RankIC")
            return [
                ("mean_rankic", metrics["mean_rankic"], True),
                ("mean_ic", metrics["mean_ic"], True),
                ("daily_zscore_mse", metrics["daily_zscore_mse"], False),
            ]

        selection: Dict[str, Any] = {"history": []}
        rounds = progress(total=config.num_boost_round, desc="boosting", unit="round", leave=False)
        reset_metrics()

        def selection_callback(environment: Any) -> None:
            current = {}
            for item in environment.evaluation_result_list or ():
                data_name, metric_name, metric_value = item[:3]
                if data_name == "valid":
                    current[str(metric_name)] = float(metric_value)
            required = ("mean_rankic", "mean_ic", "daily_zscore_mse")
            if any(name not in current for name in required):
                raise RuntimeError("LightGBM validation metrics are incomplete")
            iteration = int(environment.iteration) + 1
            key = (
                current["mean_rankic"],
                current["mean_ic"],
                -current["daily_zscore_mse"],
                -iteration,
            )
            selection["history"].append({"iteration": iteration, **current})
            if selection.get("best_key") is None or key > selection["best_key"]:
                selection["best_key"] = key
                selection["best_iteration"] = iteration
                selection["best_metrics"] = dict(current)
                selection["best_score_list"] = list(environment.evaluation_result_list)
            rounds.update(1)
            show_metric("val rank IC", current["mean_rankic"])
            show_metric("val IC", current["mean_ic"])
            show_metric("val zscore MSE", current["daily_zscore_mse"], higher_is_better=False)
            rounds.set_postfix(
                rank_ic="{:.4f}".format(current["mean_rankic"]),
                best="{:.4f}@{}".format(
                    selection["best_metrics"]["mean_rankic"], selection["best_iteration"]
                ),
                refresh=False,
            )
            if iteration - int(selection["best_iteration"]) >= config.early_stopping_rounds:
                raise lightgbm.callback.EarlyStopException(
                    int(selection["best_iteration"]) - 1,
                    selection["best_score_list"],
                )

        selection_callback.order = 30
        selection_callback.before_iteration = False
        try:
            self.booster = lightgbm.train(
                parameters,
                train_set,
                num_boost_round=config.num_boost_round,
                valid_sets=[valid_set],
                valid_names=["valid"],
                feval=validation_feval,
                callbacks=[selection_callback],
            )
        finally:
            rounds.close()
        self.best_iteration = int(selection["best_iteration"])
        self.training_record = {
            "seed": int(seed),
            "cold_start": True,
            "warm_start": False,
            "best_iteration": self.best_iteration,
            "best_validation_metrics": dict(selection["best_metrics"]),
            "validation_selection_key": "mean_rankic,mean_ic,-daily_zscore_mse,-iteration",
            "train_observation_count": int(len(train["labels_processed"])),
            "valid_observation_count": int(len(valid["labels_processed"])),
            "parameters": parameters,
            "config": config.to_dict(),
        }
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("LightGBMRegressor is not fitted")
        return np.asarray(
            self.booster.predict(
                np.asarray(features, dtype="float32"),
                num_iteration=self.best_iteration,
            ),
            dtype="float64",
        )

    def save_model(self, path: Path) -> None:
        if self.booster is None or self.best_iteration is None:
            raise RuntimeError("LightGBMRegressor is not fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        try:
            self.booster.save_model(str(temporary), num_iteration=self.best_iteration)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_model(self, path: Path, best_iteration: int) -> "LightGBMRegressor":
        if type(best_iteration) is not int or best_iteration <= 0:
            raise ValueError("best_iteration must be positive")
        self.booster = self._library().Booster(model_file=str(Path(path)))
        self.best_iteration = best_iteration
        return self


def assemble_supervised_matrix(dataset: Any) -> Dict[str, np.ndarray]:
    """Flatten all supervised rows while preserving daily validation boundaries.

    Each day's rows are freed as they are copied into the preallocated output,
    so peak memory is about one copy of the matrix plus one day, not two copies.
    """
    features = []
    processed = []
    raw = []
    row_ptr = [0]
    for position in progress(
        range(len(dataset)), desc="assemble LightGBM rows", unit="date", leave=False
    ):
        sample = dataset[position]
        matrix = flatten_sequence_sample(sample)
        mask = (
            np.asarray(sample.label_mask, dtype="bool")
            & np.isfinite(np.asarray(sample.labels, dtype="float64"))
            & np.isfinite(np.asarray(sample.labels_raw, dtype="float64"))
            & np.isfinite(matrix).all(axis=1)
        )
        features.append(matrix[mask])
        processed.append(np.asarray(sample.labels, dtype="float32")[mask])
        raw.append(np.asarray(sample.labels_raw, dtype="float64")[mask])
        row_ptr.append(row_ptr[-1] + int(mask.sum()))
    if not features or row_ptr[-1] <= 0:
        raise ValueError("LightGBM dataset contains no supervised row")
    output = np.empty((row_ptr[-1], features[0].shape[1]), dtype="float32")
    features.reverse()
    for start, end in zip(row_ptr[:-1], row_ptr[1:]):
        output[start:end] = features.pop()
    return {
        "features": output,
        "labels_processed": np.concatenate(processed).astype("float32", copy=False),
        "labels_raw": np.concatenate(raw).astype("float64", copy=False),
        "row_ptr": np.asarray(row_ptr, dtype="int64"),
    }
