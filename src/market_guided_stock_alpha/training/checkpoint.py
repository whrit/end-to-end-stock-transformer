"""Portable, contract-validated model and preprocessing checkpoints."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from market_guided_stock_alpha.dataset.layout import FeatureLayout
from market_guided_stock_alpha.training.config import TrainingConfig


def feature_layout_identity(layout: FeatureLayout) -> Dict[str, Any]:
    return {
        "stock_families": list(layout.stock_families),
        "market_family": layout.market_family,
        "intraday_market_family": layout.intraday_market_family,
        "position": layout.position,
        "lookback": layout.lookback,
        "stock_dim": layout.stock_dim,
        "market_dim": layout.market_dim,
        "intraday_market_dim": layout.intraday_market_dim,
    }


def feature_order_hash(layout: FeatureLayout) -> str:
    payload = {
        "stock": list(layout.stock_feature_names),
        "market": list(layout.market_feature_names),
        "intraday_market": list(layout.intraday_market_feature_names),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CheckpointContract:
    model_type: str
    model_config: Mapping[str, Any]
    training_config: Mapping[str, Any]
    feature_layout: Mapping[str, Any]
    feature_order_sha256: str
    preprocessing_state: Mapping[str, Any]
    seed: int

    @classmethod
    def from_components(
        cls,
        model: torch.nn.Module,
        training_config: TrainingConfig,
        layout: FeatureLayout,
        preprocessing_state: Mapping[str, Any],
    ) -> "CheckpointContract":
        if not hasattr(model, "config_dict"):
            raise ValueError("checkpointed model must expose config_dict()")
        if not preprocessing_state or preprocessing_state.get("fit_scope") != "training_dates_only":
            raise ValueError("checkpoint requires train-fit preprocessing provenance")
        return cls(
            model_type="{}.{}".format(model.__class__.__module__, model.__class__.__name__),
            model_config=dict(model.config_dict()),
            training_config=training_config.to_dict(),
            feature_layout=feature_layout_identity(layout),
            feature_order_sha256=feature_order_hash(layout),
            preprocessing_state=dict(preprocessing_state),
            seed=training_config.seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_config": dict(self.model_config),
            "training_config": dict(self.training_config),
            "feature_layout": dict(self.feature_layout),
            "feature_order_sha256": self.feature_order_sha256,
            "preprocessing_state": dict(self.preprocessing_state),
            "seed": self.seed,
        }


def _rng_state() -> Dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if payload.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    if payload.get("mps_rng_state") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(payload["mps_rng_state"])


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CheckpointManager:
    def __init__(self, directory: Path, contract: CheckpointContract) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = contract

    def _validate(self, payload: Mapping[str, Any]) -> None:
        required = {
            "format_version",
            "contract",
            "epoch",
            "validation_metric",
            "model_state_dict",
            "optimizer_state_dict",
            "history",
            "python_random_state",
            "numpy_random_state",
            "torch_cpu_rng_state",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError("checkpoint fields missing: {}".format(sorted(missing)))
        if payload["format_version"] != 1:
            raise ValueError("unsupported checkpoint format")
        if dict(payload["contract"]) != self.contract.to_dict():
            raise ValueError("checkpoint model/layout/preprocessing contract mismatch")

    def _atomic_save(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(path.name + ".partial")
        try:
            with temporary.open("wb") as handle:
                torch.save(dict(payload), handle)
                handle.flush()
                os.fsync(handle.fileno())
            self._validate(_torch_load(temporary))
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save(
        self,
        epoch: int,
        validation_metric: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        history: Any,
        is_best: bool,
    ) -> Dict[str, Path]:
        payload = {
            "format_version": 1,
            "contract": self.contract.to_dict(),
            "epoch": int(epoch),
            "validation_metric": float(validation_metric),
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "history": list(history),
            **_rng_state(),
        }
        last = self.directory / "last.pt"
        self._atomic_save(last, payload)
        result = {"last": last}
        if is_best:
            best = self.directory / "best.pt"
            self._atomic_save(best, payload)
            result["best"] = best
        return result

    def load(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        payload = _torch_load(Path(path))
        self._validate(payload)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if restore_rng:
            _restore_rng_state(payload)
        return payload

    def load_weights_only(
        self,
        path: Path,
        model: torch.nn.Module,
    ) -> Dict[str, Any]:
        """Warm-start model weights while deliberately resetting training state.

        The source target may have different train-fit preprocessing statistics,
        so warm compatibility is restricted to model, seed, and feature layout.
        Optimizer/RNG/history are intentionally not restored.
        """
        payload = _torch_load(Path(path))
        required = {"format_version", "contract", "model_state_dict"}
        missing = required.difference(payload)
        if missing:
            raise ValueError("warm-start checkpoint fields missing: {}".format(sorted(missing)))
        if payload["format_version"] != 1:
            raise ValueError("unsupported checkpoint format")
        source = dict(payload["contract"])
        expected = self.contract.to_dict()
        for field in (
            "model_type",
            "model_config",
            "feature_layout",
            "feature_order_sha256",
            "seed",
        ):
            if source.get(field) != expected[field]:
                raise ValueError("warm-start {} contract mismatch".format(field))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return payload
