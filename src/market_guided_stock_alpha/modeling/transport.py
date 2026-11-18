"""Thin whole-date tensor transport from canonical dataset samples."""

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from market_guided_stock_alpha.dataset.dataset import SequenceSample


def _raw_label_device(target: torch.device) -> torch.device:
    # MPS has no float64: keep exact raw labels on CPU; consumers move them as float32.
    return torch.device("cpu") if target.type == "mps" else target


@dataclass(frozen=True)
class DateBatch:
    """Exactly one prediction date and its complete dynamic cross-section."""

    prediction_date: pd.Timestamp
    security_codes: Tuple[str, ...]
    stock_features: torch.Tensor
    market_features: torch.Tensor
    labels: torch.Tensor
    labels_raw: torch.Tensor
    label_mask: torch.Tensor
    stock_mask: torch.Tensor
    intraday_market_features: Optional[torch.Tensor] = None

    @property
    def security_count(self) -> int:
        return len(self.security_codes)


def adapt_sequence_sample(
    sample: SequenceSample, device: Any = "cpu"
) -> DateBatch:
    """Convert one date without broadcasting market state."""
    target = torch.device(device)
    intraday_market = (
        None
        if sample.intraday_market_features is None
        else torch.as_tensor(
            np.ascontiguousarray(sample.intraday_market_features),
            dtype=torch.float32,
            device=target,
        )
    )
    return DateBatch(
        prediction_date=pd.Timestamp(sample.prediction_date),
        security_codes=tuple(sample.security_codes),
        stock_features=torch.as_tensor(
            np.ascontiguousarray(sample.stock_features),
            dtype=torch.float32,
            device=target,
        ),
        market_features=torch.as_tensor(
            np.ascontiguousarray(sample.market_features),
            dtype=torch.float32,
            device=target,
        ),
        labels=torch.as_tensor(
            np.array(sample.labels, dtype="float32", order="C", copy=True),
            dtype=torch.float32,
            device=target,
        ),
        labels_raw=torch.as_tensor(
            np.array(sample.labels_raw, dtype="float64", order="C", copy=True),
            dtype=torch.float64,
            device=_raw_label_device(target),
        ),
        label_mask=torch.as_tensor(
            np.array(sample.label_mask, dtype="bool", order="C", copy=True),
            dtype=torch.bool,
            device=target,
        ),
        stock_mask=torch.as_tensor(
            np.ascontiguousarray(sample.stock_mask), dtype=torch.bool, device=target
        ),
        intraday_market_features=intraday_market,
    )


def move_date_batch(batch: DateBatch, device: Any) -> DateBatch:
    target = torch.device(device)
    return DateBatch(
        prediction_date=batch.prediction_date,
        security_codes=batch.security_codes,
        stock_features=batch.stock_features.to(target),
        market_features=batch.market_features.to(target),
        labels=batch.labels.to(target),
        labels_raw=batch.labels_raw.to(_raw_label_device(target)),
        label_mask=batch.label_mask.to(target),
        stock_mask=batch.stock_mask.to(target),
        intraday_market_features=(
            None
            if batch.intraday_market_features is None
            else batch.intraday_market_features.to(target)
        ),
    )


def forward_date_batch(model: torch.nn.Module, batch: DateBatch) -> torch.Tensor:
    kind = getattr(model, "input_kind", None)
    if kind == "stock_and_market":
        predictions = model(batch.stock_features, batch.market_features)
    elif kind == "stock":
        predictions = model(batch.stock_features)
    elif kind == "dual_frequency":
        if batch.intraday_market_features is None:
            raise ValueError("dual-frequency model requires intraday market features")
        predictions = model(
            batch.stock_features,
            batch.market_features,
            batch.intraday_market_features,
        )
    else:
        raise ValueError("model does not declare a canonical input_kind")
    if predictions.ndim != 1 or predictions.shape[0] != batch.security_count:
        raise ValueError("model output must be one prediction per security")
    return predictions


class WholeDateLoader:
    """Deterministic iterator; it never creates random stock mini-batches."""

    date_batch_size = 1

    def __init__(
        self,
        dataset: Any,
        shuffle: bool = False,
        seed: int = 0,
        device: Any = "cpu",
    ) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[DateBatch]:
        positions = np.arange(len(self.dataset), dtype="int64")
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(positions)
        for position in positions:
            value = self.dataset[int(position)]
            batch = (
                value
                if isinstance(value, DateBatch)
                else adapt_sequence_sample(value, self.device)
            )
            if isinstance(value, DateBatch) and value.stock_features.device != self.device:
                batch = move_date_batch(value, self.device)
            yield batch
