"""Thin whole-date tensor transport from canonical dataset samples."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch

from market_guided_stock_alpha.dataset.dataset import SequenceDataset, SequenceSample
from market_guided_stock_alpha.progress import duration_text, note, progress

GIB = 1024 ** 3
# Safety margins for the epoch cache (used on CUDA only; see WholeDateLoader).
GPU_RESERVE_FRACTION = 0.30  # of GPU memory, kept free for activations
GPU_RESERVE_MIN = 6 * GIB  # d_model 512 peaks near 3.3 GB at 2,500 stocks per date
RAM_RESERVE_FRACTION = 0.15  # of the (container) memory limit
RAM_RESERVE_MIN = 8 * GIB
MAX_CACHE_WORKERS = 16
# Jobs sharing this GPU (set by the jobs.py agent). Each fills its cache at the
# same time, so free memory alone over-promises: cap the cache to its share.
GPU_JOBS_ENV = "MGSA_GPU_JOBS"


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


def _container_memory() -> Optional[Tuple[int, int]]:
    """(limit, usage) of this container's cgroup memory limit, or None if unlimited.

    Containers (e.g. vast.ai) report the whole host's RAM through psutil; the
    cgroup limit is what actually triggers the OOM killer."""
    for limit_path, usage_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            with open(limit_path) as handle:
                limit = handle.read().strip()
            with open(usage_path) as handle:
                usage = int(handle.read().strip())
        except (OSError, ValueError):
            continue
        if limit == "max" or int(limit) >= 1 << 60:
            return None
        return int(limit), usage
    return None


def ram_cache_budget() -> int:
    """Bytes of RAM the epoch cache may use, leaving a margin under the tighter
    of the machine's available memory and the container limit."""
    memory = psutil.virtual_memory()
    free, total = int(memory.available), int(memory.total)
    container = _container_memory()
    if container is not None:
        limit, usage = container
        free, total = min(free, limit - usage), min(total, limit)
    return max(0, free - max(RAM_RESERVE_MIN, int(RAM_RESERVE_FRACTION * total)))


def gpu_cache_budget(device: torch.device) -> int:
    """Bytes of GPU memory the epoch cache may use, leaving room for training,
    within this job's share of the GPU when ``MGSA_GPU_JOBS`` jobs share it."""
    if device.type != "cuda":
        return 0
    free, total = torch.cuda.mem_get_info(device)
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    share = total // max(1, int(os.environ.get(GPU_JOBS_ENV, "1")))
    available = min(int(free), share - torch.cuda.memory_allocated(device))
    return max(0, available - max(GPU_RESERVE_MIN, int(GPU_RESERVE_FRACTION * share)))


def usable_cpu_count() -> int:
    """CPUs this process may really use: its affinity, capped by a container CPU quota.

    Containers (e.g. vast.ai) often show every host core while a cgroup quota
    limits how many run at once; oversubscribing them slows OpenMP badly."""
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        count = os.cpu_count() or 1
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            with open(path) as handle:
                fields = handle.read().split()
            if path.endswith("cpu.max"):
                quota, period = fields[0], fields[1]
            else:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as handle:
                    quota, period = fields[0], handle.read().strip()
        except (OSError, IndexError):
            continue
        if quota not in ("max", "-1") and int(period) > 0:
            count = min(count, max(1, int(quota) // int(period)))
        break
    return max(1, count)


def _cache_workers() -> int:
    return max(1, min(usable_cpu_count(), MAX_CACHE_WORKERS))


def _batch_bytes(batch: DateBatch) -> int:
    tensors = [
        batch.stock_features, batch.market_features, batch.labels, batch.labels_raw,
        batch.label_mask, batch.stock_mask, batch.intraday_market_features,
    ]
    return sum(t.element_size() * t.numel() for t in tensors if t is not None)


class WholeDateLoader:
    """Deterministic iterator; it never creates random stock mini-batches.

    With ``cache=True`` an Alpha158-only ``SequenceDataset`` first tries a row
    table: every (date, security) row its samples read is normalized once (8
    overlapping history windows share each row) and kept on the GPU within
    ``gpu_cache_budget``, else in RAM within ``ram_cache_budget``; each batch
    gathers ``[N, lookback, F]`` from it. Otherwise, or when the table fits
    neither, every date's batch is built once (in parallel threads) and reused
    each epoch: on the GPU while ``gpu_cache_budget`` allows, then in RAM while
    ``ram_cache_budget`` allows; dates beyond both are rebuilt every epoch as
    without the cache. Batches and their order are unchanged, so training
    results are identical either way."""

    date_batch_size = 1

    def __init__(
        self,
        dataset: Any,
        shuffle: bool = False,
        seed: int = 0,
        device: Any = "cpu",
        cache: bool = False,
    ) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.epoch = 0
        self.cache = bool(cache)
        self._cached: Optional[List[Optional[DateBatch]]] = None
        self._table: Optional[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def _build(self, position: int, device: torch.device) -> DateBatch:
        value = self.dataset[int(position)]
        if isinstance(value, DateBatch):
            return value if value.stock_features.device == device else move_date_batch(value, device)
        return adapt_sequence_sample(value, device)

    def _fill_table(self) -> bool:
        """Build the row table (see the class docstring); False if it does not apply or fit."""
        dataset = self.dataset
        if not isinstance(dataset, SequenceDataset) or dataset.layout.stock_families != ("alpha158",):
            return False
        started = time.monotonic()
        rows, index = dataset.stock_row_index()
        features = len(dataset.artifacts["alpha158"].feature_names)
        entries = sum(item.size for item in index)
        # rows (float32 + observed flag) + int64 indices; per-date labels/masks/market stay on self.device
        size = rows.shape[0] * (4 * features + 1) + entries * 8
        gpu_budget = gpu_cache_budget(self.device) - len(index) * 4096 - entries // 8 * 13
        if size <= gpu_budget:
            place = self.device
        elif size <= ram_cache_budget():
            place = torch.device("cpu")
        else:
            note("row table needs {:.1f} GiB, more than the GPU or RAM budget: caching whole dates".format(size / GIB))
            return False
        table = torch.empty((rows.shape[0], features), dtype=torch.float32, device=place)
        observed = torch.empty(rows.shape[0], dtype=torch.bool, device=place)
        starts = np.flatnonzero(np.diff(rows[:, 0], prepend=-1))
        ends = np.append(starts[1:], rows.shape[0])
        with ThreadPoolExecutor(_cache_workers()) as pool:
            built = pool.map(lambda span: dataset.stock_rows(int(rows[span[0], 0]), rows[span[0]:span[1], 1]), zip(starts, ends))
            for start, end, (values, flags) in zip(starts, ends, progress(built, desc="row table", unit="date", total=len(starts), leave=False)):
                table[start:end] = torch.from_numpy(values)
                observed[start:end] = torch.from_numpy(np.ascontiguousarray(flags))
            light = list(pool.map(lambda position: adapt_sequence_sample(dataset.sample(position, stock=False), self.device), range(len(dataset))))
        self._cached = light
        self._table = (table, observed, [torch.from_numpy(item).to(place) for item in index])
        note(
            "row table: {} rows for {} dates in {}, {:.1f} GiB on {} ({:.1f}x fewer rows than whole dates)".format(
                rows.shape[0], len(index), duration_text(time.monotonic() - started), size / GIB,
                "GPU" if place.type == "cuda" else place.type, entries / max(1, rows.shape[0]),
            )
        )
        return True

    def _fill_cache(self) -> List[Optional[DateBatch]]:
        started = time.monotonic()
        gpu_left = gpu_cache_budget(self.device)
        ram_left = ram_cache_budget()
        cpu = torch.device("cpu")
        cached: List[Optional[DateBatch]] = []
        sizes = {"GPU": 0, "RAM": 0}
        with ThreadPoolExecutor(_cache_workers()) as pool:
            built = pool.map(lambda position: self._build(position, cpu), range(len(self.dataset)))
            for batch in progress(built, desc="cache dates", unit="date", total=len(self.dataset), leave=False):
                size = _batch_bytes(batch)
                if size <= gpu_left:
                    gpu_left -= size
                    sizes["GPU"] += size
                    cached.append(move_date_batch(batch, self.device))
                elif size <= ram_left:
                    ram_left -= size
                    sizes["RAM"] += size
                    cached.append(batch)
                else:
                    cached.append(None)
        rebuilt = sum(1 for batch in cached if batch is None)
        note(
            "cached {} dates in {}: {:.1f} GiB on GPU, {:.1f} GiB in RAM{}".format(
                len(cached) - rebuilt, duration_text(time.monotonic() - started), sizes["GPU"] / GIB, sizes["RAM"] / GIB,
                ", {} rebuilt each epoch (memory limit)".format(rebuilt) if rebuilt else "",
            )
        )
        return cached

    def __iter__(self) -> Iterator[DateBatch]:
        positions = np.arange(len(self.dataset), dtype="int64")
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(positions)
        if self.cache and self._cached is None and not self._fill_table():
            self._cached = self._fill_cache()
        for position in positions:
            batch = self._cached[position] if self._cached is not None else None
            if self._table is not None:
                table, observed, index = self._table
                rows = index[position]
                yield replace(batch, stock_features=table[rows].to(self.device), stock_mask=observed[rows].to(self.device))
            elif batch is None:
                yield self._build(int(position), self.device)
            elif batch.stock_features.device != self.device:
                yield move_date_batch(batch, self.device)
            else:
                yield batch
