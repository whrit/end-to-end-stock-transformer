"""Mask-aware losses used by the formal 03 loss experiments."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn


LOSS_NAMES = ("mse", "ccc", "listfold", "rpl", "rsr")


@dataclass(frozen=True)
class LossConfig:
    name: str = "mse"
    epsilon: float = 1e-8
    alpha: Optional[float] = None
    gamma: float = 1.0
    minimum_count: int = 30
    minimum_prediction_norm: float = 1e-6
    pairwise_chunk_size: int = 256

    def __post_init__(self) -> None:
        if self.name not in LOSS_NAMES:
            raise ValueError("loss name must be one of {}".format(", ".join(LOSS_NAMES)))
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("loss epsilon must be finite and positive")
        resolved_alpha = (
            1.0 if self.name == "rsr" else 0.5
        ) if self.alpha is None else float(self.alpha)
        object.__setattr__(self, "alpha", resolved_alpha)
        if not math.isfinite(resolved_alpha) or not math.isfinite(self.gamma):
            raise ValueError("loss alpha/gamma must be finite")
        if self.name == "rpl" and not 0.0 <= resolved_alpha <= 1.0:
            raise ValueError("RPL alpha must be in [0,1]")
        if self.name == "rpl" and self.gamma < 0.0:
            raise ValueError("RPL gamma cannot be negative")
        if type(self.minimum_count) is not int or self.minimum_count < 2:
            raise ValueError("minimum_count must be at least two")
        if self.minimum_prediction_norm <= 0.0:
            raise ValueError("minimum_prediction_norm must be positive")
        if self.name == "rsr" and resolved_alpha <= 0.0:
            raise ValueError("RSR alpha must be positive")
        if type(self.pairwise_chunk_size) is not int or self.pairwise_chunk_size <= 0:
            raise ValueError("pairwise_chunk_size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_loss_config(value: Optional[Any] = None) -> LossConfig:
    if value is None:
        return LossConfig()
    if isinstance(value, LossConfig):
        return value
    if isinstance(value, str):
        return LossConfig(name=value)
    if not isinstance(value, Mapping):
        raise ValueError("loss config must be a name, mapping, or LossConfig")
    document = dict(value)
    allowed = set(LossConfig.__dataclass_fields__)
    unexpected = set(document).difference(allowed)
    if unexpected:
        raise ValueError("unexpected loss config keys: {}".format(sorted(unexpected)))
    return LossConfig(**document)


def _masked_values(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_mask: torch.Tensor,
    loss_name: str,
    ranking_labels: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    values = [predictions, labels, label_mask]
    if ranking_labels is not None:
        values.append(ranking_labels)
    if any(value.ndim != 1 for value in values):
        raise ValueError("loss inputs must be one-dimensional")
    if any(value.shape != predictions.shape for value in values[1:]):
        raise ValueError("loss input shapes must match")
    if label_mask.dtype != torch.bool:
        raise ValueError("label_mask must be boolean")
    mask = label_mask & torch.isfinite(predictions) & torch.isfinite(labels)
    if ranking_labels is not None:
        mask &= torch.isfinite(ranking_labels)
    result: Dict[str, Any] = {
        "loss": None,
        "loss_name": loss_name,
        "components": {},
        "valid_count": int(mask.sum().item()),
        "total_count": int(predictions.numel()),
        "mask": mask,
    }
    return (
        result,
        predictions[mask],
        labels[mask],
        ranking_labels[mask] if ranking_labels is not None else None,
    )


class MaskedMSELoss(nn.Module):
    loss_name = "mse"

    def forward(self, predictions, labels, label_mask, ranking_labels=None):
        result, prediction, target, _ranking = _masked_values(
            predictions, labels, label_mask, self.loss_name
        )
        if result["valid_count"]:
            loss = torch.mean((prediction - target).square())
            result["loss"] = loss
            result["components"] = {"mse": loss}
        return result


class MaskedCCCLoss(nn.Module):
    loss_name = "ccc"

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, predictions, labels, label_mask, ranking_labels=None):
        result, prediction, target, _ranking = _masked_values(
            predictions, labels, label_mask, self.loss_name
        )
        if not result["valid_count"]:
            return result
        prediction_centered = prediction - prediction.mean()
        target_centered = target - target.mean()
        covariance = torch.mean(prediction_centered * target_centered)
        denominator = (
            torch.mean(prediction_centered.square())
            + torch.mean(target_centered.square())
            + (prediction.mean() - target.mean()).square()
            + self.epsilon
        )
        ccc = 2.0 * covariance / denominator
        loss = 1.0 - ccc
        result["loss"] = loss
        result["components"] = {"ccc": ccc, "ccc_loss": loss}
        return result


class MaskedListFoldLoss(nn.Module):
    """Historical exponential ListFold with clipped-label tie handling."""

    loss_name = "listfold"

    def forward(self, predictions, labels, label_mask, ranking_labels=None):
        result, prediction, target, ranking = _masked_values(
            predictions, labels, label_mask, self.loss_name, ranking_labels
        )
        count = int(result["valid_count"])
        pair_count = count // 2
        if pair_count <= 0:
            return result
        tie_values = target if ranking is None else ranking
        tie_order = torch.argsort(tie_values, descending=True, stable=True)
        order = tie_order[torch.argsort(target[tie_order], descending=True, stable=True)]
        if count % 2:
            order = torch.cat((order[:pair_count], order[-pair_count:]))
        scores = prediction[order].double()
        active_total = int(scores.numel())
        starts = torch.arange(pair_count, device=scores.device, dtype=torch.long)
        ends = active_total - starts
        positive_shift = scores.max().detach()
        positive_prefix = torch.cat(
            (scores.new_zeros(1), torch.cumsum(torch.exp(scores - positive_shift), dim=0))
        )
        positive_sums = positive_prefix[ends] - positive_prefix[starts]
        negative_shift = (-scores).max().detach()
        negative_prefix = torch.cat(
            (scores.new_zeros(1), torch.cumsum(torch.exp(-scores - negative_shift), dim=0))
        )
        negative_sums = negative_prefix[ends] - negative_prefix[starts]
        log_product = (
            torch.log(positive_sums)
            + positive_shift
            + torch.log(negative_sums)
            + negative_shift
        )
        active_counts = (ends - starts).to(dtype=torch.float64)
        diagonal_fraction = active_counts * torch.exp(-log_product)
        log_denominator = log_product + torch.log1p(-diagonal_fraction)
        differences = scores[starts] - scores[ends - 1]
        raw_loss = torch.mean(log_denominator - differences)
        random_baseline = torch.mean(torch.log(active_counts * (active_counts - 1.0)))
        loss = (raw_loss / random_baseline).to(dtype=predictions.dtype)
        result["loss"] = loss
        result["components"] = {
            "listfold_raw": raw_loss.to(dtype=predictions.dtype),
            "listfold_random_baseline": random_baseline.to(dtype=predictions.dtype),
            "listfold_normalized": loss,
        }
        result["fold_count"] = pair_count
        return result


def _average_rank_percentiles(values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        sorted_values, order = torch.sort(values)
        _unique, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts
        averages = (starts + ends - 1).to(dtype=values.dtype) / 2.0
        sorted_ranks = torch.repeat_interleave(averages, counts)
        ranks = torch.empty_like(sorted_ranks)
        ranks[order] = sorted_ranks
        return ranks / float(max(1, int(values.numel()) - 1))


class MaskedRPLLoss(nn.Module):
    loss_name = "rpl"

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        epsilon: float = 1e-8,
        minimum_count: int = 30,
        minimum_prediction_norm: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.minimum_count = int(minimum_count)
        self.minimum_prediction_norm = float(minimum_prediction_norm)

    def forward(self, predictions, labels, label_mask, ranking_labels=None):
        result, prediction, target, _ranking = _masked_values(
            predictions, labels, label_mask, self.loss_name
        )
        count = int(result["valid_count"])
        if not count:
            return result
        residual = prediction - target
        mse = torch.mean(residual.square())
        prediction_centered = prediction - prediction.mean()
        target_centered = target - target.mean()
        prediction_norm = torch.linalg.vector_norm(prediction_centered)
        target_norm = torch.linalg.vector_norm(target_centered)
        if (
            count < self.minimum_count
            or float(prediction_norm.detach()) < self.minimum_prediction_norm
            or float(target_norm.detach()) < self.minimum_prediction_norm
        ):
            result["loss"] = mse
            result["components"] = {"mse_fallback": mse}
            result["fallback_to_mse"] = True
            return result
        percentiles = _average_rank_percentiles(target)
        weights = 1.0 + self.gamma * (2.0 * percentiles - 1.0).square()
        weights = weights / weights.mean()
        weighted_mse = torch.mean(weights * residual.square())
        correlation = torch.sum(prediction_centered * target_centered) / (
            prediction_norm * target_norm + self.epsilon
        )
        ic_loss = 1.0 - correlation
        loss = self.alpha * weighted_mse + (1.0 - self.alpha) * ic_loss
        result["loss"] = loss
        result["components"] = {
            "weighted_mse": weighted_mse,
            "ic": correlation,
            "ic_loss": ic_loss,
        }
        result["fallback_to_mse"] = False
        return result


class MaskedRSRLoss(nn.Module):
    loss_name = "rsr"

    def __init__(self, alpha: float = 1.0, pairwise_chunk_size: int = 256) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.pairwise_chunk_size = int(pairwise_chunk_size)

    def forward(self, predictions, labels, label_mask, ranking_labels=None):
        result, prediction, target, _ranking = _masked_values(
            predictions, labels, label_mask, self.loss_name
        )
        count = int(result["valid_count"])
        if not count:
            return result
        mse = torch.mean((prediction - target).square())
        pairwise_sum = prediction.new_zeros(())
        comparable_count = 0
        for start in range(0, count, self.pairwise_chunk_size):
            stop = min(count, start + self.pairwise_chunk_size)
            prediction_difference = prediction[start:stop, None] - prediction[None, :]
            label_difference = target[start:stop, None] - target[None, :]
            comparable = label_difference != 0
            products = prediction_difference * label_difference
            pairwise_sum = pairwise_sum + torch.relu(-products)[comparable].sum()
            comparable_count += int(comparable.sum().item())
        pairwise = pairwise_sum / float(comparable_count) if comparable_count else pairwise_sum
        loss = mse + self.alpha * pairwise
        result["loss"] = loss
        result["components"] = {"mse": mse, "pairwise": pairwise}
        result["comparable_pair_count"] = comparable_count
        return result


def build_loss(value: Optional[Any] = None) -> nn.Module:
    config = normalize_loss_config(value)
    if config.name == "mse":
        return MaskedMSELoss()
    if config.name == "ccc":
        return MaskedCCCLoss(config.epsilon)
    if config.name == "listfold":
        return MaskedListFoldLoss()
    if config.name == "rpl":
        return MaskedRPLLoss(
            config.alpha,
            config.gamma,
            config.epsilon,
            config.minimum_count,
            config.minimum_prediction_norm,
        )
    if config.name == "rsr":
        return MaskedRSRLoss(config.alpha, config.pairwise_chunk_size)
    raise AssertionError("validated loss was not constructed")
