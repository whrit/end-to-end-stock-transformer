"""Whole-date neural training core, independent of rolling orchestration."""

import math
import time
from typing import Any, Dict, List, Mapping, Optional

import torch

from market_guided_stock_alpha.modeling.transport import (
    DateBatch,
    forward_date_batch,
    move_date_batch,
)
from market_guided_stock_alpha.progress import duration_text, metric as show_metric, note, progress, reset_metrics
from market_guided_stock_alpha.training.checkpoint import CheckpointManager
from market_guided_stock_alpha.training.config import TrainingConfig, create_optimizer
from market_guided_stock_alpha.training.losses import build_loss
from market_guided_stock_alpha.training.metrics import (
    aggregate_validation_losses,
    daily_rank_ic,
)
from market_guided_stock_alpha.training.seed import resolve_device, seed_everything


def _loss_error(prediction_date: Any) -> str:
    return "date {} has insufficient or non-finite training loss".format(prediction_date.date())


def _raise_first_non_finite(checks: List[torch.Tensor], dates: List[Any]) -> None:
    """Raise for the first step whose loss or gradients were non-finite (one device sync)."""
    if not checks:
        return
    flags = torch.stack(checks).cpu().tolist()
    for (loss_finite, gradients_finite), prediction_date in zip(flags, dates):
        if not loss_finite:
            raise FloatingPointError(_loss_error(prediction_date))
        if not gradients_finite:
            raise FloatingPointError("non-finite model gradient")


class Trainer:
    """One optimizer step per complete prediction-date cross-section."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        device: Any = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = resolve_device(device)
        seed_everything(config.seed, config.deterministic)
        self.model.to(self.device)
        self.optimizer = create_optimizer(model, config.optimizer)
        self.loss_function = build_loss(config.loss)
        self.checkpoints = checkpoint_manager

    def _loss(self, predictions: torch.Tensor, batch: DateBatch, check_finite: bool = True) -> Mapping[str, Any]:
        result = self.loss_function(
            predictions,
            batch.labels,
            batch.label_mask,
            batch.labels_raw.to(device=predictions.device, dtype=predictions.dtype),
        )
        loss = result.get("loss")
        if (
            int(result.get("valid_count", 0)) < self.config.minimum_valid_labels
            or loss is None
            or (check_finite and not bool(torch.isfinite(loss)))
        ):
            raise FloatingPointError(_loss_error(batch.prediction_date))
        return result

    def train_epoch(self, loader: Any, epoch: int) -> Dict[str, Any]:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        self.model.train()
        rows: List[Dict[str, Any]] = []
        # Device-side per date: loss, loss finite, gradients finite. Reading them back
        # every step would make the CPU wait for backward and the optimizer step each
        # time; checked once at epoch end instead (before validation, so no checkpoint
        # is ever saved from a non-finite step), with the same error and date.
        losses: List[torch.Tensor] = []
        checks: List[torch.Tensor] = []
        dates: List[Any] = []
        for cpu_batch in progress(
            loader, desc="epoch {} train".format(epoch), unit="date", leave=False
        ):
            batch = move_date_batch(cpu_batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            predictions = forward_date_batch(self.model, batch)
            try:
                result = self._loss(predictions, batch, check_finite=False)
            except FloatingPointError:
                _raise_first_non_finite(checks, dates)  # an earlier bad step explains this one
                raise
            loss = result["loss"]
            loss.backward()
            gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
            gradients_finite = (
                torch.stack([torch.isfinite(g).all() for g in gradients]).all()
                if gradients else torch.ones((), dtype=torch.bool, device=loss.device)
            )
            checks.append(torch.stack((torch.isfinite(loss.detach()), gradients_finite)))
            losses.append(loss.detach())
            dates.append(batch.prediction_date)
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.config.optimizer.gradient_clip_value
            )
            self.optimizer.step()
            rows.append({"valid_count": int(result["valid_count"])})
        _raise_first_non_finite(checks, dates)
        for row, value in zip(rows, torch.stack(losses).cpu().tolist() if losses else []):
            row["loss"] = float(value)
        if not rows:
            raise RuntimeError("training loader is empty")
        total = sum(row["valid_count"] for row in rows)
        return {
            "epoch": int(epoch),
            "train_loss": sum(row["loss"] * row["valid_count"] for row in rows) / total,
            "mean_daily_train_loss": sum(row["loss"] for row in rows) / len(rows),
            "train_date_count": len(rows),
            "train_observation_count": total,
            "learning_rate": self.config.optimizer.learning_rate,
        }

    def validate(self, loader: Any) -> Dict[str, Any]:
        self.model.eval()
        rows: List[Dict[str, Any]] = []
        predictions_by_date: List[Dict[str, Any]] = []
        with torch.no_grad():
            for cpu_batch in progress(loader, desc="validate", unit="date", leave=False):
                batch = move_date_batch(cpu_batch, self.device)
                predictions = forward_date_batch(self.model, batch)
                result = self._loss(predictions, batch)
                rows.append(
                    {
                        "loss": float(result["loss"].detach().cpu()),
                        "valid_count": int(result["valid_count"]),
                        "rank_ic": daily_rank_ic(
                            predictions.detach().cpu().numpy(),
                            batch.labels_raw.detach().cpu().numpy(),
                            batch.label_mask.detach().cpu().numpy(),
                        ),
                    }
                )
                predictions_by_date.append(
                    {
                        "prediction_date": str(batch.prediction_date.date()),
                        "security_codes": batch.security_codes,
                        "predictions": predictions.detach().cpu().numpy(),
                        "labels_raw": batch.labels_raw.detach().cpu().numpy(),
                        "label_mask": batch.label_mask.detach().cpu().numpy(),
                    }
                )
        aggregate = aggregate_validation_losses(rows)
        aggregate["predictions_by_date"] = predictions_by_date
        return aggregate

    def fit(self, train_loader: Any, valid_loader: Any) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        maximize = self.config.primary_metric == "mean_rank_ic"
        best_metric = float("-inf") if maximize else float("inf")
        best_epoch: Optional[int] = None
        without_improvement = 0
        stopped_early = False
        reset_metrics()
        epochs = progress(
            range(1, self.config.epochs + 1), desc="epochs", unit="epoch", leave=False
        )
        for epoch in epochs:
            started = time.monotonic()
            train = self.train_epoch(train_loader, epoch)
            validation = self.validate(valid_loader)
            validation_loss = float(validation["validation_loss"])
            metric = float(validation[self.config.primary_metric])
            if not math.isfinite(validation_loss) or not math.isfinite(metric):
                raise FloatingPointError("validation selection metric is non-finite")
            row = dict(train)
            row.update(
                {
                    key: value
                    for key, value in validation.items()
                    if key != "predictions_by_date"
                }
            )
            history.append(row)
            improved = metric > best_metric if maximize else metric < best_metric
            if improved:
                best_metric = metric
                best_epoch = epoch
                without_improvement = 0
            else:
                without_improvement += 1
            seconds = time.monotonic() - started
            schedule = self.config.early_stopping
            patience = (
                "no gain {}/{}".format(without_improvement, schedule.patience) if schedule.enabled else ""
            )
            epochs.set_postfix(
                loss="{:.4f}".format(train["train_loss"]),
                valid="{:.4f}".format(metric),
                best="{:.4f}@{}".format(best_metric, best_epoch),
                last=duration_text(seconds),
                **({"patience": "{}/{}".format(without_improvement, schedule.patience)} if schedule.enabled else {})
            )
            show_metric("train loss", train["train_loss"], higher_is_better=False)
            show_metric("val loss", validation_loss, higher_is_better=False)
            show_metric("val " + self.config.primary_metric, metric, higher_is_better=maximize)
            show_metric("epoch seconds", seconds, higher_is_better=False, fmt="{:.1f}")
            short = {"mean_rank_ic": "rIC"}.get(self.config.primary_metric, self.config.primary_metric)
            note(
                ("    epoch {:>3}/{:<3}".format(epoch, self.config.epochs), "bold"),
                ("{:>8}".format(duration_text(seconds)), "cyan"),
                ("  train ", "dim"),
                "{:.4f}".format(train["train_loss"]),
                ("  val ", "dim"),
                "{:.4f}".format(validation_loss),
                ("  {} ".format(short), "dim"),
                ("{:.4f}".format(metric), "bold green" if improved else "white"),
                ("  ★ new best" if improved else "  best {:.4f}@{}".format(best_metric, best_epoch),
                 "green" if improved else "dim"),
                ("  {}".format(patience) if patience and not improved else "", "dim"),
                detail=True,
            )
            if self.checkpoints is not None and (
                epoch % self.config.checkpoint_every == 0 or improved
            ):
                self.checkpoints.save(
                    epoch,
                    metric,
                    self.model,
                    self.optimizer,
                    history,
                    improved,
                )
            if (
                schedule.enabled
                and epoch >= schedule.minimum_epochs
                and without_improvement >= schedule.patience
            ):
                stopped_early = True
                break
        epochs.close()
        return {
            "history": history,
            "best_epoch": best_epoch,
            "primary_metric": self.config.primary_metric,
            "best_validation_metric": best_metric,
            "stopped_early": stopped_early,
        }
