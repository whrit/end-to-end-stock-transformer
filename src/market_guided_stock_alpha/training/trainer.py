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
from market_guided_stock_alpha.progress import note, progress
from market_guided_stock_alpha.training.checkpoint import CheckpointManager
from market_guided_stock_alpha.training.config import TrainingConfig, create_optimizer
from market_guided_stock_alpha.training.losses import build_loss
from market_guided_stock_alpha.training.metrics import (
    aggregate_validation_losses,
    daily_rank_ic,
)
from market_guided_stock_alpha.training.seed import resolve_device, seed_everything


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

    def _loss(self, predictions: torch.Tensor, batch: DateBatch) -> Mapping[str, Any]:
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
            or not bool(torch.isfinite(loss))
        ):
            raise FloatingPointError(
                "date {} has insufficient or non-finite training loss".format(
                    batch.prediction_date.date()
                )
            )
        return result

    def train_epoch(self, loader: Any, epoch: int) -> Dict[str, Any]:
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        self.model.train()
        rows: List[Dict[str, Any]] = []
        for cpu_batch in progress(
            loader, desc="epoch {} train".format(epoch), unit="date", leave=False
        ):
            batch = move_date_batch(cpu_batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            predictions = forward_date_batch(self.model, batch)
            result = self._loss(predictions, batch)
            loss = result["loss"]
            loss.backward()
            for parameter in self.model.parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("non-finite model gradient")
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.config.optimizer.gradient_clip_value
            )
            self.optimizer.step()
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "valid_count": int(result["valid_count"]),
                }
            )
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
            epochs.set_postfix(
                loss="{:.4f}".format(train["train_loss"]),
                valid="{:.4f}".format(metric),
                best="{:.4f}@{}".format(best_metric, best_epoch),
            )
            note(
                ("    epoch {:>3}/{:<3}".format(epoch, self.config.epochs), "bold"),
                ("  train_loss ", "dim"),
                "{:.4f}".format(train["train_loss"]),
                ("  val_loss ", "dim"),
                "{:.4f}".format(validation_loss),
                ("  {} ".format(self.config.primary_metric), "dim"),
                ("{:.4f}".format(metric), "bold green" if improved else "white"),
                ("  ★ new best" if improved else "  best {:.4f} @ {}".format(best_metric, best_epoch),
                 "green" if improved else "dim"),
                ("  {:.0f}s".format(time.monotonic() - started), "cyan"),
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
            schedule = self.config.early_stopping
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
