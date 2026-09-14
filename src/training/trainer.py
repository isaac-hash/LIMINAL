import csv
import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.config import Config
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.training.losses import LossComputer


class Trainer:
    """Trainer managing training/eval loops, optimization, checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: LossComputer,
        config: Config,
        device: torch.device | str = "cpu",
        drive_path: str | Path | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.drive_path = drive_path

        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        if config.training.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.epochs,
                eta_min=config.training.lr * 0.01,
            )
        elif config.training.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, config.training.epochs // 3),
                gamma=0.5,
            )
        else:
            self.scheduler = None

        self.best_val_acc = -1.0
        self.best_val_loss = float("inf")
        self.metrics_log: list[dict[str, Any]] = []

    def train(self) -> list[dict[str, Any]]:
        """Run full training loop across all epochs."""
        for epoch in range(self.config.training.epochs):
            train_metrics = self._train_epoch(epoch)

            val_metrics = {}
            if (epoch + 1) % self.config.training.eval_every == 0 or (epoch + 1) == self.config.training.epochs:
                val_metrics = self.evaluate(self.val_loader)
                self._maybe_save_checkpoints(epoch, val_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            log_entry = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.metrics_log.append(log_entry)
            self._write_metrics_csv(log_entry)

        self._write_metrics_json()
        return self.metrics_log

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Train for a single epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        component_losses: dict[str, float] = {}

        for batch in self.train_loader:
            facts = batch["facts"].to(self.device)
            fact_mask = batch["fact_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits, info = self.model(facts, fact_mask)
            loss, breakdown = self.loss_fn(logits, labels, info)

            loss.backward()
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )
            self.optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += bs

            for k, v in breakdown.items():
                component_losses[k] = component_losses.get(k, 0.0) + v * bs

        mean_metrics = {
            "loss": total_loss / total_samples,
            "acc": correct / total_samples,
        }
        for k, v in component_losses.items():
            mean_metrics[k] = v / total_samples

        return mean_metrics

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate model on a DataLoader without updating gradients."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                facts = batch["facts"].to(self.device)
                fact_mask = batch["fact_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits, info = self.model(facts, fact_mask)
                loss, _ = self.loss_fn(logits, labels, info)

                bs = labels.size(0)
                total_loss += loss.item() * bs
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total_samples += bs

        return {
            "loss": total_loss / max(1, total_samples),
            "acc": correct / max(1, total_samples),
        }

    def _maybe_save_checkpoints(self, epoch: int, val_metrics: dict[str, float]) -> None:
        """Save best.pt and last.pt checkpoints."""
        val_acc = val_metrics.get("acc", 0.0)

        # Always save last.pt
        last_path = self.checkpoint_dir / "last.pt"
        drive_last = Path(self.drive_path) / "last.pt" if self.drive_path else None
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            metrics=val_metrics,
            path=last_path,
            config=self.config,
            drive_path=drive_last,
        )

        # Save best.pt on validation improvement
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            best_path = self.checkpoint_dir / "best.pt"
            drive_best = Path(self.drive_path) / "best.pt" if self.drive_path else None
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=val_metrics,
                path=best_path,
                config=self.config,
                drive_path=drive_best,
            )

    def _write_metrics_csv(self, log_entry: dict[str, Any]) -> None:
        csv_path = self.checkpoint_dir / "metrics.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_entry.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

    def _write_metrics_json(self) -> None:
        json_path = self.checkpoint_dir / "metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_log, f, indent=2)

    def resume_from(self, checkpoint_path: str | Path) -> int:
        """Resume model and optimizer states from checkpoint."""
        epoch, metrics, _ = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
        )
        self.best_val_acc = metrics.get("acc", -1.0)
        return epoch
