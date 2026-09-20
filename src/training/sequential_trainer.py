import csv
import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from src.utils.config import Config
from src.training.losses import LossComputer
from src.utils.checkpoint import save_checkpoint, load_checkpoint


class SequentialTrainer:
    """Trainer for multi-turn sequential reasoning models (Phase 4).

    Handles batching across (sequences x turns), computes per-turn and aggregate
    losses, evaluates per-turn accuracy curves (turn 1 vs turn 2+), and manages
    checkpointing and metrics logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device,
        drive_path: str | Path | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.drive_path = drive_path

        self.loss_fn = LossComputer(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        if config.training.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.epochs,
            )
        else:
            self.scheduler = None

        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = -1.0
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
        """Train for a single epoch across sequential batches."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        turn_correct: dict[int, int] = {}
        turn_counts: dict[int, int] = {}
        component_losses: dict[str, float] = {}

        for batch in self.train_loader:
            facts_seq = batch["facts"].to(self.device)
            fact_mask_seq = batch["fact_mask"].to(self.device)
            turn_mask = batch["turn_mask"].to(self.device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(self.device)

            self.optimizer.zero_grad()
            all_logits, all_infos = self.model(facts_seq, fact_mask_seq, turn_mask)

            B, max_turns, _ = all_logits.shape
            batch_loss = torch.tensor(0.0, device=self.device)
            valid_turns_in_batch = 0

            for t in range(max_turns):
                mask_t = turn_mask[:, t]
                valid_indices = (mask_t > 0).nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue

                logits_t = all_logits[valid_indices, t]
                labels_t = labels[valid_indices, t]
                info_t = all_infos[t]

                # Filter tensor components in info for valid sequences
                sliced_info: dict[str, Any] = {}
                for k, v in info_t.items():
                    if isinstance(v, Tensor) and v.shape[0] == B:
                        sliced_info[k] = v[valid_indices]
                    else:
                        sliced_info[k] = v

                loss_t, breakdown_t = self.loss_fn(logits_t, labels_t, sliced_info)
                n_valid = len(valid_indices)
                batch_loss = batch_loss + loss_t * n_valid
                valid_turns_in_batch += n_valid

                preds_t = logits_t.argmax(dim=-1)
                corr_t = (preds_t == labels_t).sum().item()
                turn_correct[t] = turn_correct.get(t, 0) + corr_t
                turn_counts[t] = turn_counts.get(t, 0) + n_valid

                for k, v in breakdown_t.items():
                    component_losses[k] = component_losses.get(k, 0.0) + v * n_valid

            if valid_turns_in_batch > 0:
                normalized_loss = batch_loss / valid_turns_in_batch
                normalized_loss.backward()

                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip,
                    )
                self.optimizer.step()

                total_loss += batch_loss.item()
                total_samples += valid_turns_in_batch

        total_correct = sum(turn_correct.values())
        mean_metrics: dict[str, float] = {
            "loss": total_loss / max(1, total_samples),
            "acc": total_correct / max(1, total_samples),
        }
        for t in sorted(turn_counts.keys()):
            mean_metrics[f"acc_turn_{t+1}"] = turn_correct[t] / max(1, turn_counts[t])
        for k, v in component_losses.items():
            mean_metrics[k] = v / max(1, total_samples)

        return mean_metrics

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate sequential model on a DataLoader without updating gradients."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        turn_correct: dict[int, int] = {}
        turn_counts: dict[int, int] = {}

        with torch.no_grad():
            for batch in loader:
                facts_seq = batch["facts"].to(self.device)
                fact_mask_seq = batch["fact_mask"].to(self.device)
                turn_mask = batch["turn_mask"].to(self.device)
                labels = (batch["labels"] if "labels" in batch else batch["label"]).to(self.device)

                all_logits, all_infos = self.model(facts_seq, fact_mask_seq, turn_mask)
                B, max_turns, _ = all_logits.shape

                for t in range(max_turns):
                    mask_t = turn_mask[:, t]
                    valid_indices = (mask_t > 0).nonzero(as_tuple=True)[0]
                    if len(valid_indices) == 0:
                        continue

                    logits_t = all_logits[valid_indices, t]
                    labels_t = labels[valid_indices, t]
                    info_t = all_infos[t]

                    sliced_info: dict[str, Any] = {}
                    for k, v in info_t.items():
                        if isinstance(v, Tensor) and v.shape[0] == B:
                            sliced_info[k] = v[valid_indices]
                        else:
                            sliced_info[k] = v

                    loss_t, _ = self.loss_fn(logits_t, labels_t, sliced_info)
                    n_valid = len(valid_indices)

                    preds_t = logits_t.argmax(dim=-1)
                    corr_t = (preds_t == labels_t).sum().item()
                    turn_correct[t] = turn_correct.get(t, 0) + corr_t
                    turn_counts[t] = turn_counts.get(t, 0) + n_valid

                    total_loss += loss_t.item() * n_valid
                    total_samples += n_valid

        total_correct = sum(turn_correct.values())
        eval_metrics: dict[str, float] = {
            "loss": total_loss / max(1, total_samples),
            "acc": total_correct / max(1, total_samples),
        }
        for t in sorted(turn_counts.keys()):
            eval_metrics[f"acc_turn_{t+1}"] = turn_correct[t] / max(1, turn_counts[t])

        return eval_metrics

    def _maybe_save_checkpoints(self, epoch: int, val_metrics: dict[str, float]) -> None:
        """Save best.pt and last.pt checkpoints."""
        val_acc = val_metrics.get("acc", 0.0)

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
