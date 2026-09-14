from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from src.utils.config import Config


class LossComputer:
    """Computes total composite loss with modular loss breakdown."""

    def __init__(self, config: Config):
        self.config = config
        self.ce = nn.CrossEntropyLoss()

    def __call__(
        self,
        logits: Tensor,
        labels: Tensor,
        info: dict[str, Any] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Args:
            logits: Tensor[B, num_classes] model predictions
            labels: Tensor[B] ground truth class indices
            info: dict optional dictionary of intermediate states and gates
            
        Returns:
            total_loss: scalar Tensor
            breakdown: dict mapping loss component names to float values
        """
        loss_answer = self.ce(logits, labels)
        total_loss = loss_answer

        breakdown = {
            "loss_total": loss_answer.item(),
            "loss_answer": loss_answer.item(),
        }

        # Modular hooks for Phase 2+ (Activity regularization, halt penalty, etc.)
        if self.config.activity.enabled and info is not None and "A_final" in info:
            A_final = info["A_final"]
            # L_sparsity = ||A||_1 / N
            loss_sparse = self.config.activity.sparsity_lambda * A_final.mean()
            # Entropy regularizer: -p log p to encourage binary decisions
            eps = 1e-8
            p = A_final.clamp(eps, 1.0 - eps)
            entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()
            loss_entropy = self.config.activity.entropy_lambda * entropy

            total_loss = total_loss + loss_sparse + loss_entropy
            breakdown["loss_sparse"] = loss_sparse.item()
            breakdown["loss_entropy"] = loss_entropy.item()
            breakdown["loss_total"] = total_loss.item()

        return total_loss, breakdown
