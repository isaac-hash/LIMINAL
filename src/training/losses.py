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

        # Phase 3: ponder cost — penalise using more steps than necessary
        # L_ponder = lambda_ponder * mean(n_steps)
        if self.config.resolution.enabled and info is not None and "n_steps" in info:
            n_steps = info["n_steps"]   # Tensor[B], detached from workspace
            # Re-attach gradient path via ponder_weights if present so the loss
            # actually trains the halt gate.  n_steps in info is detached for
            # logging; we compute a differentiable version from ponder_weights.
            if "ponder_weights" in info:
                pw = info["ponder_weights"]                         # [B, T]
                T_actual = pw.shape[1]
                device = pw.device
                dtype = pw.dtype
                step_idx = torch.arange(1, T_actual + 1, device=device, dtype=dtype).unsqueeze(0)
                n_steps_diff = (pw * step_idx).sum(dim=1)           # [B], differentiable
            else:
                n_steps_diff = n_steps  # fallback (detached)

            loss_ponder = self.config.resolution.ponder_lambda * n_steps_diff.mean()
            total_loss = total_loss + loss_ponder
            breakdown["loss_ponder"] = loss_ponder.item()
            breakdown["mean_steps"] = n_steps.mean().item()
            breakdown["loss_total"] = total_loss.item()

        # Phase 7: write sparsity penalty (LearnedWriteController only)
        # Applied when external workspace is enabled and write_gate_mean is present.
        if (
            self.config.external.enabled
            and self.config.external.learned_gate
            and info is not None
            and info.get("write_gate_mean") is not None
        ):
            gate_mean = info["write_gate_mean"]
            loss_write_sparse = self.config.external.write_sparsity_lambda * gate_mean
            total_loss = total_loss + loss_write_sparse
            breakdown["loss_write_sparse"] = loss_write_sparse.item()
            breakdown["write_gate_mean"] = gate_mean.item()
            breakdown["loss_total"] = total_loss.item()

        return total_loss, breakdown

