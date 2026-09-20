"""Phase 3 — Adaptive Resolution: per-step halt gate (ACT-style).

The HaltGate reads the current global workspace state (pooled across active
slots) and outputs a scalar halt probability h_t ∈ (0, 1).  The LatentWorkspace
accumulates these probabilities to decide when each example is "done."

Design rationale
----------------
- Halting is a *global* decision ("has the workspace computed enough?"), not a
  per-slot decision.  Per-slot routing is handled by ActivityGate (Phase 2).
- Input is the activity-weighted mean pool of V_t — the same representation the
  decoder will ultimately read — so the gate sees the same signal as the answer.
- Two-layer MLP with ReLU keeps the gate lightweight (~32*32 + 32 = 1056 params).
"""

import torch
import torch.nn as nn
from torch import Tensor


class HaltGate(nn.Module):
    """Computes a scalar halt probability from the pooled workspace state.

    Architecture:
        pool(V_t, A_t)  →  [B, d]
        Linear(d, hidden) → ReLU → Linear(hidden, 1) → Sigmoid
        output: h_t  ∈ (0, 1)  shape [B]

    The halt probability is consumed by LatentWorkspace._forward_graph_adaptive
    to accumulate per-example ponder weights (ACT remainder distribution).

    Args:
        latent_dim:      Dimensionality of each slot state (d).
        halt_hidden_dim: Width of the hidden layer in the halt MLP.
        eps:             Small constant for safe division in activity-weighted pool.
    """

    def __init__(self, latent_dim: int, halt_hidden_dim: int = 32, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, halt_hidden_dim),
            nn.ReLU(),
            nn.Linear(halt_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, V: Tensor, A: Tensor | None = None) -> Tensor:
        """Compute halt probability from current slot states.

        Args:
            V: Tensor[B, N, d]  current slot states
            A: Tensor[B, N]     slot activity weights (optional, default uniform)

        Returns:
            h: Tensor[B]        halt probability per example in (0, 1)
        """
        B, N, d = V.shape

        if A is None:
            A = torch.ones((B, N), device=V.device, dtype=V.dtype)

        # Activity-weighted mean pool: [B, d]
        weights = A.unsqueeze(-1)                                  # [B, N, 1]
        pool = (V * weights).sum(dim=1)                            # [B, d]
        sum_w = A.sum(dim=1, keepdim=True).clamp(min=self.eps)    # [B, 1]
        pool = pool / sum_w                                        # [B, d]

        h = self.mlp(pool)   # [B, 1]
        return h.squeeze(-1)  # [B]
