"""Phase 4 — Persistence: Carry latent state across related sequence turns.

The PersistenceGate computes per-slot blending factors between prior workspace
latent representations (V_prior) and freshly encoded slot representations (V_new):

    gate[i] = sigmoid(MLP([v_prior_i ; v_new_i]))      ∈ (0, 1) per slot
    V_0'[i] = gate[i] * v_prior_i + (1 - gate[i]) * v_new_i

Design rationale
----------------
- Per-slot gating: Different latent slots specialize in different knowledge.
  For instance, a central hub slot representing the running entity/budget can
  be preserved with a high gate, while other slots can re-encode new attributes.
- Smooth transition: Using negative bias initialization on the final linear layer
  ensures that by default at turn 0 or cold start, the gate leans towards fresh
  encodings (gate ≈ 0), preserving baseline stability until persistence is learned.
- Detached recurrence: To maintain bounded memory and avoid vanishing gradients
  over long sequences, gradients are detached between turns by default.
"""

from typing import Any
import torch
import torch.nn as nn
from torch import Tensor


class PersistenceGate(nn.Module):
    """Per-slot gating module for blending prior latent state with fresh encoding.

    Args:
        latent_dim:      Dimensionality of each slot state (d).
        gate_hidden_dim: Width of the hidden layer in the gate MLP (default 32).
        init_bias:       Initial bias for output linear layer (-2.0 leans toward V_new).
    """

    def __init__(
        self,
        latent_dim: int,
        gate_hidden_dim: int = 32,
        init_bias: float = -2.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.gate_hidden_dim = gate_hidden_dim

        linear2 = nn.Linear(gate_hidden_dim, 1)
        if linear2.bias is not None:
            nn.init.constant_(linear2.bias, init_bias)

        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, gate_hidden_dim),
            nn.ReLU(),
            linear2,
            nn.Sigmoid(),
        )

    def compute_gate(self, V_prior: Tensor, V_new: Tensor) -> Tensor:
        """Compute per-slot gate values in (0, 1).

        Args:
            V_prior: Tensor[B, N, d] prior slot representations from previous turn
            V_new:   Tensor[B, N, d] freshly encoded slot representations for current turn

        Returns:
            gates:   Tensor[B, N] per-slot gate values in (0, 1)
        """
        # Concatenate along slot feature dimension: [B, N, 2d]
        combined = torch.cat([V_prior, V_new], dim=-1)
        gates = self.mlp(combined).squeeze(-1)  # [B, N]
        return gates

    def forward(
        self,
        V_prior: Tensor,
        V_new: Tensor,
        return_gates: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Blend prior slot states with fresh encodings.

        Args:
            V_prior:      Tensor[B, N, d] prior slot representations
            V_new:        Tensor[B, N, d] fresh slot encodings
            return_gates: If True, returns (V_blended, gates); otherwise returns V_blended

        Returns:
            V_blended:    Tensor[B, N, d] blended initial latent slots
            gates:        Tensor[B, N] gate activations (only if return_gates=True)
        """
        gates = self.compute_gate(V_prior, V_new)  # [B, N]
        gates_expanded = gates.unsqueeze(-1)       # [B, N, 1]

        V_blended = gates_expanded * V_prior + (1.0 - gates_expanded) * V_new

        if return_gates:
            return V_blended, gates
        return V_blended
