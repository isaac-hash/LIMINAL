import torch
import torch.nn as nn
from torch import Tensor


class ActivityGate(nn.Module):
    """Computes per-slot activity scores A in [0, 1].

    Each slot i independently decides how 'active' it should be given its
    current state v_i.  The gate is a small 2-layer MLP followed by sigmoid:

        a_i = sigmoid(W_2 * ReLU(W_1 * v_i + b_1) + b_2)

    This output feeds into:
      - MessagePassingLayer: masks sender messages  (a_j * m_ij)
      - InvariantReadout   : weighted mean pooling  (a_i * v_i)
      - LossComputer       : sparsity + entropy reg (when activity.enabled)

    Args:
        latent_dim:      Dimensionality of each slot state (d).
        gate_hidden_dim: Width of the single hidden layer.
    """

    def __init__(self, latent_dim: int, gate_hidden_dim: int = 32):
        super().__init__()
        self.gate_hidden_dim = gate_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, V: Tensor) -> Tensor:
        """Compute activity scores for all slots.

        Args:
            V: Tensor[B, N, d]  current slot states

        Returns:
            A: Tensor[B, N]     activity gates in (0, 1)
        """
        # Apply gate MLP independently to each slot: [B, N, d] -> [B, N, 1]
        A = self.mlp(V)        # [B, N, 1]
        return A.squeeze(-1)   # [B, N]
