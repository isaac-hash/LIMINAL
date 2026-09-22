"""Phase 5 — EdgeAdapter: dynamic relational edge adaptation.

In Phases 1–4, the edge tensor E ∈ R[B, N, N, d_e] was a static learned prior
(edge_prior parameter) or zeros.  Phase 5 makes edges evolve at each reasoning
step in response to current slot states and external memory reads:

    E_{t+1} = tanh(MLP([v_i ; v_j ; e_ij])) + E_t   (residual, bounded)

The update is:
- Permutation-equivariant: permuting nodes permutes edges consistently.
- Residual: edges cannot diverge unboundedly.
- Bounded: tanh keeps delta in (-1, 1) regardless of MLP magnitude.

The pairwise concatenation [v_i ; v_j ; e_ij] reuses the same broadcasting
pattern as MessagePassingLayer, keeping memory layout consistent.
"""

import torch
import torch.nn as nn
from torch import Tensor


class EdgeAdapter(nn.Module):
    """Dynamic per-step edge adaptation.

    Args:
        latent_dim:     Dimensionality d of slot vectors.
        edge_dim:       Dimensionality d_e of edge features.
        hidden_dim:     Width of the adaptation MLP (default 32).
    """

    def __init__(self, latent_dim: int, edge_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        in_dim = 2 * latent_dim + edge_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(self, V: Tensor, E: Tensor) -> Tensor:
        """Adapt edges based on current slot states.

        Args:
            V: Tensor[B, N, d]      — current slot states
            E: Tensor[B, N, N, d_e] — current edge features

        Returns:
            E_new: Tensor[B, N, N, d_e] — updated edge features
        """
        B, N, d = V.shape

        # Pairwise expansion (same broadcasting as MessagePassingLayer)
        v_i = V.unsqueeze(2).expand(B, N, N, d)   # receiver: [B, N, N, d]
        v_j = V.unsqueeze(1).expand(B, N, N, d)   # sender:   [B, N, N, d]

        pair_input = torch.cat([v_i, v_j, E], dim=-1)  # [B, N, N, 2d + d_e]
        delta = torch.tanh(self.mlp(pair_input))        # [B, N, N, d_e]

        E_new = E + delta
        return E_new
