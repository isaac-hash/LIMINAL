import torch
import torch.nn as nn
from torch import Tensor


class MessagePassingLayer(nn.Module):
    """Equivariant relational message-passing layer for small slot graphs (N <= 32).
    
    Implemented via vectorized tensor broadcasting without external graph library dependencies.
    """

    def __init__(self, latent_dim: int, edge_dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # φ_θ(v_i, v_j, e_ij) -> m_ij
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # GRU update per slot
        self.update = nn.GRUCell(input_size=latent_dim, hidden_size=latent_dim)

    def forward(self, V: Tensor, E: Tensor | None = None, A: Tensor | None = None) -> Tensor:
        """
        Args:
            V: Tensor[B, N, d]      Node/slot states
            E: Tensor[B, N, N, d_e] Edge features (optional, default zeros)
            A: Tensor[B, N]         Activity gates (optional, default ones)
            
        Returns:
            V_new: Tensor[B, N, d]  Updated node states
        """
        B, N, d = V.shape

        if E is None:
            E = torch.zeros((B, N, N, self.edge_dim), device=V.device, dtype=V.dtype)
        if A is None:
            A = torch.ones((B, N), device=V.device, dtype=V.dtype)

        # Pairwise expansion:
        # v_i (receiver): [B, N, 1, d] -> expand to [B, N, N, d]
        # v_j (sender):   [B, 1, N, d] -> expand to [B, N, N, d]
        v_i = V.unsqueeze(2).expand(B, N, N, d)
        v_j = V.unsqueeze(1).expand(B, N, N, d)

        pair_input = torch.cat([v_i, v_j, E], dim=-1)  # [B, N, N, 2d + d_e]
        messages = self.msg_mlp(pair_input)            # [B, N, N, d]

        # Mask messages by sender activity A_j (dim 2)
        a_j = A.unsqueeze(1).unsqueeze(-1)             # [B, 1, N, 1]
        messages = messages * a_j

        # Aggregate incoming messages: sum_j m_ij -> [B, N, d]
        m_aggregated = messages.sum(dim=2)

        # GRU update: reshape to [B*N, d]
        V_flat = V.reshape(B * N, d)
        m_flat = m_aggregated.reshape(B * N, d)
        V_new_flat = self.update(m_flat, V_flat)
        V_new = V_new_flat.reshape(B, N, d)

        return V_new
