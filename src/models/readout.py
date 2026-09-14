import torch
import torch.nn as nn
from torch import Tensor


class InvariantReadout(nn.Module):
    """Permutation-invariant aggregation of slot states.
    
    G* = sum_i (a_i * v_i) / (sum_i a_i + eps)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, V: Tensor, A: Tensor | None = None) -> Tensor:
        """
        Args:
            V: Tensor[B, N, d]  slot representations
            A: Tensor[B, N]     slot activity gates (optional, default ones)
            
        Returns:
            G_star: Tensor[B, d] pooled representation
        """
        B, N, d = V.shape
        if A is None:
            A = torch.ones((B, N), device=V.device, dtype=V.dtype)

        weights = A.unsqueeze(-1)  # [B, N, 1]
        weighted_nodes = V * weights  # [B, N, d]
        sum_nodes = weighted_nodes.sum(dim=1)  # [B, d]
        sum_weights = A.sum(dim=1, keepdim=True).clamp(min=self.eps)  # [B, 1]

        return sum_nodes / sum_weights
