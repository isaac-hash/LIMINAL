import torch.nn as nn
from torch import Tensor


class Decoder(nn.Module):
    """Maps final pooled latent representation to task output logits."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h: Tensor) -> Tensor:
        """
        Args:
            h: Tensor[B, latent_dim]
            
        Returns:
            logits: Tensor[B, num_classes]
        """
        return self.mlp(h)
