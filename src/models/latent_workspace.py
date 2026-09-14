from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from src.utils.config import ModelConfig
from src.models.message_passing import MessagePassingLayer
from src.models.readout import InvariantReadout


class LatentWorkspace(nn.Module):
    """Unified Latent Workspace supporting both Vector and Graph modes.
    
    - Vector mode: single recurrent vector updated with GRU recurrence across T steps.
    - Graph mode: N slot representations updated via relational message passing across T steps.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        if config.type == "vector":
            self.gru_cell = nn.GRUCell(
                input_size=config.latent_dim,
                hidden_size=config.latent_dim,
            )
        elif config.type == "graph":
            self.msg_layers = nn.ModuleList([
                MessagePassingLayer(
                    latent_dim=config.latent_dim,
                    edge_dim=config.edge_dim,
                    hidden_dim=config.msg_hidden_dim,
                )
                for _ in range(config.num_msg_layers)
            ])
            self.readout = InvariantReadout()
            # Learnable edge parameter prior
            self.edge_prior = nn.Parameter(
                torch.zeros((config.latent_slots, config.latent_slots, config.edge_dim))
            )
        else:
            raise ValueError(f"Unknown workspace model type: {config.type}")

    def forward(
        self,
        V_0: Tensor,
        E_0: Tensor | None = None,
        A_0: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """
        Args:
            V_0: Tensor[B, num_slots, latent_dim] initial slot/vector state
            E_0: Tensor[B, N, N, d_e] optional initial edge tensor
            A_0: Tensor[B, N] optional initial activity tensor
            
        Returns:
            h_final: Tensor[B, latent_dim] pooled representation for decoder
            info: dict containing intermediate trajectories and states
        """
        if self.config.type == "vector":
            return self._forward_vector(V_0)
        else:
            return self._forward_graph(V_0, E_0, A_0)

    def _forward_vector(self, V_0: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Vector baseline recurrence."""
        h = V_0.squeeze(1)  # [B, latent_dim]
        trajectory = [h.detach()]

        for _ in range(self.config.reasoning_steps):
            h = self.gru_cell(h, h)
            trajectory.append(h.detach())

        return h, {"trajectory": trajectory, "h_final": h}

    def _forward_graph(
        self,
        V: Tensor,
        E: Tensor | None = None,
        A: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Graph workspace relational message passing."""
        B, N, d = V.shape

        if E is None:
            # Broadcast learnable edge prior
            E = self.edge_prior.unsqueeze(0).expand(B, N, N, self.config.edge_dim)
        if A is None:
            A = torch.ones((B, N), device=V.device, dtype=V.dtype)

        trajectory = [V.detach()]
        for _ in range(self.config.reasoning_steps):
            for layer in self.msg_layers:
                V = layer(V, E, A)
            trajectory.append(V.detach())

        h_final = self.readout(V, A)
        return h_final, {
            "trajectory": trajectory,
            "V_final": V,
            "A_final": A,
            "E_final": E,
            "h_final": h_final,
        }
