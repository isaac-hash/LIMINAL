from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from src.utils.config import ModelConfig, ActivityConfig
from src.models.message_passing import MessagePassingLayer
from src.models.readout import InvariantReadout
from src.models.activity import ActivityGate


class LatentWorkspace(nn.Module):
    """Unified Latent Workspace supporting both Vector and Graph modes.

    - Vector mode: single recurrent vector updated with GRU recurrence across T steps.
    - Graph mode: N slot representations updated via relational message passing across T steps.
      When activity.enabled=True, an ActivityGate dynamically computes per-slot gate scores
      at every step; otherwise all gates are 1 (static, Phase 1 behaviour).
    """

    def __init__(self, config: ModelConfig, activity_config: ActivityConfig | None = None):
        super().__init__()
        self.config = config
        self.activity_config = activity_config

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
            # Phase 2A: optional activity gate (only instantiated when enabled)
            if activity_config is not None and activity_config.enabled:
                self.activity_gate = ActivityGate(
                    latent_dim=config.latent_dim,
                    gate_hidden_dim=activity_config.gate_hidden_dim,
                )
            else:
                self.activity_gate = None
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
            A_0: Tensor[B, N] optional initial activity tensor (ignored when
                 activity_gate is active; activity is recomputed each step)

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
        """Graph workspace relational message passing.

        When activity_gate is set, recomputes A at every step from the current
        slot states V.  Otherwise uses the static ones tensor (Phase 1 baseline).
        """
        B, N, d = V.shape

        if E is None:
            # Broadcast learnable edge prior
            E = self.edge_prior.unsqueeze(0).expand(B, N, N, self.config.edge_dim)
        if A is None and self.activity_gate is None:
            A = torch.ones((B, N), device=V.device, dtype=V.dtype)

        trajectory = [V.detach()]
        activity_trajectory: list[Tensor] = []

        for _ in range(self.config.reasoning_steps):
            # Compute activity gates: dynamic (Phase 2) or static ones (Phase 1)
            if self.activity_gate is not None:
                A = self.activity_gate(V)          # [B, N] — differentiable
            elif A is None:
                A = torch.ones((B, N), device=V.device, dtype=V.dtype)

            activity_trajectory.append(A.detach())

            for layer in self.msg_layers:
                V = layer(V, E, A)
            trajectory.append(V.detach())

        # Final activity for readout (recompute if dynamic)
        if self.activity_gate is not None:
            A = self.activity_gate(V)

        h_final = self.readout(V, A)

        info: dict[str, Any] = {
            "trajectory": trajectory,
            "activity_trajectory": activity_trajectory,
            "V_final": V,
            "A_final": A,
            "E_final": E,
            "h_final": h_final,
        }
        return h_final, info
