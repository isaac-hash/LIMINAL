from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from src.utils.config import ModelConfig, ActivityConfig, ResolutionConfig
from src.models.message_passing import MessagePassingLayer
from src.models.readout import InvariantReadout
from src.models.activity import ActivityGate
from src.models.resolution import HaltGate


class LatentWorkspace(nn.Module):
    """Unified Latent Workspace supporting Vector, Graph, and Adaptive Resolution modes.

    - Vector mode: single recurrent vector updated with GRU recurrence across T steps.
    - Graph mode: N slot representations updated via relational message passing.
      - Phase 1 (static):     fixed T steps, all activity gates = 1.
      - Phase 2 (activity):   fixed T steps, ActivityGate computes per-slot gates.
      - Phase 3 (resolution): dynamic T steps via ACT-style HaltGate; each example
        accumulates halt probabilities and exits when cumulative P >= halt_threshold.
    """

    def __init__(
        self,
        config: ModelConfig,
        activity_config: ActivityConfig | None = None,
        resolution_config: ResolutionConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.activity_config = activity_config
        self.resolution_config = resolution_config

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

            # Phase 3: optional halt gate (only instantiated when enabled)
            if resolution_config is not None and resolution_config.enabled:
                self.halt_gate = HaltGate(
                    latent_dim=config.latent_dim,
                    halt_hidden_dim=resolution_config.halt_hidden_dim,
                )
            else:
                self.halt_gate = None
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
        elif self.halt_gate is not None:
            return self._forward_graph_adaptive(V_0, E_0, A_0)
        else:
            return self._forward_graph(V_0, E_0, A_0)

    def _forward_vector(self, V_0: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """Vector baseline recurrence."""
        h = V_0.squeeze(1)  # [B, latent_dim]
        trajectory = [h.detach()]

        for _ in range(self.config.reasoning_steps):
            h = self.gru_cell(h, h)
            trajectory.append(h.detach())

        return h, {"trajectory": trajectory, "h_final": h, "V_final": h.unsqueeze(1)}

    def _forward_graph(
        self,
        V: Tensor,
        E: Tensor | None = None,
        A: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Graph workspace relational message passing (fixed steps, Phases 1 & 2).

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

            assert A is not None
            activity_trajectory.append(A.detach())

            for layer in self.msg_layers:
                V = layer(V, E, A)
            trajectory.append(V.detach())

        # Final activity for readout (recompute if dynamic)
        if self.activity_gate is not None:
            A = self.activity_gate(V)
        assert A is not None

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

    def _forward_graph_adaptive(
        self,
        V: Tensor,
        E: Tensor | None = None,
        A: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """ACT-style dynamic graph workspace (Phase 3).

        Each example in the batch independently accumulates a halt probability.
        When an example's cumulative halt probability H_t >= halt_threshold, it
        has "halted": further steps do not contribute to its output.

        The final h_final for each example is:
            h_final[b] = sum_t  w_t[b] * readout(V_t[b], A_t[b])

        where ponder weights w_t follow the ACT remainder distribution:
            w_t[b] = h_t[b]   for t < T_halt[b]
            w_t[b] = R_t[b]   for t = T_halt[b]   (remainder = 1 - cumulative)
            w_t[b] = 0        for t > T_halt[b]

        This guarantees sum_t w_t[b] = 1 for every example b.

        New info keys:
            halt_probs:      list[Tensor[B]] — raw halt probability at each step
            ponder_weights:  Tensor[B, T_actual] — ACT weight per step
            n_steps:         Tensor[B] — effective (fractional) step count per example
            effective_steps: float — mean n_steps across batch
        """
        assert self.halt_gate is not None
        rc = self.resolution_config
        assert rc is not None

        B, N, d = V.shape
        T_max = rc.max_reasoning_steps

        if E is None:
            E = self.edge_prior.unsqueeze(0).expand(B, N, N, self.config.edge_dim)

        device = V.device
        dtype = V.dtype

        # ACT accumulation state
        halted = torch.zeros(B, dtype=torch.bool, device=device)   # [B]
        cumulative_h = torch.zeros(B, device=device, dtype=dtype)  # [B]
        # Output accumulator: weighted sum of readout vectors
        h_acc = torch.zeros(B, d, device=device, dtype=dtype)      # [B, d]

        # Diagnostics (detached)
        halt_probs: list[Tensor] = []
        ponder_weights_list: list[Tensor] = []  # per-step weight [B]

        trajectory = [V.detach()]
        activity_trajectory: list[Tensor] = []
        A_current: Tensor = torch.ones((B, N), device=device, dtype=dtype)

        for _t in range(T_max):
            # --- Activity gates (Phase 2 or static) ---
            if self.activity_gate is not None:
                A_current = self.activity_gate(V)   # [B, N]
            else:
                A_current = torch.ones((B, N), device=device, dtype=dtype)

            activity_trajectory.append(A_current.detach())

            # --- Message passing update ---
            for layer in self.msg_layers:
                V = layer(V, E, A_current)
            trajectory.append(V.detach())

            # --- Halt probability ---
            h_t = self.halt_gate(V, A_current)  # [B], differentiable

            # Remainder: probability budget remaining before this step
            remainder = (1.0 - cumulative_h).clamp(min=0.0)

            # Ponder weight:
            #   final step (just halted)  -> remainder
            #   non-halted step           -> h_t
            #   already halted            -> 0
            not_halted = ~halted                                                # [B]
            would_halt = (cumulative_h + h_t >= rc.halt_threshold) & not_halted  # [B]

            weight = torch.where(
                would_halt,
                remainder,
                torch.where(not_halted, h_t, torch.zeros_like(h_t))
            )

            # Accumulate weighted readout
            h_step = self.readout(V, A_current)   # [B, d]
            h_acc = h_acc + weight.unsqueeze(-1) * h_step

            # Update cumulative halt probability for not-yet-halted examples
            cumulative_h = cumulative_h + torch.where(
                not_halted, h_t, torch.zeros_like(h_t)
            )

            # Mark newly halted
            halted = halted | would_halt

            halt_probs.append(h_t.detach())
            ponder_weights_list.append(weight.detach())

            if halted.all():
                break

        # If any example never halted (ponder budget not exhausted), assign
        # remaining weight to its last step readout to keep weights summing to 1.
        not_fully_halted = ~halted
        if not_fully_halted.any():
            remaining = (1.0 - cumulative_h).clamp(min=0.0)
            h_step = self.readout(V, A_current)
            h_acc = h_acc + (remaining * not_fully_halted.float()).unsqueeze(-1) * h_step

        # Final activity for info dict
        if self.activity_gate is not None:
            A_final = self.activity_gate(V)
        else:
            A_final = torch.ones((B, N), device=device, dtype=dtype)

        # Effective step count per example via ACT ponder weight distribution
        ponder_weights = torch.stack(ponder_weights_list, dim=1)   # [B, T_actual]
        T_actual = ponder_weights.shape[1]
        step_indices = torch.arange(1, T_actual + 1, device=device, dtype=dtype).unsqueeze(0)
        n_steps = (ponder_weights * step_indices).sum(dim=1)       # [B]

        info: dict[str, Any] = {
            "trajectory": trajectory,
            "activity_trajectory": activity_trajectory,
            "V_final": V,
            "A_final": A_final,
            "E_final": E,
            "h_final": h_acc,
            # Phase 3 specific
            "halt_probs": halt_probs,
            "ponder_weights": ponder_weights,
            "n_steps": n_steps,
            "effective_steps": n_steps.mean().item(),
        }
        return h_acc, info
