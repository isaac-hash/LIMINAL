from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from src.utils.config import ModelConfig, ActivityConfig, ResolutionConfig, ExternalConfig
from src.models.message_passing import MessagePassingLayer
from src.models.readout import InvariantReadout
from src.models.activity import ActivityGate
from src.models.resolution import HaltGate
from src.models.external_workspace import ExternalWorkspaceState
from src.models.externaliser import WriteController
from src.models.reader import ReadController
from src.models.relationships import EdgeAdapter


class LatentWorkspace(nn.Module):
    """Unified Latent Workspace supporting Vector, Graph, and Adaptive Resolution modes.

    - Vector mode: single recurrent vector updated with GRU recurrence across T steps.
    - Graph mode: N slot representations updated via relational message passing.
      - Phase 1 (static):     fixed T steps, all activity gates = 1.
      - Phase 2 (activity):   fixed T steps, ActivityGate computes per-slot gates.
      - Phase 3 (resolution): dynamic T steps via ACT-style HaltGate; each example
        accumulates halt probabilities and exits when cumulative P >= halt_threshold.
      - Phase 4 (persistence): latent state carried across sequence turns via PersistenceGate.
      - Phase 5 (external):   coupled internal-external loop with structured scratchpad.
        Each reasoning step: Read(V,W) → EdgeAdapt(V,E) → Activity → MsgPass → Write(W,V,A).
    """

    def __init__(
        self,
        config: ModelConfig,
        activity_config: ActivityConfig | None = None,
        resolution_config: ResolutionConfig | None = None,
        external_config: ExternalConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.activity_config = activity_config
        self.resolution_config = resolution_config
        self.external_config = external_config

        if config.type == "vector":
            self.gru_cell = nn.GRUCell(
                input_size=config.latent_dim,
                hidden_size=config.latent_dim,
            )
            self.write_controller: WriteController | None = None
            self.read_controller: ReadController | None = None
            self.edge_adapter: EdgeAdapter | None = None
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

            # Phase 5: optional external workspace controllers
            if external_config is not None and external_config.enabled:
                self.write_controller = WriteController(config.latent_dim, external_config)
                self.read_controller = ReadController(config.latent_dim, external_config)
                if external_config.edge_adaptation:
                    self.edge_adapter = EdgeAdapter(
                        latent_dim=config.latent_dim,
                        edge_dim=config.edge_dim,
                        hidden_dim=external_config.edge_hidden_dim,
                    )
                else:
                    self.edge_adapter = None
            else:
                self.write_controller = None
                self.read_controller = None
                self.edge_adapter = None
        else:
            raise ValueError(f"Unknown workspace model type: {config.type}")

    def forward(
        self,
        V_0: Tensor,
        E_0: Tensor | None = None,
        A_0: Tensor | None = None,
        workspace: ExternalWorkspaceState | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """
        Args:
            V_0:       Tensor[B, num_slots, latent_dim] initial slot/vector state
            E_0:       Tensor[B, N, N, d_e] optional initial edge tensor
            A_0:       Tensor[B, N] optional initial activity tensor (ignored when
                       activity_gate is active; activity is recomputed each step)
            workspace: Optional ExternalWorkspaceState to carry across sequence turns
                       (Phase 5). If None and external is enabled, a fresh empty
                       workspace is initialised at the start of the forward pass.

        Returns:
            h_final: Tensor[B, latent_dim] pooled representation for decoder
            info: dict containing intermediate trajectories and states
        """
        if self.config.type == "vector":
            return self._forward_vector(V_0)
        elif self.halt_gate is not None:
            return self._forward_graph_adaptive(V_0, E_0, A_0, workspace=workspace)
        else:
            return self._forward_graph(V_0, E_0, A_0, workspace=workspace)


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
        workspace: ExternalWorkspaceState | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Graph workspace relational message passing (fixed steps, Phases 1–5).

        When activity_gate is set, recomputes A at every step from the current
        slot states V.  Otherwise uses the static ones tensor (Phase 1 baseline).

        When external workspace is enabled (Phase 5), each step runs:
          Read(V,W) -> EdgeAdapt(V,E) -> Activity -> MsgPass -> Write(W,V,A)
        """
        B, N, d = V.shape
        ext = self.external_config
        use_external = ext is not None and ext.enabled and self.read_controller is not None

        if E is None:
            E = self.edge_prior.unsqueeze(0).expand(B, N, N, self.config.edge_dim)
        if A is None and self.activity_gate is None:
            A = torch.ones((B, N), device=V.device, dtype=V.dtype)

        # Initialise external workspace for this forward pass
        if use_external:
            assert ext is not None
            if workspace is None:
                workspace = ExternalWorkspaceState.init_empty(
                    B, ext.num_slots, ext.record_dim, V.device, V.dtype
                )
            workspace.reset_pass_flags()

        trajectory = [V.detach()]
        activity_trajectory: list[Tensor] = []
        workspace_trajectory: list[ExternalWorkspaceState] = []
        read_weights_trajectory: list[Tensor] = []

        for _t in range(self.config.reasoning_steps):
            # 1. Read from external workspace (before message passing)
            if use_external and workspace is not None and self.read_controller is not None:
                V, rw = self.read_controller(V, A if A is not None else torch.ones((B, N), device=V.device, dtype=V.dtype), workspace)
                read_weights_trajectory.append(rw.detach())

            # 2. Dynamic edge adaptation
            if self.edge_adapter is not None:
                E = self.edge_adapter(V, E)

            # 3. Compute activity gates: dynamic (Phase 2) or static ones (Phase 1)
            if self.activity_gate is not None:
                A = self.activity_gate(V)          # [B, N] — differentiable
            elif A is None:
                A = torch.ones((B, N), device=V.device, dtype=V.dtype)

            assert A is not None
            activity_trajectory.append(A.detach())

            # 4. Message passing update
            for layer in self.msg_layers:
                V = layer(V, E, A)
            trajectory.append(V.detach())

            # 5. Write to external workspace
            if use_external and workspace is not None and self.write_controller is not None:
                workspace = self.write_controller(workspace, V, A, step=_t)
                workspace_trajectory.append(workspace.clone(detach=True))

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
            "workspace_trajectory": workspace_trajectory,
            "read_weights_trajectory": read_weights_trajectory,
        }
        return h_final, info


    def _forward_graph_adaptive(
        self,
        V: Tensor,
        E: Tensor | None = None,
        A: Tensor | None = None,
        workspace: ExternalWorkspaceState | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """ACT-style dynamic graph workspace (Phases 3–5).

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

        New info keys (Phase 3):
            halt_probs:      list[Tensor[B]] — raw halt probability at each step
            ponder_weights:  Tensor[B, T_actual] — ACT weight per step
            n_steps:         Tensor[B] — effective (fractional) step count per example
            effective_steps: float — mean n_steps across batch

        New info keys (Phase 5, when external enabled):
            workspace_trajectory:     list[ExternalWorkspaceState] — per-step snapshots
            read_weights_trajectory:  list[Tensor[B, heads, M]] — per-step read attention
        """
        assert self.halt_gate is not None
        rc = self.resolution_config
        assert rc is not None

        B, N, d = V.shape
        T_max = rc.max_reasoning_steps

        ext = self.external_config
        use_external = ext is not None and ext.enabled and self.read_controller is not None

        if E is None:
            E = self.edge_prior.unsqueeze(0).expand(B, N, N, self.config.edge_dim)

        device = V.device
        dtype = V.dtype

        # Initialise external workspace for this forward pass
        if use_external:
            assert ext is not None
            if workspace is None:
                workspace = ExternalWorkspaceState.init_empty(
                    B, ext.num_slots, ext.record_dim, device, dtype
                )
            workspace.reset_pass_flags()

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
        workspace_trajectory: list[ExternalWorkspaceState] = []
        read_weights_trajectory: list[Tensor] = []
        A_current: Tensor = torch.ones((B, N), device=device, dtype=dtype)

        for _t in range(T_max):
            # 1. Read from external workspace (before activity + message passing)
            if use_external and workspace is not None and self.read_controller is not None:
                V, rw = self.read_controller(V, A_current, workspace)
                read_weights_trajectory.append(rw.detach())

            # 2. Dynamic edge adaptation
            if self.edge_adapter is not None:
                E = self.edge_adapter(V, E)

            # 3. Activity gates (Phase 2 or static)
            if self.activity_gate is not None:
                A_current = self.activity_gate(V)   # [B, N]
            else:
                A_current = torch.ones((B, N), device=device, dtype=dtype)

            activity_trajectory.append(A_current.detach())

            # 4. Message passing update
            for layer in self.msg_layers:
                V = layer(V, E, A_current)
            trajectory.append(V.detach())

            # 5. Write to external workspace
            if use_external and workspace is not None and self.write_controller is not None:
                workspace = self.write_controller(workspace, V, A_current, step=_t)
                workspace_trajectory.append(workspace.clone(detach=True))

            # 6. Halt probability
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
            # Phase 5 specific
            "workspace_trajectory": workspace_trajectory,
            "read_weights_trajectory": read_weights_trajectory,
        }
        return h_acc, info

