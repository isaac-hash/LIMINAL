"""Phase 5/7 — Write controllers for the external workspace.

Phase 5  WriteController        Deterministic top-K write policy.
Phase 7  LearnedWriteController Gumbel-softmax learned write gate.

Gradient contract (Phase 5 WriteController)
-------------------------------------------
Gradients flow through the **continuous payload tensors** (records[b, m*]).
They do NOT flow through:
  - slot selection (argtopk on activity scores — discrete)
  - type assignment (argmax on type_logits — discrete)
  - eviction target selection (argmin on step_written — discrete)

Gradient contract (Phase 7 LearnedWriteController)
---------------------------------------------------
Gradients flow through:
  - write gate (Gumbel-softmax — soft and continuous during training)
  - payload projection (always differentiable)
Type assignment remains discrete (argmax), which is acceptable because the
type head is unsupervised and its gradient path is low priority.  Eviction
remains discrete (LRU argmin).

Tau annealing
-------------
The caller (SequentialTrainer) is responsible for periodically calling
    model.workspace.write_controller.set_tau(new_tau)
Each epoch, tau is decayed linearly from gumbel_tau_start → gumbel_tau_end
over gumbel_anneal_epochs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.external_workspace import ExternalWorkspaceState
from src.utils.config import ExternalConfig


# ---------------------------------------------------------------------------
# Phase 5 — Deterministic top-K write controller
# ---------------------------------------------------------------------------

class WriteController(nn.Module):
    """Deterministic top-K write controller.

    For each batch item, selects the top-K latent slots by activity score,
    projects their continuous representations into external record space,
    assigns a discrete type via argmax, and writes to empty or LRU slots.

    Args:
        latent_dim:  Dimensionality d of latent slot vectors.
        config:      ExternalConfig holding num_slots, record_dim, num_types,
                     write_top_k, write_dedup flags.
    """

    def __init__(self, latent_dim: int, config: ExternalConfig):
        super().__init__()
        self.config = config

        # Projection: latent slot -> record payload
        self.payload_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, config.record_dim),
        )

        # Type classifier: latent slot -> type logits (unsupervised in Phase 5)
        # Near-uniform init: small random weights + zero bias prevents early type collapse.
        type_head = nn.Linear(latent_dim, config.num_types)
        nn.init.normal_(type_head.weight, std=0.01)
        nn.init.zeros_(type_head.bias)
        self.type_head = type_head

    def forward(
        self,
        workspace: ExternalWorkspaceState,
        V: Tensor,
        A: Tensor,
        step: int,
    ) -> tuple[ExternalWorkspaceState, dict]:
        """Write top-K active slots from V into the external workspace.

        Args:
            workspace: Current ExternalWorkspaceState [B, M, d_ext].
            V:         Latent slot states Tensor[B, N, d].
            A:         Activity gate values Tensor[B, N] in (0, 1).
            step:      Current reasoning step index (used as LRU key).

        Returns:
            Tuple of (updated ExternalWorkspaceState, info dict).
            info['write_gate'] is a hard [B, N] top-K mask for selectivity logging.
        """
        B, N, d = V.shape
        K = min(self.config.write_top_k, N)

        # Detach activity scores — selection is not differentiable
        scores = A.detach()  # [B, N]

        # Top-K slot indices per batch item (discrete, no gradient)
        _, topk_indices = torch.topk(scores, K, dim=1)  # [B, K]

        # Build hard mask for logging / selectivity analysis
        write_mask = torch.zeros(B, N, device=V.device)
        write_mask.scatter_(1, topk_indices, 1.0)

        # Project all slots to payloads in one vectorised call (differentiable)
        V_flat = V.reshape(B * N, d)
        payloads_flat = self.payload_proj(V_flat)          # [B*N, d_ext]
        payloads = payloads_flat.reshape(B, N, self.config.record_dim)

        # Type logits (discrete argmax — no gradient through type selection)
        type_logits_flat = self.type_head(V_flat.detach())  # [B*N, num_types]
        type_ids_flat = type_logits_flat.argmax(dim=-1)     # [B*N]
        type_ids = type_ids_flat.reshape(B, N)              # [B, N]

        # Clone tensors we need to mutate (keep backward graph on records only)
        new_records = workspace.records.clone()              # [B, M, d_ext]
        new_types = workspace.types.clone()                  # [B, M]
        new_mask = workspace.mask.clone()                    # [B, M]
        new_step_written = workspace.step_written.clone()    # [B, M]
        # Ensure wrote_this_pass can track all N latent slots
        if workspace.wrote_this_pass.shape[1] < N:
            expanded = torch.zeros(B, N, device=V.device, dtype=torch.bool)
            expanded[:, :workspace.wrote_this_pass.shape[1]] = workspace.wrote_this_pass
            workspace.wrote_this_pass = expanded

        new_wrote_this_pass = workspace.wrote_this_pass.clone()

        for b in range(B):
            for k_idx in range(K):
                slot_i = int(topk_indices[b, k_idx].item())

                # Deduplication: skip if this latent slot already wrote this forward pass
                if self.config.write_dedup and new_wrote_this_pass[b, slot_i].item():
                    continue

                # Find target external slot: prefer empty, then LRU
                empty_slots = (~new_mask[b]).nonzero(as_tuple=True)[0]
                if empty_slots.numel() > 0:
                    target_m = int(empty_slots[0].item())
                else:
                    # All full — evict least recently written (LRU)
                    target_m = int(new_step_written[b].argmin().item())

                # Write payload (differentiable)
                new_records[b, target_m] = payloads[b, slot_i]

                # Write metadata (non-differentiable)
                new_types[b, target_m] = type_ids[b, slot_i]
                new_mask[b, target_m] = True
                new_step_written[b, target_m] = step
                new_wrote_this_pass[b, slot_i] = True

        workspace.records = new_records
        workspace.types = new_types
        workspace.mask = new_mask
        workspace.step_written = new_step_written
        workspace.wrote_this_pass = new_wrote_this_pass
        info: dict = {"write_gate": write_mask.detach()}  # [B, N] hard mask
        return workspace, info


# ---------------------------------------------------------------------------
# Phase 7 — Learned Gumbel-softmax write controller
# ---------------------------------------------------------------------------

class LearnedWriteController(nn.Module):
    """Differentiable write controller using a Gumbel-softmax write gate.

    For each latent slot i, a small MLP scores whether that slot should write
    to the external workspace at this step.  During training the scores are
    passed through a Gumbel-softmax (temperature tau) so gradients flow.
    During inference (eval mode or tau near 0) the gate collapses to hard {0,1}.

    Architecture
    ------------
    gate_net : [v_i || a_i]  (d+1) -> hidden -> 1 logit  (per slot, shared weights)
    write_gate: Gumbel-softmax over [logit, -logit], index-0 = p_write

    Tau annealing
    -------------
    Call set_tau(tau) each epoch.  Annealed linearly from
    gumbel_tau_start -> gumbel_tau_end over gumbel_anneal_epochs.

    Sparsity
    --------
    info['write_gate_mean'] (scalar, differentiable) is exposed so
    LossComputer can apply write_sparsity_lambda * write_gate_mean.

    Args:
        latent_dim:  Dimensionality d of latent slot vectors.
        config:      ExternalConfig (uses record_dim, num_types, num_slots,
                     write_dedup, learned_gate, gumbel_tau_start).
    """

    def __init__(self, latent_dim: int, config: ExternalConfig):
        super().__init__()
        self.config = config
        self.tau: float = config.gumbel_tau_start

        # Payload projection (differentiable)
        self.payload_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, config.record_dim),
        )

        # Write gate MLP: [v_i || a_i] -> logit (shared weights across slots)
        gate_hidden = max(16, latent_dim // 2)
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim + 1, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        # Negative bias -> model starts conservative (writes few slots)
        nn.init.constant_(self.gate_net[-1].bias, -1.0)

        # Type classifier (discrete, detached — same as Phase 5)
        type_head = nn.Linear(latent_dim, config.num_types)
        nn.init.normal_(type_head.weight, std=0.01)
        nn.init.zeros_(type_head.bias)
        self.type_head = type_head

    def set_tau(self, tau: float) -> None:
        """Update the Gumbel temperature. Called by the trainer each epoch."""
        self.tau = max(float(tau), 1e-4)

    def forward(
        self,
        workspace: ExternalWorkspaceState,
        V: Tensor,
        A: Tensor,
        step: int,
    ) -> tuple[ExternalWorkspaceState, dict]:
        """Write to the external workspace using a learned Gumbel-softmax gate.

        Args:
            workspace: Current ExternalWorkspaceState [B, M, d_ext].
            V:         Latent slot states Tensor[B, N, d].
            A:         Activity gate values Tensor[B, N] in (0, 1).
            step:      Current reasoning step index (LRU key).

        Returns:
            Tuple of (updated ExternalWorkspaceState, info dict).
            info keys:
                'write_gate'      Tensor[B, N] soft gate (differentiable during train).
                'write_gate_mean' scalar Tensor, mean gate (for sparsity loss).
                'write_gate_hard' Tensor[B, N] binarised gate (for logging).
        """
        B, N, d = V.shape

        # ── Gate computation ────────────────────────────────────────────────
        a_expanded = A.unsqueeze(-1)                         # [B, N, 1]
        gate_input = torch.cat([V, a_expanded], dim=-1)      # [B, N, d+1]
        gate_logit = self.gate_net(gate_input).squeeze(-1)   # [B, N]

        if self.training:
            # Gumbel-softmax: 2-class [write, no-write] per slot
            logits_2 = torch.stack([gate_logit, -gate_logit], dim=-1)  # [B, N, 2]
            gate_soft = F.gumbel_softmax(logits_2, tau=self.tau, hard=False)[..., 0]  # [B, N]
            gate_hard = (gate_soft.detach() > 0.5).float()
        else:
            # Hard gate at inference (no Gumbel noise)
            gate_soft = (gate_logit > 0).float()
            gate_hard = (gate_logit.detach() > 0).float()

        # ── Payload projection (gradient flows through gate_soft) ───────────
        V_flat = V.reshape(B * N, d)
        payloads_flat = self.payload_proj(V_flat)             # [B*N, d_ext]
        payloads = payloads_flat.reshape(B, N, self.config.record_dim)
        # Scale payload by soft gate so gradient flows through the gate
        payloads_gated = payloads * gate_soft.unsqueeze(-1)   # [B, N, d_ext]

        # ── Type logits (discrete, detached) ────────────────────────────────
        type_logits_flat = self.type_head(V_flat.detach())
        type_ids_flat = type_logits_flat.argmax(dim=-1)
        type_ids = type_ids_flat.reshape(B, N)

        # ── Write to workspace (hard gate controls which slots fire) ─────────
        new_records = workspace.records.clone()
        new_types = workspace.types.clone()
        new_mask = workspace.mask.clone()
        new_step_written = workspace.step_written.clone()

        if workspace.wrote_this_pass.shape[1] < N:
            expanded = torch.zeros(B, N, device=V.device, dtype=torch.bool)
            expanded[:, :workspace.wrote_this_pass.shape[1]] = workspace.wrote_this_pass
            workspace.wrote_this_pass = expanded
        new_wrote_this_pass = workspace.wrote_this_pass.clone()

        for b in range(B):
            write_slots_b = gate_hard[b].nonzero(as_tuple=True)[0].tolist()
            for slot_i in write_slots_b:
                slot_i = int(slot_i)
                if self.config.write_dedup and new_wrote_this_pass[b, slot_i].item():
                    continue

                empty_slots = (~new_mask[b]).nonzero(as_tuple=True)[0]
                if empty_slots.numel() > 0:
                    target_m = int(empty_slots[0].item())
                else:
                    target_m = int(new_step_written[b].argmin().item())

                # Differentiable write (through payloads_gated -> gate_soft -> gate_net)
                new_records[b, target_m] = payloads_gated[b, slot_i]
                new_types[b, target_m] = type_ids[b, slot_i]
                new_mask[b, target_m] = True
                new_step_written[b, target_m] = step
                new_wrote_this_pass[b, slot_i] = True

        workspace.records = new_records
        workspace.types = new_types
        workspace.mask = new_mask
        workspace.step_written = new_step_written
        workspace.wrote_this_pass = new_wrote_this_pass

        info: dict = {
            "write_gate": gate_soft,                   # [B, N] differentiable
            "write_gate_mean": gate_soft.mean(),       # scalar, for sparsity loss
            "write_gate_hard": gate_hard,              # [B, N] for logging
        }
        return workspace, info
