"""Phase 5 — WriteController: deterministic top-K write policy.

Selects the K most active latent slots, projects them into record space,
and allocates to the external workspace using a greedy LRU policy.

Gradient contract
-----------------
Gradients flow through the **continuous payload tensors** (records[b, m*]).
They do NOT flow through:
  - slot selection (argtopk on activity scores — discrete)
  - type assignment (argmax on type_logits — discrete)
  - eviction target selection (argmin on step_written — discrete)
This is intentional Phase 5 behaviour.  Phase 7 will replace slot selection
with Gumbel-Softmax to make the full write operation differentiable.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.external_workspace import ExternalWorkspaceState
from src.utils.config import ExternalConfig


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

        # Projection: latent slot → record payload
        self.payload_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, config.record_dim),
        )

        # Type classifier: latent slot → type logits (unsupervised in Phase 5)
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
    ) -> ExternalWorkspaceState:
        """Write top-K active slots from V into the external workspace.

        Args:
            workspace: Current ExternalWorkspaceState [B, M, d_ext].
            V:         Latent slot states Tensor[B, N, d].
            A:         Activity gate values Tensor[B, N] in (0, 1).
            step:      Current reasoning step index (used as LRU key).

        Returns:
            Updated ExternalWorkspaceState (modified in-place and returned).
        """
        B, N, d = V.shape
        K = min(self.config.write_top_k, N)

        # Detach activity scores — selection is not differentiable
        scores = A.detach()  # [B, N]

        # Top-K slot indices per batch item (discrete, no gradient)
        _, topk_indices = torch.topk(scores, K, dim=1)  # [B, K]

        # Project all slots to payloads in one vectorised call (differentiable)
        # V: [B, N, d] → payloads: [B, N, d_ext]
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
                empty_slots = (~new_mask[b]).nonzero(as_tuple=True)[0]  # indices of empty slots
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
        return workspace
