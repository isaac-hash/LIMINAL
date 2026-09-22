"""Phase 5 — ReadController: multi-head cross-attention over external records.

Query:  global pooled latent representation  [B, d_latent]
Keys:   external records enriched with type embeddings  [B, M, d_attn]
Values: projected external records  [B, M, d_val]

Empty slots (mask=False) are excluded from attention via -inf masking.

Output is projected back to latent_dim and optionally gated per-slot
before being added to the latent slot tensor (see latent_workspace.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.external_workspace import ExternalWorkspaceState
from src.utils.config import ExternalConfig


class ReadController(nn.Module):
    """Multi-head cross-attention read over external workspace records.

    Args:
        latent_dim:  Dimensionality d of internal latent slot vectors.
        config:      ExternalConfig with record_dim, num_types, read_heads flags.
    """

    def __init__(self, latent_dim: int, config: ExternalConfig):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.num_heads = config.read_heads
        self.record_dim = config.record_dim

        # Each attention head operates on record_dim // num_heads dimensions
        assert config.record_dim % config.read_heads == 0, (
            f"record_dim ({config.record_dim}) must be divisible by read_heads ({config.read_heads})"
        )
        self.head_dim = config.record_dim // config.read_heads

        # Type embedding enriches keys with semantic type signal
        self.type_embedding = nn.Embedding(config.num_types, config.record_dim)

        # Projections
        self.key_proj   = nn.Linear(config.record_dim, config.record_dim)
        self.value_proj = nn.Linear(config.record_dim, config.record_dim)
        self.query_proj = nn.Linear(latent_dim, config.record_dim)

        # Output projection back to latent dim
        self.out_proj = nn.Linear(config.record_dim, latent_dim)

        # Per-slot gated injection gate (only used when config.read_gate=True)
        if config.read_gate:
            self.gate_linear = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        V: Tensor,
        A: Tensor,
        workspace: ExternalWorkspaceState,
    ) -> tuple[Tensor, Tensor]:
        """Read from external workspace conditioned on current latent state.

        Args:
            V:         Latent slot states Tensor[B, N, d_latent].
            A:         Activity gate values Tensor[B, N].
            workspace: Current ExternalWorkspaceState.

        Returns:
            V_out:        Updated slot tensor Tensor[B, N, d_latent].
            read_weights: Attention weights Tensor[B, num_heads, M] for logging.
        """
        B, N, d = V.shape
        M = workspace.records.shape[1]

        # --- Build Query from global pooled latent (activity-weighted mean) ---
        A_norm = A / (A.sum(dim=1, keepdim=True).clamp(min=1e-6))  # [B, N]
        pooled = (A_norm.unsqueeze(-1) * V).sum(dim=1)             # [B, d]
        Q = self.query_proj(pooled)                                 # [B, record_dim]

        # --- Build Keys (records enriched by type embeddings) ---
        type_emb = self.type_embedding(workspace.types)             # [B, M, record_dim]
        enriched = workspace.records + type_emb                     # [B, M, record_dim]
        K = self.key_proj(enriched)                                 # [B, M, record_dim]
        V_ext = self.value_proj(workspace.records)                  # [B, M, record_dim]

        # --- Multi-head reshape ---
        # Q: [B, record_dim] → [B, num_heads, 1, head_dim]
        Q_h = Q.view(B, self.num_heads, 1, self.head_dim)
        # K, V_ext: [B, M, record_dim] → [B, num_heads, M, head_dim]
        K_h = K.view(B, self.num_heads, M, self.head_dim)
        V_h = V_ext.view(B, self.num_heads, M, self.head_dim)

        # --- Scaled dot-product attention with empty-slot masking ---
        scale = self.head_dim ** -0.5
        scores = torch.einsum("bhqd,bhmd->bhqm", Q_h, K_h) * scale  # [B, num_heads, 1, M]

        # Mask empty slots: -inf so they receive ~0 attention after softmax
        attn_mask = ~workspace.mask                                   # [B, M]  True = empty
        attn_mask_h = attn_mask.unsqueeze(1).unsqueeze(2)             # [B, 1, 1, M]
        scores = scores.masked_fill(attn_mask_h, float("-inf"))

        # Handle all-empty case: if every slot is masked, softmax → nan; clamp to zeros
        all_empty = attn_mask.all(dim=1)  # [B]
        if all_empty.any():
            scores[all_empty] = 0.0       # uniform → equal weights → safe softmax

        weights = F.softmax(scores, dim=-1)   # [B, num_heads, 1, M]
        read_weights = weights.squeeze(2)     # [B, num_heads, M] — exported for logging

        # Weighted sum of values: weights [B,heads,1,M] x V_h [B,heads,M,head_dim]
        # → [B, heads, head_dim]
        attn_out = torch.einsum("bhqm,bhmd->bhd", weights, V_h)  # [B, num_heads, head_dim]
        # Merge heads → [B, record_dim]
        attn_out = attn_out.reshape(B, self.config.record_dim)


        # Project to latent dim
        R = self.out_proj(attn_out)   # [B, d_latent]

        # --- Per-slot gated injection into V ---
        if self.config.read_gate:
            gate = torch.sigmoid(self.gate_linear(R))   # [B, d_latent]
            gate_exp = gate.unsqueeze(1)                 # [B, 1, d_latent] → broadcast over N
            R_exp = R.unsqueeze(1)                       # [B, 1, d_latent]
            V_out = V + gate_exp * R_exp                 # [B, N, d_latent]
        else:
            V_out = V + R.unsqueeze(1)

        return V_out, read_weights
