import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """Encodes input facts into an initial latent workspace representation.
    
    Supports:
    - Vector baseline: masked mean pooling into a single vector [B, 1, latent_dim]
    - Graph workspace: scatter/accumulation into N slot vectors [B, num_slots, latent_dim]
    """

    def __init__(
        self,
        type_vocab_size: int = 16,
        entity_vocab_size: int = 64,
        key_vocab_size: int = 32,
        op_vocab_size: int = 16,
        embed_dim: int = 16,
        latent_dim: int = 32,
        num_slots: int = 1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        self.type_embed = nn.Embedding(type_vocab_size, embed_dim)
        self.entity_embed = nn.Embedding(entity_vocab_size, embed_dim)
        self.key_embed = nn.Embedding(key_vocab_size, embed_dim)
        self.op_embed = nn.Embedding(op_vocab_size, embed_dim)
        self.val_proj = nn.Linear(1, embed_dim)

        fact_feature_dim = 5 * embed_dim
        self.fact_mlp = nn.Sequential(
            nn.Linear(fact_feature_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, facts: Tensor, fact_mask: Tensor) -> Tensor:
        """
        Args:
            facts: Tensor[B, max_facts, 5] of integer tokens [type, entity, key, val, op]
            fact_mask: Tensor[B, max_facts] of 1.0 (valid) / 0.0 (pad)
            
        Returns:
            V_0: Tensor[B, num_slots, latent_dim]
        """
        B, max_facts, _ = facts.shape

        type_tokens = facts[..., 0]
        entity_tokens = facts[..., 1]
        key_tokens = facts[..., 2]
        val_tokens = facts[..., 3].unsqueeze(-1).float()
        op_tokens = facts[..., 4]

        e_type = self.type_embed(type_tokens)
        e_entity = self.entity_embed(entity_tokens)
        e_key = self.key_embed(key_tokens)
        e_val = self.val_proj(val_tokens)
        e_op = self.op_embed(op_tokens)

        # Concatenate 5 features -> [B, max_facts, 5 * embed_dim]
        fact_repr = torch.cat([e_type, e_entity, e_key, e_val, e_op], dim=-1)
        fact_vectors = self.fact_mlp(fact_repr)  # [B, max_facts, latent_dim]

        # Apply fact mask
        mask_expanded = fact_mask.unsqueeze(-1)  # [B, max_facts, 1]
        fact_vectors = fact_vectors * mask_expanded

        if self.num_slots == 1:
            # Masked sum / count -> mean pooling
            denom = mask_expanded.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1, 1]
            v_pool = fact_vectors.sum(dim=1, keepdim=True) / denom        # [B, 1, latent_dim]
            return v_pool
        else:
            # Distribute facts round-robin across N slots
            # Slot indices for each fact index
            slot_idx = (torch.arange(max_facts, device=facts.device) % self.num_slots) # [max_facts]
            
            V_0 = torch.zeros((B, self.num_slots, self.latent_dim), device=facts.device, dtype=fact_vectors.dtype)
            # Slot-wise accumulation
            for s in range(self.num_slots):
                s_mask = (slot_idx == s).float().unsqueeze(0).unsqueeze(-1) # [1, max_facts, 1]
                slot_facts = fact_vectors * s_mask
                slot_counts = (mask_expanded * s_mask).sum(dim=1).clamp(min=1.0) # [B, 1]
                V_0[:, s, :] = slot_facts.sum(dim=1) / slot_counts

            return V_0
