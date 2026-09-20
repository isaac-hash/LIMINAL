from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from src.utils.config import ModelConfig, ActivityConfig, ResolutionConfig
from src.data.dataset import Vocabulary
from src.models.encoder import Encoder
from src.models.latent_workspace import LatentWorkspace
from src.models.decoder import Decoder


class ReasoningModel(nn.Module):
    """End-to-end reasoning model composing Encoder -> LatentWorkspace -> Decoder.

    Passes activity_config through to LatentWorkspace so that Phase 2 activity
    gates are instantiated when activity.enabled=True, without changing any
    external interface for Phase 1 callers (activity_config defaults to None).
    """

    def __init__(
        self,
        config: ModelConfig,
        vocab: Vocabulary,
        activity_config: ActivityConfig | None = None,
        resolution_config: ResolutionConfig | None = None,
    ):
        super().__init__()
        self.config = config

        num_slots = 1 if config.type == "vector" else config.latent_slots

        self.encoder = Encoder(
            type_vocab_size=len(vocab.fact_types) + 2,
            entity_vocab_size=len(vocab.entities) + 2,
            key_vocab_size=len(vocab.keys) + 2,
            op_vocab_size=len(vocab.ops) + 2,
            embed_dim=16,
            latent_dim=config.latent_dim,
            num_slots=num_slots,
        )

        self.workspace = LatentWorkspace(config, activity_config=activity_config, resolution_config=resolution_config)

        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_classes=config.num_classes,
        )

    def forward(self, facts: Tensor, fact_mask: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """
        Args:
            facts: Tensor[B, max_facts, 5]
            fact_mask: Tensor[B, max_facts]

        Returns:
            logits: Tensor[B, num_classes]
            info: dict containing intermediate trajectories and states
        """
        V_0 = self.encoder(facts, fact_mask)
        h_final, info = self.workspace(V_0)
        logits = self.decoder(h_final)

        info["V_0"] = V_0
        info["logits"] = logits
        return logits, info
