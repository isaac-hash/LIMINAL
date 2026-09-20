from typing import Any
import torch
import torch.nn as nn
from torch import Tensor
from src.utils.config import ModelConfig, ActivityConfig, ResolutionConfig, PersistenceConfig
from src.data.dataset import Vocabulary
from src.models.model import ReasoningModel
from src.models.persistence import PersistenceGate


class SequentialReasoningModel(nn.Module):
    """Multi-turn sequential reasoning wrapper supporting persistent latent state.

    At each turn t:
      1. New facts are encoded to fresh latent representations V_new.
      2. If persistence is enabled and prior state V_prior exists:
             V_0, gates = persistence_gate(V_prior, V_new)
         Else:
             V_0 = V_new (standard reset behavior)
      3. Reasoning occurs in the latent workspace (fixed or adaptive steps).
      4. V_prior is carried forward to turn t + 1 (detached by default).
    """

    def __init__(
        self,
        config: ModelConfig,
        vocab: Vocabulary,
        activity_config: ActivityConfig | None = None,
        resolution_config: ResolutionConfig | None = None,
        persistence_config: PersistenceConfig | None = None,
    ):
        super().__init__()
        self.config = config
        self.persistence_config = persistence_config or PersistenceConfig()
        self.base_model = ReasoningModel(
            config,
            vocab,
            activity_config=activity_config,
            resolution_config=resolution_config,
        )

        if self.persistence_config.enabled:
            self.persistence_gate = PersistenceGate(
                latent_dim=config.latent_dim,
                gate_hidden_dim=self.persistence_config.gate_hidden_dim,
                init_bias=self.persistence_config.init_bias,
            )
        else:
            self.persistence_gate = None

    def forward(
        self,
        facts_seq: Tensor,
        fact_mask_seq: Tensor,
        turn_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[dict[str, Any]]]:
        """Process a sequence of turns across a batch.

        Args:
            facts_seq:     Tensor[B, max_turns, max_facts, 5]
            fact_mask_seq: Tensor[B, max_turns, max_facts]
            turn_mask:     Tensor[B, max_turns] (optional, default all 1s)

        Returns:
            stacked_logits: Tensor[B, max_turns, num_classes]
            all_infos:      list of per-turn info dictionaries (length max_turns)
        """
        B, max_turns, max_facts, _ = facts_seq.shape
        device = facts_seq.device

        if turn_mask is None:
            turn_mask = torch.ones((B, max_turns), device=device)

        V_prior: Tensor | None = None
        all_logits: list[Tensor] = []
        all_infos: list[dict[str, Any]] = []

        for t in range(max_turns):
            facts_t = facts_seq[:, t]          # [B, max_facts, 5]
            fact_mask_t = fact_mask_seq[:, t]  # [B, max_facts]

            V_new = self.base_model.encoder(facts_t, fact_mask_t)

            gate_values: Tensor | None = None
            if V_prior is not None and self.persistence_gate is not None:
                V_0, gate_values = self.persistence_gate(V_prior, V_new, return_gates=True)
            else:
                V_0 = V_new

            h_final, info = self.base_model.workspace(V_0)
            logits = self.base_model.decoder(h_final)

            info["V_0"] = V_0
            info["logits"] = logits
            info["V_new"] = V_new
            if gate_values is not None:
                info["persistence_gates"] = gate_values

            # Determine final latent slots to carry forward
            V_final = info.get("V_final")
            if V_final is None:
                V_final = h_final.unsqueeze(1)
                info["V_final"] = V_final

            # Prepare prior state for turn t + 1
            if self.persistence_config.detach_between_turns:
                V_next_prior = V_final.detach()
            else:
                V_next_prior = V_final

            # If an example's turn t was masked out, zero its carried state
            valid_t = turn_mask[:, t].view(-1, 1, 1)
            V_prior = torch.where(valid_t > 0, V_next_prior, torch.zeros_like(V_next_prior))

            all_logits.append(logits)
            all_infos.append(info)

        stacked_logits = torch.stack(all_logits, dim=1)  # [B, max_turns, num_classes]
        return stacked_logits, all_infos
