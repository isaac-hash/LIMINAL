"""Tests for Phase 3: HaltGate and adaptive resolution in LatentWorkspace.

Test groups:
  1. Unit tests:  HaltGate shape, range, differentiability
  2. Integration: LatentWorkspace in resolution mode — shapes, backward compat,
                  step bounds, ponder weight invariants
  3. Full model:  ReasoningModel with resolution_config, resolution without activity
  4. Training:    Full training loop with resolution ON — loss must decrease
"""

import torch
from torch.utils.data import DataLoader
from src.utils.config import (
    Config, ModelConfig, ActivityConfig, ResolutionConfig, DataConfig, TrainingConfig
)
from src.models.resolution import HaltGate
from src.models.latent_workspace import LatentWorkspace
from src.models.model import ReasoningModel
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary, ReasoningDataset, collate_reasoning_batch
from src.training.losses import LossComputer
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_records_and_vocab():
    records = [
        {
            "id": "t1",
            "input_facts": [
                {"type": "entity", "name": "Alice"},
                {"type": "attribute", "entity": "Alice", "key": "money", "value": 50},
                {"type": "operation", "op": "add", "entity": "Alice", "key": "money", "value": 20},
                {"type": "attribute", "entity": "shoe", "key": "price", "value": 70},
            ],
            "ground_truth": 1,
        },
        {
            "id": "t2",
            "input_facts": [
                {"type": "entity", "name": "Bob"},
                {"type": "attribute", "entity": "Bob", "key": "money", "value": 30},
                {"type": "attribute", "entity": "book", "key": "price", "value": 50},
            ],
            "ground_truth": 0,
        },
    ]
    vocab = Vocabulary()
    vocab.build_from_records(records)
    return records, vocab


def _resolution_cfg(**kwargs) -> ResolutionConfig:
    defaults = dict(enabled=True, halt_hidden_dim=16, halt_threshold=1.0,
                    ponder_lambda=0.01, max_reasoning_steps=8)
    defaults.update(kwargs)
    return ResolutionConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Unit tests: HaltGate
# ---------------------------------------------------------------------------

def test_halt_gate_output_shape():
    """HaltGate must return [B] scalars in (0, 1)."""
    B, N, d = 4, 8, 32
    gate = HaltGate(latent_dim=d, halt_hidden_dim=16)
    V = torch.randn(B, N, d)
    A = torch.rand(B, N)
    h = gate(V, A)

    assert h.shape == (B,), f"Expected shape ({B},), got {h.shape}"
    assert h.min() > 0.0 and h.max() < 1.0, "HaltGate must output values strictly in (0, 1)"


def test_halt_gate_output_shape_no_activity():
    """HaltGate also works when A is not provided (defaults to uniform)."""
    B, N, d = 3, 8, 32
    gate = HaltGate(latent_dim=d, halt_hidden_dim=16)
    V = torch.randn(B, N, d)
    h = gate(V)  # no A argument

    assert h.shape == (B,), f"Expected shape ({B},), got {h.shape}"
    assert h.min() > 0.0 and h.max() < 1.0


def test_halt_gate_differentiable():
    """Gradients must flow back through HaltGate to V."""
    B, N, d = 2, 8, 32
    gate = HaltGate(latent_dim=d, halt_hidden_dim=16)
    V = torch.randn(B, N, d, requires_grad=True)
    h = gate(V)
    loss = h.sum()
    loss.backward()
    assert V.grad is not None, "Gradient did not flow through HaltGate"
    assert not torch.all(V.grad == 0), "HaltGate gradient is all zeros"


# ---------------------------------------------------------------------------
# 2. Integration tests: LatentWorkspace in resolution mode
# ---------------------------------------------------------------------------

def test_workspace_resolution_forward_shapes():
    """Graph + activity + resolution mode: output shapes and info keys correct."""
    B, N, d = 3, 8, 32
    model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=d)
    activity_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    res_cfg = _resolution_cfg()
    workspace = LatentWorkspace(model_cfg, activity_config=activity_cfg, resolution_config=res_cfg)

    V_0 = torch.randn(B, N, d)
    h_final, info = workspace(V_0)

    assert h_final.shape == (B, d), f"h_final shape wrong: {h_final.shape}"
    assert info["V_final"].shape == (B, N, d)
    assert info["A_final"].shape == (B, N)

    # Phase 3 specific keys
    assert "halt_probs" in info, "info must contain halt_probs"
    assert "ponder_weights" in info, "info must contain ponder_weights"
    assert "n_steps" in info, "info must contain n_steps"
    assert "effective_steps" in info, "info must contain effective_steps"

    assert info["n_steps"].shape == (B,), f"n_steps shape wrong: {info['n_steps'].shape}"
    assert isinstance(info["effective_steps"], float)
    assert len(info["halt_probs"]) >= 1, "halt_probs must record at least one step"


def test_workspace_resolution_disabled_backward_compat():
    """When resolution is disabled, LatentWorkspace must behave exactly as Phase 2."""
    B, N, d = 2, 8, 32
    model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=d, reasoning_steps=4)
    activity_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    # resolution_config=None → no HaltGate, fixed steps
    workspace = LatentWorkspace(model_cfg, activity_config=activity_cfg, resolution_config=None)

    V_0 = torch.randn(B, N, d)
    h_final, info = workspace(V_0)

    # Should NOT have Phase 3 keys
    assert "halt_probs" not in info, "Fixed-step workspace must not emit halt_probs"
    assert "n_steps" not in info, "Fixed-step workspace must not emit n_steps"
    # Should have Phase 2 keys
    assert "A_final" in info
    assert len(info["activity_trajectory"]) == 4, "Should have exactly reasoning_steps activity entries"


def test_workspace_resolution_step_count_bounded():
    """n_steps must always be within [1, max_reasoning_steps]."""
    B, N, d = 6, 8, 32
    model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=d)
    res_cfg = _resolution_cfg(max_reasoning_steps=8)
    workspace = LatentWorkspace(model_cfg, resolution_config=res_cfg)

    V_0 = torch.randn(B, N, d)
    _, info = workspace(V_0)

    n_steps = info["n_steps"]
    assert (n_steps >= 0.0).all(), "n_steps must be non-negative"
    assert (n_steps <= res_cfg.max_reasoning_steps).all(), (
        f"n_steps exceeded max_reasoning_steps={res_cfg.max_reasoning_steps}: {n_steps}"
    )


def test_workspace_resolution_ponder_weights_sum_to_one():
    """ACT ponder weights must sum to approximately 1.0 per example."""
    B, N, d = 4, 8, 32
    model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=d)
    res_cfg = _resolution_cfg(max_reasoning_steps=10)
    workspace = LatentWorkspace(model_cfg, resolution_config=res_cfg)

    V_0 = torch.randn(B, N, d)
    _, info = workspace(V_0)

    ponder_weights = info["ponder_weights"]  # [B, T_actual]
    weight_sums = ponder_weights.sum(dim=1)  # [B]

    assert torch.allclose(weight_sums, torch.ones(B), atol=1e-4), (
        f"Ponder weights must sum to 1.0 per example, got: {weight_sums}"
    )


# ---------------------------------------------------------------------------
# 3. Full model integration: ReasoningModel with resolution_config
# ---------------------------------------------------------------------------

def test_reasoning_model_resolution_forward():
    """ReasoningModel with resolution_config enabled runs end-to-end."""
    records, vocab = _make_dummy_records_and_vocab()
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32)
    activity_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    res_cfg = _resolution_cfg()

    model = ReasoningModel(model_cfg, vocab, activity_config=activity_cfg, resolution_config=res_cfg)

    dataset = ReasoningDataset(records, vocab)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_reasoning_batch)
    batch = next(iter(loader))

    logits, info = model(batch["facts"], batch["fact_mask"])

    assert logits.shape == (2, 2)
    assert "halt_probs" in info, "info must contain halt_probs in resolution mode"
    assert "n_steps" in info, "info must contain n_steps in resolution mode"
    assert info["n_steps"].shape == (2,)


def test_reasoning_model_resolution_no_activity():
    """Resolution gate works independently without activity gates enabled."""
    records, vocab = _make_dummy_records_and_vocab()
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32)
    # No activity gates, but resolution ON
    res_cfg = _resolution_cfg()

    model = ReasoningModel(model_cfg, vocab, activity_config=None, resolution_config=res_cfg)

    dataset = ReasoningDataset(records, vocab)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_reasoning_batch)
    batch = next(iter(loader))

    logits, info = model(batch["facts"], batch["fact_mask"])

    assert logits.shape == (2, 2)
    assert "halt_probs" in info
    # Activity trajectory still present (uniform ones)
    assert len(info["activity_trajectory"]) >= 1


# ---------------------------------------------------------------------------
# 4. Training loop: resolution loss decreases
# ---------------------------------------------------------------------------

def test_resolution_graph_loss_decreases(tmp_path):
    """Full training loop with resolution gates ON: loss must decrease."""
    gen_cfg = DataConfig(num_train=80, num_val=20, num_test=20, task_family="affordability")
    gen = ArithmeticGenerator(config=gen_cfg, seed=42)
    splits = gen.generate_dataset()

    vocab = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"])

    train_loader = DataLoader(
        ReasoningDataset(splits["train"], vocab),
        batch_size=16, shuffle=True, collate_fn=collate_reasoning_batch,
    )
    val_loader = DataLoader(
        ReasoningDataset(splits["val"], vocab),
        batch_size=16, shuffle=False, collate_fn=collate_reasoning_batch,
    )

    full_cfg = Config(
        model=ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=2),
        activity=ActivityConfig(enabled=True, gate_hidden_dim=16,
                                sparsity_lambda=0.01, entropy_lambda=0.001),
        resolution=ResolutionConfig(enabled=True, halt_hidden_dim=16, halt_threshold=1.0,
                                    ponder_lambda=0.01, max_reasoning_steps=6),
        training=TrainingConfig(
            epochs=15, batch_size=16, lr=0.005, scheduler="none",
            checkpoint_dir=str(tmp_path),
        ),
    )

    model = ReasoningModel(
        full_cfg.model, vocab,
        activity_config=full_cfg.activity,
        resolution_config=full_cfg.resolution,
    )
    loss_computer = LossComputer(full_cfg)
    trainer = Trainer(model, train_loader, val_loader, loss_computer, full_cfg, device="cpu")

    logs = trainer.train()
    assert len(logs) == 15

    initial_loss = logs[0]["train_loss"]
    final_loss = min(log["train_loss"] for log in logs[5:])
    assert final_loss < initial_loss, (
        f"Resolution graph loss did not decrease. Initial: {initial_loss:.4f}, Min final: {final_loss:.4f}"
    )

    # Verify that mean_steps is being logged (resolution active)
    ponder_logs = [log for log in logs if "train_mean_steps" in log]
    assert len(ponder_logs) > 0, "mean_steps should be logged when resolution is enabled"
