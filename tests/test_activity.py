"""Tests for Phase 2A: ActivityGate and adaptive LatentWorkspace."""
import torch
from torch.utils.data import DataLoader
from src.utils.config import Config, ModelConfig, ActivityConfig, DataConfig, TrainingConfig
from src.models.activity import ActivityGate
from src.models.latent_workspace import LatentWorkspace
from src.models.model import ReasoningModel
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary, ReasoningDataset, collate_reasoning_batch
from src.training.losses import LossComputer
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Unit tests: ActivityGate
# ---------------------------------------------------------------------------

def test_activity_gate_output_shape():
    """ActivityGate must return [B, N] in (0, 1)."""
    B, N, d = 4, 8, 32
    gate = ActivityGate(latent_dim=d, gate_hidden_dim=16)
    V = torch.randn(B, N, d)
    A = gate(V)

    assert A.shape == (B, N), f"Expected ({B}, {N}), got {A.shape}"
    assert A.min() >= 0.0 and A.max() <= 1.0, "Activity gate must output values in [0, 1]"


def test_activity_gate_differentiable():
    """Gradients must flow back through the gate."""
    B, N, d = 2, 8, 32
    gate = ActivityGate(latent_dim=d, gate_hidden_dim=16)
    V = torch.randn(B, N, d, requires_grad=True)
    A = gate(V)
    loss = A.sum()
    loss.backward()
    assert V.grad is not None, "Gradient did not flow through ActivityGate"
    assert not torch.all(V.grad == 0), "ActivityGate gradient is all zeros"


# ---------------------------------------------------------------------------
# Integration tests: LatentWorkspace with activity gates
# ---------------------------------------------------------------------------

def test_workspace_activity_gate_forward_shapes():
    """LatentWorkspace in graph+activity mode: shapes must be correct."""
    B, N, d = 3, 8, 32
    model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=d)
    activity_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    workspace = LatentWorkspace(model_cfg, activity_config=activity_cfg)

    V_0 = torch.randn(B, N, d)
    h_final, info = workspace(V_0)

    assert h_final.shape == (B, d), f"h_final shape wrong: {h_final.shape}"
    assert info["V_final"].shape == (B, N, d)
    assert info["A_final"].shape == (B, N)
    # One activity snapshot per reasoning step
    assert len(info["activity_trajectory"]) == model_cfg.reasoning_steps


def test_workspace_static_backward_compat():
    """When activity is disabled, LatentWorkspace must behave exactly as Phase 1."""
    B, N, d = 2, 8, 32
    model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=d, reasoning_steps=4)
    # No activity_config — static ones mode
    workspace = LatentWorkspace(model_cfg)

    V_0 = torch.randn(B, N, d)
    h_final, info = workspace(V_0)

    # A_final must be all-ones in static mode
    assert info["A_final"].shape == (B, N)
    assert torch.allclose(
        info["A_final"],
        torch.ones(B, N),
        atol=1e-6,
    ), "Static mode: A_final should be all-ones"


# ---------------------------------------------------------------------------
# Full model integration: ReasoningModel with activity_config
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
        }
    ]
    vocab = Vocabulary()
    vocab.build_from_records(records)
    return records, vocab


def test_reasoning_model_activity_forward():
    """ReasoningModel with activity_config=ActivityConfig(enabled=True) runs end-to-end."""
    records, vocab = _make_dummy_records_and_vocab()
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=4)
    activity_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    model = ReasoningModel(model_cfg, vocab, activity_config=activity_cfg)

    dataset = ReasoningDataset(records, vocab)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_reasoning_batch)
    batch = next(iter(loader))

    logits, info = model(batch["facts"], batch["fact_mask"])

    assert logits.shape == (1, 2)
    assert info["A_final"].shape == (1, 8), "A_final must be present in activity mode"
    assert len(info["activity_trajectory"]) == 4, "Must record one A per reasoning step"


def test_reasoning_model_no_activity_compat():
    """ReasoningModel with no activity_config still returns correct shapes (Phase 1 compat)."""
    records, vocab = _make_dummy_records_and_vocab()
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=4)
    model = ReasoningModel(model_cfg, vocab)  # no activity_config

    dataset = ReasoningDataset(records, vocab)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_reasoning_batch)
    batch = next(iter(loader))

    logits, info = model(batch["facts"], batch["fact_mask"])

    assert logits.shape == (1, 2)
    assert info["V_final"].shape == (1, 8, 32)


# ---------------------------------------------------------------------------
# Training loop: activity loss decreases with gates active
# ---------------------------------------------------------------------------

def test_activity_graph_loss_decreases(tmp_path):
    """Full training loop with activity gates ON: loss must decrease."""
    gen_cfg = DataConfig(num_train=80, num_val=20, num_test=20, task_family="affordability")
    gen = ArithmeticGenerator(config=gen_cfg, seed=42)
    splits = gen.generate_dataset()

    vocab = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"])

    train_loader = DataLoader(
        ReasoningDataset(splits["train"], vocab),
        batch_size=16,
        shuffle=True,
        collate_fn=collate_reasoning_batch,
    )
    val_loader = DataLoader(
        ReasoningDataset(splits["val"], vocab),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
    )

    full_cfg = Config(
        model=ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=2),
        activity=ActivityConfig(enabled=True, gate_hidden_dim=16, sparsity_lambda=0.01, entropy_lambda=0.001),
        training=TrainingConfig(
            epochs=15,
            batch_size=16,
            lr=0.005,
            scheduler="none",
            checkpoint_dir=str(tmp_path),
        ),
    )

    model = ReasoningModel(full_cfg.model, vocab, activity_config=full_cfg.activity)
    loss_computer = LossComputer(full_cfg)
    trainer = Trainer(model, train_loader, val_loader, loss_computer, full_cfg, device="cpu")

    logs = trainer.train()
    assert len(logs) == 15
    initial_loss = logs[0]["train_loss"]
    final_loss = min(log["train_loss"] for log in logs[5:])
    assert final_loss < initial_loss, (
        f"Activity graph loss did not decrease. Initial: {initial_loss:.4f}, Min final: {final_loss:.4f}"
    )
