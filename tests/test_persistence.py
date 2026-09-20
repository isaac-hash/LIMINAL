"""Tests for Phase 4: Multi-turn Persistence in LIMINAL.

Test groups:
  1. Unit tests (PersistenceGate):
     - Output shape
     - Value ranges in (0, 1)
     - Differentiability across inputs and parameters
     - Zero-prior behavior
  2. Data tests:
     - Affordability sequence generation structure and consistency
     - Sequence dataset collation tensor shapes
  3. Integration tests (SequentialReasoningModel):
     - Forward pass tensor shapes
     - Backward compatibility when persistence is disabled
  4. Training loop:
     - Sequential training loss decreases
     - Persistence vs reset evaluation
"""

import pytest
import torch
from torch.utils.data import DataLoader

from src.utils.config import (
    Config,
    ModelConfig,
    ActivityConfig,
    ResolutionConfig,
    PersistenceConfig,
    DataConfig,
    TrainingConfig,
)
from src.models.persistence import PersistenceGate
from src.models.sequential_model import SequentialReasoningModel
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary
from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
from src.training.sequential_trainer import SequentialTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_sequence_records():
    records = [
        {
            "sequence_id": "seq_0",
            "task_family": "affordability_sequence",
            "turns": [
                {
                    "id": "seq_0_t0",
                    "task_family": "affordability_sequence",
                    "turn_index": 0,
                    "sequence_id": "seq_0",
                    "input_facts": [
                        {"type": "entity", "name": "Alice"},
                        {"type": "attribute", "entity": "Alice", "key": "money", "value": 50},
                        {"type": "attribute", "entity": "shoe", "key": "price", "value": 40},
                    ],
                    "ground_truth": 1,
                    "proof_trace": [],
                },
                {
                    "id": "seq_0_t1",
                    "task_family": "affordability_sequence",
                    "turn_index": 1,
                    "sequence_id": "seq_0",
                    "input_facts": [
                        {"type": "entity", "name": "Alice"},
                        {"type": "attribute", "entity": "Alice", "key": "money", "value": 50},
                        {"type": "operation", "op": "sub", "entity": "Alice", "key": "money", "value": 20},
                        {"type": "attribute", "entity": "shoe", "key": "price", "value": 40},
                    ],
                    "ground_truth": 0,
                    "proof_trace": [],
                },
            ],
        },
        {
            "sequence_id": "seq_1",
            "task_family": "affordability_sequence",
            "turns": [
                {
                    "id": "seq_1_t0",
                    "task_family": "affordability_sequence",
                    "turn_index": 0,
                    "sequence_id": "seq_1",
                    "input_facts": [
                        {"type": "entity", "name": "Bob"},
                        {"type": "attribute", "entity": "Bob", "key": "money", "value": 30},
                        {"type": "attribute", "entity": "book", "key": "price", "value": 25},
                    ],
                    "ground_truth": 1,
                    "proof_trace": [],
                },
                {
                    "id": "seq_1_t1",
                    "task_family": "affordability_sequence",
                    "turn_index": 1,
                    "sequence_id": "seq_1",
                    "input_facts": [
                        {"type": "entity", "name": "Bob"},
                        {"type": "attribute", "entity": "Bob", "key": "money", "value": 30},
                        {"type": "operation", "op": "add", "entity": "Bob", "key": "money", "value": 10},
                        {"type": "attribute", "entity": "book", "key": "price", "value": 25},
                    ],
                    "ground_truth": 1,
                    "proof_trace": [],
                },
            ],
        },
    ]
    vocab = Vocabulary()
    vocab.build_from_records(records)
    return records, vocab


# ---------------------------------------------------------------------------
# 1. Unit Tests: PersistenceGate
# ---------------------------------------------------------------------------

def test_persistence_gate_output_shape():
    """1. PersistenceGate output shape should match [B, N, d]."""
    B, N, d = 4, 8, 32
    gate = PersistenceGate(latent_dim=d, gate_hidden_dim=16)
    V_prior = torch.randn(B, N, d)
    V_new = torch.randn(B, N, d)

    V_blended = gate(V_prior, V_new)
    assert V_blended.shape == (B, N, d)


def test_persistence_gate_values_in_range():
    """2. Gate activations must be strictly within (0, 1) per slot."""
    B, N, d = 4, 8, 32
    gate = PersistenceGate(latent_dim=d, gate_hidden_dim=16)
    V_prior = torch.randn(B, N, d)
    V_new = torch.randn(B, N, d)

    gates = gate.compute_gate(V_prior, V_new)
    assert gates.shape == (B, N)
    assert (gates >= 0.0).all() and (gates <= 1.0).all()

    V_blended, gates_ret = gate(V_prior, V_new, return_gates=True)
    assert torch.equal(gates, gates_ret)
    assert V_blended.shape == (B, N, d)


def test_persistence_gate_differentiable():
    """3. Gradients must flow through inputs and gate parameters."""
    B, N, d = 2, 4, 16
    gate = PersistenceGate(latent_dim=d, gate_hidden_dim=8)
    V_prior = torch.randn(B, N, d, requires_grad=True)
    V_new = torch.randn(B, N, d, requires_grad=True)

    V_blended = gate(V_prior, V_new)
    loss = V_blended.sum()
    loss.backward()

    assert V_prior.grad is not None and torch.isfinite(V_prior.grad).all()
    assert V_new.grad is not None and torch.isfinite(V_new.grad).all()
    for p in gate.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_persistence_gate_zero_prior():
    """4. When V_prior is all zeros, output should be close to V_new."""
    B, N, d = 3, 6, 24
    gate = PersistenceGate(latent_dim=d, gate_hidden_dim=12, init_bias=-3.0)
    V_prior = torch.zeros(B, N, d)
    V_new = torch.randn(B, N, d)

    gates = gate.compute_gate(V_prior, V_new)
    # With init_bias=-3.0, gates should be small (< 0.2)
    assert (gates < 0.25).all()

    V_blended = gate(V_prior, V_new)
    # Blended output should be very close to V_new
    cos_sim = torch.cosine_similarity(V_blended.view(-1, d), V_new.view(-1, d), dim=-1)
    assert (cos_sim > 0.95).all()


# ---------------------------------------------------------------------------
# 2. Data Tests
# ---------------------------------------------------------------------------

def test_affordability_sequence_generation():
    """5. Synthetic sequence generator produces expected structure and consistency."""
    data_cfg = DataConfig(
        task_family="affordability_sequence",
        sequence_turns=3,
        ops_per_turn=1,
        num_train=10,
        num_val=5,
        num_test=5,
        seed=123,
    )
    generator = ArithmeticGenerator(config=data_cfg, seed=123)
    splits = generator.generate_dataset()

    assert "train" in splits and "val" in splits and "test" in splits
    assert len(splits["train"]) == 10

    seq = splits["train"][0]
    assert "sequence_id" in seq
    assert len(seq["turns"]) == 3

    # Check shared protagonist entity and item across all turns in a sequence
    turn_0 = seq["turns"][0]
    entity_names = [f["name"] for f in turn_0["input_facts"] if f.get("type") == "entity"]
    protagonist = entity_names[0]

    for t_idx, turn in enumerate(seq["turns"]):
        assert turn["turn_index"] == t_idx
        assert turn["ground_truth"] in (0, 1)
        # Verify protagonist is in input facts
        turn_entities = [f.get("name") or f.get("entity") for f in turn["input_facts"] if f.get("name") or f.get("entity")]
        assert protagonist in turn_entities


def test_affordability_sequence_incremental_generation():
    """Verify that incremental mode omits start_money and prior ops in turns t > 0."""
    data_cfg = DataConfig(
        task_family="affordability_sequence",
        sequence_turns=3,
        ops_per_turn=1,
        incremental_turns=True,
        num_train=5,
        num_val=2,
        num_test=2,
        seed=42,
    )
    generator = ArithmeticGenerator(config=data_cfg, seed=42)
    splits = generator.generate_dataset()
    seq = splits["train"][0]

    # Turn 0 must have start_money attribute
    turn_0 = seq["turns"][0]
    has_start_money_t0 = any(f.get("type") == "attribute" and f.get("key") == "money" for f in turn_0["input_facts"])
    assert has_start_money_t0

    # Turn 1 and Turn 2 must NOT have start_money attribute (only new ops and price)
    for t_idx in (1, 2):
        turn_t = seq["turns"][t_idx]
        has_start_money_t = any(f.get("type") == "attribute" and f.get("key") == "money" for f in turn_t["input_facts"])
        assert not has_start_money_t, f"Turn {t_idx} should not contain start money in incremental mode"
        # Must have operation and item price
        has_op = any(f.get("type") == "operation" for f in turn_t["input_facts"])
        has_price = any(f.get("type") == "attribute" and f.get("key") == "price" for f in turn_t["input_facts"])
        assert has_op and has_price



def test_sequence_dataset_collation():
    """6. Sequence dataset and batch collation produce expected tensor shapes."""
    records, vocab = _make_dummy_sequence_records()
    dataset = SequenceReasoningDataset(records, vocab)
    assert len(dataset) == 2

    batch = [dataset[0], dataset[1]]
    collated = collate_sequence_batch(batch)

    assert "facts" in collated
    assert "fact_mask" in collated
    assert "turn_mask" in collated
    assert "labels" in collated

    B, T, F, _ = collated["facts"].shape
    assert B == 2
    assert T == 2
    assert F >= 3
    assert collated["fact_mask"].shape == (B, T, F)
    assert collated["turn_mask"].shape == (B, T)
    assert collated["labels"].shape == (B, T)
    assert (collated["turn_mask"] == 1.0).all()


# ---------------------------------------------------------------------------
# 3. Integration Tests: SequentialReasoningModel
# ---------------------------------------------------------------------------

def test_sequential_model_forward_shapes():
    """7. SequentialReasoningModel outputs [B, max_turns, num_classes] and per-turn info."""
    records, vocab = _make_dummy_sequence_records()
    dataset = SequenceReasoningDataset(records, vocab)
    batch = collate_sequence_batch([dataset[0], dataset[1]])

    m_cfg = ModelConfig(type="graph", latent_slots=4, latent_dim=16, num_classes=2, reasoning_steps=2)
    p_cfg = PersistenceConfig(enabled=True, gate_hidden_dim=8)

    model = SequentialReasoningModel(
        config=m_cfg,
        vocab=vocab,
        persistence_config=p_cfg,
    )

    logits, all_infos = model(batch["facts"], batch["fact_mask"], batch["turn_mask"])
    assert logits.shape == (2, 2, 2)  # [B=2, T=2, num_classes=2]
    assert len(all_infos) == 2

    # Turn 0 has no prior state, so no persistence_gates
    assert "persistence_gates" not in all_infos[0]
    # Turn 1 blended with prior state, so persistence_gates exists
    assert "persistence_gates" in all_infos[1]
    assert all_infos[1]["persistence_gates"].shape == (2, 4)


def test_sequential_model_no_persistence_compat():
    """8. When persistence is disabled, model runs cleanly with independent turns."""
    records, vocab = _make_dummy_sequence_records()
    dataset = SequenceReasoningDataset(records, vocab)
    batch = collate_sequence_batch([dataset[0], dataset[1]])

    m_cfg = ModelConfig(type="graph", latent_slots=4, latent_dim=16, num_classes=2, reasoning_steps=2)
    p_cfg = PersistenceConfig(enabled=False)

    model = SequentialReasoningModel(
        config=m_cfg,
        vocab=vocab,
        persistence_config=p_cfg,
    )
    assert model.persistence_gate is None

    logits, all_infos = model(batch["facts"], batch["fact_mask"], batch["turn_mask"])
    assert logits.shape == (2, 2, 2)
    assert len(all_infos) == 2
    assert "persistence_gates" not in all_infos[1]


# ---------------------------------------------------------------------------
# 4. Training Loop Tests
# ---------------------------------------------------------------------------

def test_persistence_loss_decreases(tmp_path):
    """9. SequentialTrainer training loop runs and decreases loss over epochs."""
    torch.manual_seed(42)
    data_cfg = DataConfig(
        task_family="affordability_sequence",
        sequence_turns=2,
        ops_per_turn=1,
        num_train=24,
        num_val=8,
        num_test=8,
        seed=42,
    )
    generator = ArithmeticGenerator(config=data_cfg, seed=42)
    splits = generator.generate_dataset()

    vocab = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"])

    train_ds = SequenceReasoningDataset(splits["train"], vocab)
    val_ds = SequenceReasoningDataset(splits["val"], vocab)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_sequence_batch)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_sequence_batch)

    cfg = Config(
        model=ModelConfig(type="graph", latent_slots=4, latent_dim=16, num_classes=2, reasoning_steps=2),
        persistence=PersistenceConfig(enabled=True, gate_hidden_dim=8),
        training=TrainingConfig(epochs=6, batch_size=8, lr=5e-3, checkpoint_dir=str(tmp_path)),
    )

    model = SequentialReasoningModel(
        config=cfg.model,
        vocab=vocab,
        persistence_config=cfg.persistence,
    )

    trainer = SequentialTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=torch.device("cpu"),
    )

    log = trainer.train()
    assert len(log) == 6
    # Loss should decrease from first epoch to last epoch
    assert log[-1]["train_loss"] < log[0]["train_loss"]


def test_persistence_vs_reset_accuracy(tmp_path):
    """10. Persistent model vs reset model evaluation on multi-turn data."""
    torch.manual_seed(42)
    data_cfg = DataConfig(
        task_family="affordability_sequence",
        sequence_turns=2,
        ops_per_turn=1,
        num_train=16,
        num_val=8,
        num_test=8,
        seed=42,
    )
    generator = ArithmeticGenerator(config=data_cfg, seed=42)
    splits = generator.generate_dataset()

    vocab = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"])

    val_ds = SequenceReasoningDataset(splits["val"], vocab)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_sequence_batch)

    cfg_pers = Config(
        model=ModelConfig(type="graph", latent_slots=4, latent_dim=16, num_classes=2, reasoning_steps=2),
        persistence=PersistenceConfig(enabled=True, gate_hidden_dim=8),
        training=TrainingConfig(epochs=1, batch_size=8, checkpoint_dir=str(tmp_path / "pers")),
    )
    cfg_reset = Config(
        model=ModelConfig(type="graph", latent_slots=4, latent_dim=16, num_classes=2, reasoning_steps=2),
        persistence=PersistenceConfig(enabled=False),
        training=TrainingConfig(epochs=1, batch_size=8, checkpoint_dir=str(tmp_path / "reset")),
    )

    model_pers = SequentialReasoningModel(cfg_pers.model, vocab, persistence_config=cfg_pers.persistence)
    model_reset = SequentialReasoningModel(cfg_reset.model, vocab, persistence_config=cfg_reset.persistence)

    trainer_pers = SequentialTrainer(model_pers, val_loader, val_loader, cfg_pers, torch.device("cpu"))
    trainer_reset = SequentialTrainer(model_reset, val_loader, val_loader, cfg_reset, torch.device("cpu"))

    eval_pers = trainer_pers.evaluate(val_loader)
    eval_reset = trainer_reset.evaluate(val_loader)

    assert "acc" in eval_pers and "acc_turn_1" in eval_pers and "acc_turn_2" in eval_pers
    assert "acc" in eval_reset and "acc_turn_1" in eval_reset and "acc_turn_2" in eval_reset
