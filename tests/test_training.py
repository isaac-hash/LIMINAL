import torch
from torch.utils.data import DataLoader
from src.utils.config import Config, ModelConfig, DataConfig, TrainingConfig
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary, ReasoningDataset, collate_reasoning_batch
from src.models.model import ReasoningModel
from src.training.losses import LossComputer
from src.training.trainer import Trainer


def test_vector_baseline_forward_shapes():
    vocab = Vocabulary()
    dummy_records = [
        {
            "id": "test_1",
            "input_facts": [
                {"type": "entity", "name": "Alice"},
                {"type": "attribute", "entity": "Alice", "key": "money", "value": 50},
                {"type": "operation", "op": "add", "entity": "Alice", "key": "money", "value": 20},
                {"type": "attribute", "entity": "shoe", "key": "price", "value": 70},
            ],
            "ground_truth": 1,
        }
    ]
    vocab.build_from_records(dummy_records)

    config = ModelConfig(type="vector", latent_slots=1, latent_dim=64, reasoning_steps=3, num_classes=2)
    model = ReasoningModel(config, vocab)

    dataset = ReasoningDataset(dummy_records, vocab)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_reasoning_batch)
    batch = next(iter(loader))

    logits, info = model(batch["facts"], batch["fact_mask"])

    assert logits.shape == (1, 2)
    assert info["V_0"].shape == (1, 1, 64)
    assert len(info["trajectory"]) == 4  # initial + 3 steps
    assert info["h_final"].shape == (1, 64)


def test_graph_baseline_forward_shapes():
    vocab = Vocabulary()
    dummy_records = [
        {
            "id": "test_1",
            "input_facts": [
                {"type": "entity", "name": "Alice"},
                {"type": "attribute", "entity": "Alice", "key": "money", "value": 50},
                {"type": "operation", "op": "add", "entity": "Alice", "key": "money", "value": 20},
                {"type": "attribute", "entity": "shoe", "key": "price", "value": 70},
            ],
            "ground_truth": 1,
        }
    ]
    vocab.build_from_records(dummy_records)

    config = ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=4, num_classes=2)
    model = ReasoningModel(config, vocab)

    dataset = ReasoningDataset(dummy_records, vocab)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_reasoning_batch)
    batch = next(iter(loader))

    logits, info = model(batch["facts"], batch["fact_mask"])

    assert logits.shape == (1, 2)
    assert info["V_0"].shape == (1, 8, 32)
    assert len(info["trajectory"]) == 5  # initial + 4 steps
    assert info["V_final"].shape == (1, 8, 32)
    assert info["h_final"].shape == (1, 32)


def test_vector_baseline_loss_decreases(tmp_path):
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
        model=ModelConfig(type="vector", latent_slots=1, latent_dim=64, reasoning_steps=2),
        training=TrainingConfig(epochs=6, batch_size=16, lr=0.01, checkpoint_dir=str(tmp_path)),
    )

    model = ReasoningModel(full_cfg.model, vocab)
    loss_computer = LossComputer(full_cfg)
    trainer = Trainer(model, train_loader, val_loader, loss_computer, full_cfg, device="cpu")

    logs = trainer.train()
    assert len(logs) == 6
    initial_loss = logs[0]["train_loss"]
    final_loss = logs[-1]["train_loss"]
    assert final_loss < initial_loss, f"Initial loss {initial_loss} did not decrease; final: {final_loss}"


def test_graph_baseline_loss_decreases(tmp_path):
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
        training=TrainingConfig(epochs=15, batch_size=16, lr=0.005, scheduler="none", checkpoint_dir=str(tmp_path)),
    )

    model = ReasoningModel(full_cfg.model, vocab)
    loss_computer = LossComputer(full_cfg)
    trainer = Trainer(model, train_loader, val_loader, loss_computer, full_cfg, device="cpu")

    logs = trainer.train()
    assert len(logs) == 15
    initial_loss = logs[0]["train_loss"]
    final_loss = min(log["train_loss"] for log in logs[5:])
    assert final_loss < initial_loss, f"Initial loss {initial_loss} did not decrease; min final: {final_loss}"
