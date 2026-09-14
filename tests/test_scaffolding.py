from pathlib import Path
import pytest
import torch
import torch.nn as nn
from dataclasses import FrozenInstanceError

from src.utils.device import get_device, print_hardware_info
from src.utils.config import load_config, Config, set_seed
from src.utils.checkpoint import save_checkpoint, load_checkpoint


def test_device_detection():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")

    printed_device = print_hardware_info()
    assert printed_device == device


def test_config_loading_base():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config = load_config(config_path)

    assert isinstance(config, Config)
    assert config.model.type == "graph"
    assert config.model.latent_slots == 8
    assert config.model.latent_dim == 32
    assert config.activity.enabled is False
    assert config.resolution.enabled is False
    assert config.external.enabled is False
    assert config.persistence.enabled is False
    assert config.data.task_family == "affordability"
    assert config.training.batch_size == 16
    assert config.seed == 42


def test_config_immutability():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    config = load_config(config_path)

    with pytest.raises(FrozenInstanceError):
        config.model.latent_slots = 16  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        config.seed = 999  # type: ignore[misc]


def test_config_overrides():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
    overrides = {
        "model": {"type": "vector", "latent_slots": 1, "latent_dim": 256},
        "training": {"epochs": 10},
    }
    config = load_config(config_path, overrides=overrides)

    assert config.model.type == "vector"
    assert config.model.latent_slots == 1
    assert config.model.latent_dim == 256
    assert config.training.epochs == 10
    # Non-overridden fields retain defaults:
    assert config.model.num_msg_layers == 1
    assert config.training.batch_size == 16


def test_checkpoint_roundtrip(tmp_path):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x):
            return self.linear(x)

    model1 = SimpleModel()
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
    
    # Perform a dummy update so optimizer has state
    x = torch.randn(2, 4)
    out = model1(x).sum()
    out.backward()
    optimizer1.step()

    ckpt_file = tmp_path / "checkpoints" / "best.pt"
    metrics = {"loss": 0.123, "val_acc": 0.95}

    save_checkpoint(
        model=model1,
        optimizer=optimizer1,
        epoch=5,
        metrics=metrics,
        path=ckpt_file,
        config={"test": True},
    )

    assert ckpt_file.exists()

    model2 = SimpleModel()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)

    epoch, loaded_metrics, config = load_checkpoint(
        ckpt_file, model=model2, optimizer=optimizer2, device="cpu"
    )

    assert epoch == 5
    assert loaded_metrics["val_acc"] == 0.95
    assert config == {"test": True}

    # Verify weights match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_set_seed():
    set_seed(1234)
    t1 = torch.randn(5)
    set_seed(1234)
    t2 = torch.randn(5)
    assert torch.allclose(t1, t2)
