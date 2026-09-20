from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import random
import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class ModelConfig:
    type: str = "graph"
    latent_slots: int = 8
    latent_dim: int = 32
    edge_dim: int = 1
    msg_hidden_dim: int = 64
    update_hidden_dim: int = 64
    num_msg_layers: int = 1
    reasoning_steps: int = 4
    max_reasoning_steps: int = 16
    decoder_hidden_dim: int = 64
    num_classes: int = 2


@dataclass(frozen=True)
class ActivityConfig:
    enabled: bool = False
    gate_hidden_dim: int = 32
    sparsity_lambda: float = 0.01
    entropy_lambda: float = 0.001


@dataclass(frozen=True)
class ResolutionConfig:
    enabled: bool = False
    halt_hidden_dim: int = 32        # Hidden width of the halt gate MLP
    halt_threshold: float = 1.0     # ACT halts when cumulative H_t >= this
    ponder_lambda: float = 0.01     # λ_ponder weight on mean n_steps cost
    max_reasoning_steps: int = 16   # Hard cap on adaptive loop (overrides model.max_reasoning_steps)


@dataclass(frozen=True)
class ExternalConfig:
    enabled: bool = False


@dataclass(frozen=True)
class PersistenceConfig:
    enabled: bool = False
    gate_hidden_dim: int = 32       # Hidden width of per-slot persistence gate MLP
    detach_between_turns: bool = True   # Detach V_prior gradient between turns (no BPTT)
    init_bias: float = -2.0         # Initial bias of persistence gate MLP


@dataclass(frozen=True)
class DataConfig:
    task_family: str = "affordability"  # single family, "mixed", or "affordability_sequence"
    # Mixed-family mode: list of families to interleave (used when task_family="mixed")
    mixed_families: tuple[str, ...] = ("affordability", "multi_step")
    num_train: int = 8000
    num_val: int = 1000
    num_test: int = 1000
    max_entities: int = 4
    max_operations: int = 2
    distractor_ratio: float = 0.0
    seed: int = 42
    # Sequence mode (task_family="affordability_sequence")
    sequence_turns: int = 3         # number of related turns per sequence
    ops_per_turn: int = 1           # operations added to the chain each turn
    incremental_turns: bool = False # if True, turn t>0 contains ONLY new ops (requires persistence)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    log_every: int = 10
    eval_every: int = 1
    checkpoint_dir: str = "results/default"


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    activity: ActivityConfig = field(default_factory=ActivityConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    external: ExternalConfig = field(default_factory=ExternalConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively update nested dictionaries."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML config file, apply optional dictionary overrides, and return a frozen Config.
    
    Args:
        path: Path to the YAML configuration file.
        overrides: Optional dictionary of overrides to overlay.
        
    Returns:
        Config: Immutable configuration object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    if overrides:
        raw_dict = _deep_update(raw_dict, overrides)

    model_data = raw_dict.get("model", {})
    activity_data = raw_dict.get("activity", {})
    resolution_data = raw_dict.get("resolution", {})
    external_data = raw_dict.get("external", {})
    persistence_data = raw_dict.get("persistence", {})
    data_data = raw_dict.get("data", {})
    training_data = raw_dict.get("training", {})
    seed = raw_dict.get("seed", 42)

    # Convert mixed_families list→tuple if loaded from YAML (YAML gives lists)
    if "mixed_families" in data_data and isinstance(data_data["mixed_families"], list):
        data_data["mixed_families"] = tuple(data_data["mixed_families"])

    return Config(
        model=ModelConfig(**model_data),
        activity=ActivityConfig(**activity_data),
        resolution=ResolutionConfig(**resolution_data),
        external=ExternalConfig(**external_data),
        persistence=PersistenceConfig(**persistence_data),
        data=DataConfig(**data_data),
        training=TrainingConfig(**training_data),
        seed=seed,
    )


def set_seed(seed: int) -> None:
    """Set random seed across standard library, NumPy, and PyTorch for deterministic replay."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
