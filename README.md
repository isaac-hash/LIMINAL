# LIMINAL: Adaptive Internal–External Latent Workspace

An experimental research prototype investigating persistent, adaptive latent graph workspaces coupled with structured external workspaces for multi-step reasoning.

---

## Hardware Profile & Design Principles

- **Primary Development**: Local CPU-first workflow (laptop-friendly, lightweight footprint).
- **Compute Scaling**: Google Colab Free / Cloud GPU for larger batch runs without codebase divergence.
- **Minimal Dependencies**: Pure PyTorch tensor operations for message passing (no heavy PyG or external graph frameworks required for $N \le 32$ slots).
- **Reproducibility**: Parameterized data generation with deterministic seed guarantees.

---

## Directory Structure

```
liminal/
├── configs/                  # Experiment YAML configs
│   ├── base.yaml             # Single source of truth configuration
│   └── baselines/            # Baseline experiment definitions
├── src/
│   ├── data/                 # Synthetic task generator & PyTorch dataset
│   ├── models/               # Latent workspace, message passing, encoder/decoder
│   ├── training/             # Loss computations, Trainer, optimizers
│   ├── evaluation/           # Evaluation metrics and causal probes
│   └── utils/                # Device detection, config loading, checkpoints
├── tests/                    # Pytest unit and integration test suite
├── scripts/                  # Training and evaluation entry points
└── pyproject.toml            # Project definition & dependencies
```

---

## Setup & Quickstart

```bash
# Install editable package with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Inspect hardware environment
python -c "from src.utils.device import print_hardware_info; print_hardware_info()"
```

