# LIMINAL: Adaptive Internal–External Latent Workspace

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> *"Don't just use latent space to represent the answer. Make the latent space itself the place where the computation happens."*

**LIMINAL** is a research prototype investigating an alternative computing substrate for machine reasoning. Instead of forcing intermediate reasoning through sequential natural-language tokens (as in standard chain-of-thought autoregression), LIMINAL models reasoning as the **iterative transformation of a relational latent computational workspace** $G_t = (V_t, E_t, A_t)$ that can dynamically scale, self-prune, and eventually interface with an explicit external workspace.

---

## The Core Concept

### 1. Latent Relational Workspace
Intermediate multi-step reasoning is framed as message passing over an evolving set of latent slots:
$$G_t = (V_t, E_t, A_t)$$
- **$V_t$ (Latent Slots)**: Vector states representing uncommitted entity, concept, or intermediate scratchpad representations.
- **$E_t$ (Relational Edges)**: Pairwise interactions and message passing between slots computed via edge transformations.
- **$A_t$ (Activity Gates)**: Soft dynamic gates $a_{i,t} \in [0, 1]$ that control the participation and capacity of each slot at step $t$.

### 2. Adaptive Capacity & Self-Pruning
Rather than allocating static computational hardware or routing across feedforward parameters (MoE), LIMINAL regulates the **active capacity of its internal working memory**. Under sparsity and entropy pressure, the network learns to dynamically silence unneeded slots for simple tasks while reserving dormant capacity for complex reasoning.

### 3. Permutation Equivariance & Invariance
Slot storage order carries no formal semantics. Intermediate transformations are permutation-equivariant, and final readout is order-independent through activity-weighted pooling before decoding.

---

## Empirical Benchmarks & Results (Phase 1 & 2A)

Evaluated on controlled multi-step affordability composition tasks ("John has £50, Mary gives £20, shoe costs £70. Can John buy it?"):

| Model Variant | Parameters | Test Accuracy | Active Slots ($A_t > 0.5$) | Effective Capacity | Key Takeaway |
|---|:---:|:---:|:---:|:---:|---|
| **Vector Baseline** | 498,530 | 99.60% | N/A (Monolithic) | 100% | High accuracy, but requires a large 512-d recurrent vector. |
| **Static Graph** | 19,266 | 99.80% | 8 / 8 (Static) | 100% | **26× parameter reduction** with superior accuracy; proves relational inductive bias. |
| **Adaptive Graph** | **19,554** | **99.40%** | **1.0 / 8 (avg)** | **27.4% (2.19 / 8)** | **Self-pruned 73% of its capacity** while maintaining near-perfect accuracy. |

### Emergent Slot Specialization (Phase 2A Analysis)
Under $L_{total} = L_{task} + \lambda_{sparse} L_{sparse} + \lambda_{ent} L_{ent}$, the 8-slot adaptive model self-organized into:
- **Primary anchor (Slot 4)**: Mean activity $0.874 \pm 0.211$ (active across all reasoning steps).
- **Conditional buffers (Slots 1, 2, 3)**: Mean activity $0.41 - 0.46$ (recruited dynamically when needed).
- **Dormant / dead slots (Slots 0, 5, 6, 7)**: Mean activity $\approx 0.001$ (safely suppressed).

---

## Project Roadmap

- [x] **Phase 1**: Deterministic recurrent vector baseline vs. fixed-slot relational graph.
- [x] **Phase 2A**: Adaptive soft activity gates & self-pruning capacity dynamics.
- [ ] **Phase 2B**: Dynamic slot recruitment under high cognitive load (distractor facts & multi-hop chains).
- [ ] **Phase 3**: Adaptive resolution critic ($R_t \ge \tau$) for dynamic test-time halting.
- [ ] **Phase 4**: Latent state persistence across sequential conversational turns.
- [ ] **Phase 5–6**: Structured external workspace ($W_t$), bidirectional co-refinement, and learned externalization policies.
- [ ] **Phase 7**: Continuous-time latent dynamics (Neural ODEs / Liquid Time-Constant networks).

---

## Repository Structure

```text
liminal/
├── configs/
│   ├── base.yaml                   # Base configuration
│   ├── activity_graph.yaml         # Phase 2A adaptive gating configuration
│   └── baselines/                  # Vector & static graph baseline configs
├── src/
│   ├── data/                       # Synthetic task generators & PyTorch datasets
│   ├── models/
│   │   ├── activity.py             # Soft activity gates & regularizers
│   │   ├── encoder.py              # Fact & entity encoders
│   │   ├── latent_workspace.py     # Graph latent workspace & message passing
│   │   ├── message_passing.py      # Pairwise edge & node update blocks
│   │   ├── readout.py              # Permutation-invariant readout pooling
│   │   └── decoder.py              # Task prediction decoders
│   ├── training/                   # Loss functions, regularizers, Trainer loop
│   ├── evaluation/                 # Activity analysis & diagnostic plotting
│   └── utils/                      # Device management, checkpointing, configs
├── scripts/
│   ├── train.py                    # Training entry point
│   └── analyze_activity.py         # Slot trajectory and activity visualizer
├── tests/                          # Scaffolding and activity gate test suites
└── results/                        # Saved checkpoints, metrics, and heatmaps
```

---

## Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/isaac-hash/LIMINAL.git
cd LIMINAL

# Install package in editable mode
pip install -e ".[dev]"
```

### 2. Run Test Suite

```bash
python -m pytest tests/ -v
```

### 3. Train Models

```bash
# Train the Adaptive Graph Model (Phase 2A)
python scripts/train.py --config configs/activity_graph.yaml

# Train the Vector Baseline
python scripts/train.py --config configs/baselines/vector.yaml
```

### 4. Analyze Activity Gates & Trajectories

```bash
python scripts/analyze_activity.py --config configs/activity_graph.yaml
```
Outputs heatmaps, per-class activations, and step-by-step trajectory plots to `results/activity_graph/`.

---

## Design Principles

- **CPU-First Research**: Fully trainable and testable on commodity hardware (e.g. AMD Ryzen 3 laptops) without mandatory cloud GPU dependencies.
- **Pure PyTorch Implementation**: Minimal dependencies; message passing implemented directly via clean tensor operations without heavy external graph library overhead for $N \le 32$ slots.
- **Falsifiable Science**: Every architectural hypothesis (structure, gating, externalization) is paired with a strictly controlled baseline and causal ablation test.
