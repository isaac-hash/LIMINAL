# Adaptive Internal–External Latent Workspace — Implementation Plan

> **Hybrid Laptop + Colab research prototype**: a persistent, adaptive latent graph workspace coupled to an explicit structured external workspace, tested on controlled synthetic reasoning tasks.

## Development Workflow

```
              GitHub
                 │
        ┌────────┴────────┐
        │                 │
      Laptop           Colab Free
        │                 │
     coding              GPU
     testing           training
     debugging         experiments
     inspection        ablation sweeps
        │                 │
        └────────┬────────┘
                 │
             Drive / results
```

- **Laptop** = engineering workstation (develop, debug, inspect, tiny experiments)
- **Colab Free** = experimental compute (training, GPU acceleration, larger runs)
- **GitHub** = source of truth for code
- **Google Drive** = datasets, checkpoints, results persistence

> [!IMPORTANT]
> The code is **device-agnostic** — the same codebase runs on laptop CPU, Colab CPU, or Colab CUDA with no architectural changes.

## Hardware

### Laptop (Tier 1 — always local)

| Resource | Available |
|----------|-----------|
| CPU | AMD Ryzen 3 3250U |
| RAM | 12 GB |
| GPU | Integrated Radeon (**not used**) |
| Storage | ~22 GB free (target ≥40 GB) |

### Colab Free (Tier 2 — training & experiments)

| Resource | Typical |
|----------|---------|
| GPU | NVIDIA T4 *when available* (not guaranteed) |
| GPU Memory | ~15 GB class |
| Runtime | Up to 12 hours (variable, may terminate) |
| Storage | Ephemeral VM + Google Drive mount |

> [!WARNING]
> Colab Free does not guarantee GPU type, availability, or session length. All training code must support **checkpointed experiment batches** — never "train for 9 hours and hope." Store artifacts on Drive/GitHub, not the ephemeral VM.

## Experiment Scales

| Scale | Params | Where | Phase |
|-------|--------|-------|-------|
| **1 — Sanity** | 8 slots × 32 dim, 1 layer, 10k examples | Runs anywhere | Weeks 1–2 |
| **2 — Real MVP** | 16–32 slots × 64–128 dim, 2–3 layers, 25k–100k examples | Colab GPU | Weeks 3–10 |
| **3 — Serious research** | 32–64 slots × 128–256 dim, multiple seeds, ODE/probabilistic | Colab GPU (less predictable) | Weeks 11–12+ |
| **4 — Language integration** | NL encoder → workspace → decoder | Paid GPU / post-MVP | Future |

## Model Parameters

| Parameter | Scale 1 (local) | Scale 2 (Colab) |
|-----------|-----------------|-----------------|
| Latent slots | **8** | **16–32** |
| Latent dimension | **32** | **64–128** |
| Message-passing layers | **1** | **2–3** |
| Reasoning steps | **2–8** | **8–16** |
| Max reasoning steps | **16** | **16** |
| Edge representation | scalar | low-dimensional |
| External records | ≤16 | ≤32 |
| Batch size | **8–16** | **32–64** |
| Dataset size | **10k** | **25k–100k** |
| Optimiser | Adam | Adam |
| Precision | FP32 | FP32 (FP16 optional) |

## Success Ladder

| Level | Claim | When |
|-------|-------|------|
| **0 — Infrastructure** | The system trains | End Week 2 |
| **1 — Latent evidence** | Dynamic graph beats recurrent-vector baseline on compositional tasks | End Week 4 |
| **2 — Workspace evidence** | Externalisation improves performance/efficiency | End Week 7 |
| **3 — Causal evidence** ★ | Manipulating external state systematically changes later latent computation | **End Week 8** |
| **4 — Strong architectural** | Benefits hold under parameter matching, compute matching, harder compositions, distractors, multiple seeds | End Week 12 |

---

## Proposed Changes

All code is new. The project starts from the [LIMINAL](file:///c:/Users/HP/Desktop/LIMINAL) directory (currently contains only spec documents). Changes are grouped into 8 phases.

---

### Phase 1 — Minimal Computational Substrate (Weeks 1–2)

> Laptop + Colab. Environment, data pipeline, recurrent-vector baseline, fixed-slot graph, first comparison.

#### [NEW] Project scaffolding & environment

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies: **PyTorch, NumPy, pytest, matplotlib, pyyaml** |
| `README.md` | Project overview, setup, architecture diagram |
| `.gitignore` | Python + PyTorch + Colab ignores |
| `configs/base.yaml` | Default hyperparams (Scale 1 values) |
| `configs/baselines/vector_baseline.yaml` | Single recurrent vector override |
| `configs/baselines/static_graph.yaml` | Fixed edges, no activity gates |
| `notebooks/colab_setup.ipynb` | Colab bootstrap: clone repo, install deps, detect hardware, mount Drive |

> [!NOTE]
> **Minimal stack to start**: Python, PyTorch, NumPy, pytest, Matplotlib, Git. No PyG, no W&B, no Hydra, no torchdiffeq. Added only when the experiment requires them.

#### [NEW] `src/utils/device.py` — Device management

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- Every module uses this — same codebase on laptop CPU or Colab CUDA.
- Runtime hardware detection at start of every run:
  ```python
  print("CUDA:", torch.cuda.is_available())
  if torch.cuda.is_available():
      print("GPU:", torch.cuda.get_device_name(0))
      print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
  ```

#### [NEW] `src/data/` — Synthetic data generation

| File | Purpose |
|------|---------|
| `src/data/__init__.py` | Package init |
| `src/data/arithmetic.py` | Parameterised arithmetic-composition generator. Start with `50 + 20 = 70`, then structured multi-entity affordability tasks. JSONL output with `input_facts`, `ground_truth`, `proof_trace`, `task_family`, `difficulty`, `seed`, `split`. |
| `src/data/dataset.py` | PyTorch `Dataset` / `DataLoader` wrappers. Archive-based loading for Colab (avoid reading many small files from Drive). |

#### [NEW] `src/models/encoder.py` — Input encoder

- Fact-level MLP encoder → scatter/sum into N fixed slots.
- Output: `V_0 ∈ ℝ^{N×d}`, `E_0 ∈ ℝ^{N×N}` (scalar edges), `A_0 ∈ ℝ^N` (all ones initially).
- All tensors created on `device`.

#### [NEW] `src/models/latent_workspace.py` — Core workspace container

- Holds `V_t`, `E_t`, `A_t`.
- `step(X, R_t) → G_{t+1}`: orchestrates message passing → state update.
- Week 1: single recurrent vector baseline mode (1 slot, GRU update, no graph).
- Week 2: multi-slot with relational message passing.

#### [NEW] `src/models/message_passing.py` — Custom equivariant update

> **No PyTorch Geometric.** For 8–32 slots, plain tensor ops are simpler and more inspectable.

- `m_i = Σ_j φ_θ(v_i, v_j, e_ij)` — ordinary tensor operations.
- `φ_θ`: small MLP on `[v_i ∥ v_j ∥ e_ij]`.
- `Update_θ`: GRU-cell gated update mixing `v_i` and `m_i`.
- Permutation-equivariant by construction (symmetric sum aggregation).

#### [NEW] `src/models/readout.py` — Invariant aggregation

- `G* = Σ_i a_i · v_i` (activity-weighted mean pooling).
- Permutation-invariant.

#### [NEW] `src/models/decoder.py` — Answer head

- MLP: `G* → logits` for classification / `G* → ℝ` for regression.

#### [NEW] `src/training/trainer.py` — Training loop

- Standard PyTorch loop with gradient clipping.
- Logs to **CSV/JSON** (no W&B dependency yet — add later if needed).
- **Checkpoint policy**: `best.pt` + `last.pt` only. Config and metrics alongside.
- **Colab resilience**: periodic checkpointing to Drive, resume-from-checkpoint support.
- Config-driven: reads YAML, constructs model/optimizer/scheduler.

#### [NEW] `src/training/losses.py` — Loss functions

- `L_answer`: cross-entropy or MSE.
- Placeholder slots for `L_resolution`, `L_activity_reg`, `L_write`.

#### [NEW] `src/evaluation/evaluator.py` — Evaluation harness

- Exact accuracy, per-task-family breakdown, compositional generalisation metric.

#### [NEW] `src/utils/config.py` — Config loading

- YAML → frozen dataclasses, config inheritance, seed management.

#### [NEW] `src/utils/checkpoint.py` — Checkpoint management

- `save(model, optimizer, epoch, metrics, path)` / `load(path) → state`.
- Colab-aware: auto-copy to/from Drive path if mounted.
- Resume support: `trainer.resume_from(checkpoint_path)`.

#### [NEW] `tests/`

| File | Purpose |
|------|---------|
| `tests/test_data.py` | Valid records, correct ground truths, reproducible seeds. |
| `tests/test_message_passing.py` | Permutation equivariance: `f(Pv, Pe) = P·f(v, e)`. |
| `tests/test_readout.py` | Permutation invariance: `readout(Pv) = readout(v)`. |
| `tests/test_training.py` | Smoke test: 1 epoch on 100 samples, loss decreases. |

#### [NEW] `scripts/train.py` — Entry point

- Parses config, instantiates model/data/trainer, runs training.
- Prints hardware detection at start.

#### [NEW] `notebooks/colab_setup.ipynb` — Colab bootstrap

- Clone repo from GitHub, install dependencies.
- Detect and report hardware (CPU/GPU type/VRAM).
- Mount Google Drive for checkpoint persistence.
- Run a quick smoke test to verify the environment.

**Exit criterion (Level 0):** Vector baseline trains. Fixed-slot graph trains. Both solve 2-op arithmetic at >90%. Permutation tests pass. Runs on both laptop CPU and Colab.

---

### Phase 2 — Adaptive Latent Activity (Week 3)

> Primarily Colab. Activity gates for slot recruitment/suppression.

#### [NEW] `src/models/activity.py` — Activity controller

- Per-slot gate: `a_i = σ(f_θ(v_i, pool(V), X))`.
- Soft gating: `v_i' = a_i · Update(v_i, m_i)`.

#### [MODIFY] `src/models/latent_workspace.py`

- Integrate `ActivityController` into `step()`.
- Log per-slot activity values at each iteration.

#### [MODIFY] `src/training/losses.py`

- `L_activity_reg = λ_act · mean(A_t)` — sparsity.
- Entropy bonus — prevent all-zero collapse.

#### [NEW] `src/evaluation/activity_analysis.py`

- Activity heatmap, active-slot count vs iteration, activity entropy.

#### [NEW] `configs/experiments/activity_ablation.yaml`

#### [NEW] `tests/test_activity.py`

**Exit criterion:** Non-uniform activation. Measurable difference in all-active vs gated ablation.

---

### Phase 3 — Resolution & Adaptive Halting (Week 4)

> Primarily Colab. Learned stopping criterion.

#### [NEW] `src/models/resolution.py` — Resolution critic

- Composite signal: confidence + stability + constraints → scalar `r_t`.
- 2-layer MLP. Stop when `r_t ≥ τ`, hard cap at `T_max=16`.

#### [MODIFY] `src/models/latent_workspace.py`

- Adaptive iteration loop with ACT-style ponder cost.

#### [MODIFY] `src/training/losses.py`

- `L_resolution` (BCE) + `L_ponder`.

#### [NEW] `configs/experiments/halting_ablation.yaml`

#### [NEW] `tests/test_resolution.py`

**Exit criterion (supports Level 1):** Early halting on solved cases. Average iterations < `T_max` without accuracy loss.

---

### Phase 4 — Persistence (Week 5)

> Colab. Carry latent state across related inputs.

#### [NEW] `src/models/persistence.py`

- Gated blend: `G'_0 = gate · G_t + (1 - gate) · Encode(X_new)`.

#### [MODIFY] `src/data/arithmetic.py`

- Multi-turn task generator (2–3 related facts across turns).

#### [MODIFY] `src/models/latent_workspace.py`

- `persistent_mode` flag, prior-state blending.

#### [NEW] `configs/experiments/persistence_ablation.yaml`

#### [NEW] `tests/test_persistence.py`

**Exit criterion:** Persistent model shows measurable benefit on related-input tasks vs reset.

---

### Phase 5 — External Workspace v1 (Weeks 6–7)

> Colab. Structured scratchpad, deterministic read/write.

#### [NEW] `src/models/external_workspace.py`

- Fixed-size buffer of M≤32 typed record slots.
- Types: `ENTITY`, `ATTRIBUTE`, `RELATION`, `OPERATION`, `CONSTRAINT`, `STATUS`.
- Serialisable to/from JSON. **Not** natural language. **Not** a database.

#### [NEW] `src/models/externaliser.py` — Write controller

- Week 6: deterministic top-K writes. Week 9: learned gated policy.

#### [NEW] `src/models/reader.py` — Read controller

- Attention-based read: query = pooled latent, keys/values = external records.

#### [MODIFY] `src/models/latent_workspace.py`

- Integrate external workspace into iteration loop:
  1. Latent update: `G_{t+1} = F(G_t, X, R_t)`
  2. External write: `W_{t+1} = H(W_t, G_{t+1})`
  3. External read: `R_{t+1} = Read(G_{t+1}, W_{t+1})`

#### [NEW] `src/models/relationships.py` — Edge adaptation

- `e_ij' = MLP(v_i, v_j, e_ij)`.

#### [NEW] `src/evaluation/workspace_analysis.py`

#### [NEW] `tests/test_external_workspace.py`

- Round-trip, serialisation, no label leakage, corruption isolation.

#### [NEW] `configs/experiments/external_ablation.yaml`

**Exit criterion (supports Level 2):** Inspectable workspace at each iteration. Trains end-to-end with coupled loop.

---

### Phase 6 — Bidirectional Coupling & Causal Intervention (Week 8)

> ★ **Central MVP milestone.** Colab.

Full loop:
```
INPUT → LATENT → reason → EXTERNALISE → EXTERNAL → READ → LATENT → reason → RESOLVE → ANSWER
```

#### [NEW] `src/evaluation/causal_intervention.py`

**Protocol:**
1. Run to step `t₁` → snapshot `G_{t₁}` and `W₁`.
2. Apply targeted corruption to `W₁` → `W̃₁`.
3. Continue from **identical** `G_{t₁}` with `W̃₁`.
4. Measure divergence.

**Corruption types:**

| Type | Example | Purpose |
|------|---------|---------|
| `swap_value` | `money(John)=70 → 60` | Relevant perturbation |
| `delete_record` | Remove relevant record | Information loss |
| `randomise_record` | Random vector | Content destruction |
| `irrelevant_swap` | Change irrelevant record | **Negative control** |

**Metrics:** L2 trajectory distance, accuracy delta, iteration delta, resolution-time delta.

#### [MODIFY] `src/models/latent_workspace.py`

- `snapshot()` / `restore()` methods. Full trajectory logging.

#### [NEW] `scripts/run_causal_experiment.py`

#### [NEW] `tests/test_causal_intervention.py`

**Exit criterion (Level 3 ★):** Relevant corruption → systematic downstream changes. Irrelevant corruption → no significant effect.

---

### Phase 7 — Learned Externalisation (Weeks 9–10)

> Colab. Replace deterministic writes with learned gating.

#### [MODIFY] `src/models/externaliser.py`

- Learned write gate with Gumbel-softmax. Write-cost regularisation.

#### [NEW] `src/data/externalisation_sensitive.py`

- Tasks with many intermediate dependencies.

#### [MODIFY] `src/data/arithmetic.py`

- Increased difficulty (up to 8-op chains, 6+ entities).

#### [NEW] `configs/experiments/externalisation_comparison.yaml`

- Three-way: never / always / learned.

#### [NEW] `src/evaluation/selectivity_analysis.py`

**Exit criterion:** Learned beats never- and always-externalise. Writes are selective and task-dependent.

---

### Phase 8 — Full Evaluation & Evidence Package (Weeks 11–12)

> Colab for training runs. Laptop for analysis and visualisation.

#### [NEW] `scripts/run_baselines.py`

| ID | Baseline | Config |
|----|----------|--------|
| A | Single recurrent vector | `configs/baselines/vector_baseline.yaml` |
| B | Static latent graph | `configs/baselines/static_graph.yaml` |
| C | Dynamic graph (no external) | `configs/baselines/dynamic_graph.yaml` |
| D | Dynamic graph + external (full) | `configs/base.yaml` |

3–5 seeds per baseline (Colab GPU time permitting).

#### [NEW] `scripts/run_ablations.py`

| Ablation | Question |
|----------|----------|
| D − activity | Does adaptive capacity matter? |
| D − external write | Can reasoning work without externalisation? |
| D − external read | Is written information reused? |
| D − resolution | Does adaptive stopping matter? |
| D − persistence | Does state across inputs matter? |

#### [NEW] `src/evaluation/statistical_analysis.py`

- Mean ± std across seeds, paired comparisons, bootstrap CIs.

#### [NEW] `src/evaluation/visualisation.py`

- Latent trajectory PCA, activity heatmaps, write/read timelines, resolution plots, scaling curves.

#### [NEW] `results/`, `docs/`

**Exit criterion (Level 4):** ≥3 baselines under matched budgets. Ablations isolate contributions. Causal intervention completed. Reproducible from committed code + configs.

---

## Colab Resilience Strategy

> [!WARNING]
> Colab Free runtimes can terminate at any time. The following patterns protect against data loss.

1. **Checkpointed batches**: Save after every epoch (or every N steps). Never rely on a single long session.
   ```
   experiment_001 → checkpoint → experiment_002 → checkpoint → ...
   ```
2. **Drive-backed artifacts**: Auto-copy `best.pt`, `last.pt`, `config.yaml`, `metrics.json` to mounted Drive.
3. **Resume support**: Every training script supports `--resume path/to/checkpoint`.
4. **Archive datasets**: Load datasets as single archive files from Drive, extract to VM local storage. Avoid reading many small files from mounted Drive.
5. **Hardware detection**: Print GPU type/VRAM at start of every run. Adapt batch size if needed.
6. **Storage discipline**: Keep only `best.pt` + `last.pt` per run. Delete redundant checkpoints.

---

## Verification Plan

### Automated Tests

```bash
# All tests — run after every phase
pytest tests/ -v --tb=short

# Permutation equivariance/invariance
pytest tests/test_message_passing.py tests/test_readout.py -v

# Training smoke test (should complete <60s on CPU)
pytest tests/test_training.py -v
```

### Phase Exit Checks

| Check | When | Where | Method |
|-------|------|-------|--------|
| Vector baseline >90% on 2-op arithmetic | End Week 1 | Laptop/Colab | `scripts/train.py` |
| Graph ≥ vector | End Week 2 | Laptop/Colab | Compare metrics |
| Non-uniform activity patterns | End Week 3 | Colab | Activity plots |
| Early halting on easy tasks | End Week 4 | Colab | Iteration histogram |
| Persistent state helps multi-turn | End Week 5 | Colab | Accuracy comparison |
| Inspectable external workspace | End Week 7 | Laptop | JSON serialisation |
| **Corruption → downstream changes** | **End Week 8** ★ | **Colab** | Causal intervention |
| Selective learned writes | End Week 10 | Colab | Selectivity analysis |
| All baselines reproducible | End Week 12 | Colab | Re-run from configs |

### Falsification Checks (Ongoing)

- Full model ≤ vector baseline after matching parameters/compute
- Corruption has no downstream effect on latent trajectory
- Activity gates collapse to all-on or all-off
- Resolution critic predicts the answer instead of evaluating state sufficiency
- Graph collapses to a single effective vector

---

## Repository Structure

```
LIMINAL/
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── base.yaml
│   ├── baselines/
│   │   ├── vector_baseline.yaml
│   │   ├── static_graph.yaml
│   │   └── dynamic_graph.yaml
│   └── experiments/
│       ├── activity_ablation.yaml
│       ├── halting_ablation.yaml
│       ├── persistence_ablation.yaml
│       ├── external_ablation.yaml
│       └── externalisation_comparison.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── arithmetic.py
│   │   ├── dataset.py
│   │   └── externalisation_sensitive.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── latent_workspace.py
│   │   ├── message_passing.py
│   │   ├── activity.py
│   │   ├── relationships.py
│   │   ├── external_workspace.py
│   │   ├── externaliser.py
│   │   ├── reader.py
│   │   ├── resolution.py
│   │   ├── readout.py
│   │   ├── decoder.py
│   │   └── persistence.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── activity_analysis.py
│   │   ├── workspace_analysis.py
│   │   ├── causal_intervention.py
│   │   ├── selectivity_analysis.py
│   │   ├── statistical_analysis.py
│   │   └── visualisation.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── device.py
│       └── checkpoint.py
├── tests/
│   ├── test_data.py
│   ├── test_message_passing.py
│   ├── test_readout.py
│   ├── test_training.py
│   ├── test_activity.py
│   ├── test_resolution.py
│   ├── test_persistence.py
│   ├── test_external_workspace.py
│   └── test_causal_intervention.py
├── scripts/
│   ├── train.py
│   ├── run_causal_experiment.py
│   ├── run_baselines.py
│   └── run_ablations.py
├── notebooks/
│   └── colab_setup.ipynb
├── results/
├── notebooks/
└── docs/
    ├── architecture.md
    ├── experiments.md
    └── decisions.md
```
