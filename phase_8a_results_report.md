# LIMINAL — Phase 8A Results Report: Two-Track Architecture Benchmark & Memory Audit

**Project**: Adaptive Internal–External Latent Workspace (LIMINAL)  
**Date**: September 2026  
**Hardware Evaluated**: CPU (PyTorch 2.11, Threaded Inference)  
**Evaluation Splits**: 500 sequences × 5 turns (2,500 decisions per track, seed=999)  
**Artifact Archive**: `/content/drive/MyDrive/LIMINAL_results/phase8a/`  

---

## 1. Executive Summary

Phase 8A consolidates the core findings across all phases of the LIMINAL research programme. Rather than forcing models trained on distinct curricula into a single misleading aggregate, we adopt a **Two-Track Evaluation Paradigm** coupled with an empirical audit of the learned externalisation gate:

1. **Track 1: Canonical Sequential Reasoning (`affordability_sequence`)**
   Evaluates how architectural inductive biases govern multi-turn memory retention over a 5-turn horizon across **Vector Baseline (Model A)**, **Static Graph (Model B)**, and **Persistent Graph (Model C)**.
2. **Track 2: Learned Externalisation & Write Controller Audit (`ext_sensitive`)**
   Evaluates the full internal–external workspace hierarchy (**Model D — Full LIMINAL**) to test whether the learned Gumbel write controller adaptively gates external memory.

```
========================================================================================
  Phase 8A Core Findings at a Glance
========================================================================================
  1. Gated Latent Persistence (Model C) is Structurally Necessary:
     • Overall Accuracy : 89.72% (Highest on Track 1)
     • Turn 5 Accuracy  : 87.00% (+17.0% over Vector Baseline, +5.2% over Static Graph)
     • Test Loss        : 0.9783 (Lowest by a wide margin)
     • Parameters       : 23,557 (95.3% smaller than the Vector Baseline)
     • Key Finding      : Persistence is strictly necessary when past facts are not
       re-supplied in the current prompt.

  2. Dense Vector Models Suffer Catastrophic Horizon Collapse:
     • Vector Baseline collapses from 98.2% at Turn 2 to 70.0% at Turn 5 (-28.2% drop).
     • Despite utilizing ~21× more parameters (498.5K), vector representations suffer
       from representational smearing across multi-turn sequences.

  3. Diagnostic Audit Confirms Optimization Trap (Dead Gate) in Model D:
     • Pre-activation write logits saturated between -3.1 and -24.6 (mean -9.2).
     • 0.000% of latent slots ever produced a positive logit; type_head weights
       remained untouched from random initialization.
     • Root Cause       : -1.0 bias init + immediate L1 penalty (λ_write = 0.01) starved
       the gate of gradients before external utility was discovered.
========================================================================================
```

---

## 2. Track 1: Canonical Sequential Reasoning Benchmark

### Experimental Protocol
* **Task Family**: `affordability_sequence` (5 turns per sequence).
* **Dataset**: 500 held-out test sequences (2,500 total turn decisions, seed=999).
* **Hardware**: CPU inference (batch size = 32).

### Quantitative Results Table

| Model | Architecture | Params | Overall Acc | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | Test Loss | Latency (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Vector Baseline** | Dense MLP Embedding | 498,530 | **85.20%** | 88.8% | 98.2% | 90.6% | 78.4% | 70.0% | 2.3826 | 1.28 ms/seq |
| **Model B: Static Graph** | Non-recurrent GNN (resets) | 19,266 | **86.68%** | 73.0% | 100.0% | 93.4% | 85.2% | 81.8% | 1.5344 | 2.26 ms/seq |
| **Model C: Persistent Graph** | Recurrent GNN + Gated Memory | 23,557 | **89.72%** | **99.0%** | 86.6% | 86.6% | **89.4%** | **87.0%** | **0.9783** | 2.57 ms/seq |
| *Model D: Full LIMINAL (OOD)* | Full Hierarchy (Learned Gate) | 34,158 | *38.64%* | *34.8%* | *37.0%* | *39.2%* | *39.2%* | *43.0%* | *12.6041* | 4.90 ms/seq |

---

### In-Depth Turn Decay Analysis (Track 1)

```
Turn Retention Trajectory:
Turn:            T1        T2        T3        T4        T5       Retention (Peak → T5)
----------------------------------------------------------------------------------------
Model A (Vector): 88.8%    98.2%     90.6%     78.4%     70.0%     -28.2%  (Catastrophic)
Model B (Static): 73.0%   100.0%     93.4%     85.2%     81.8%     -18.2%  (Moderate)
Model C (Persist):99.0%    86.6%     86.6%     89.4%     87.0%     -12.0%  (Robust)
```

1. **Why the Vector Baseline Collapses (-28.2% drop)**:
   * At Turn 2, the vector model excels (98.2%) because the context is short.
   * However, by Turn 5, it drops to 70.0%. Because all entities, prices, and arithmetic operations are mapped into a single flat dense vector, updates across multiple turns cause continuous "vector smearing". The network loses track of which specific entity holds which accumulated balance.
2. **Precision on Model C's Advantage (+17% over Vector, +5.2% over Static)**:
   * Model B achieves 81.8% at Turn 5 because it resets node states between turns ($h^{(t)} = 0$). When sequence state is **not re-supplied in subsequent turns**, memoryless models are structurally denied past context.
   * Model C maintains **87.0% accuracy at Turn 5** with an overall loss of **0.9783**. This demonstrates that gated persistence ($\mathbf{h}^{(t)} = \sigma(\mathbf{z}) \odot \mathbf{h}^{(t-1)} + (1 - \sigma(\mathbf{z})) \odot \mathbf{\tilde{h}}^{(t)}$) is **structurally necessary to prevent information loss** in un-prompted sequential reasoning.

---

## 3. Track 2: Learned Externalisation & Optimization Audit

### Experimental Protocol
* **Task Family**: `ext_sensitive` (5-turn multi-hop balance tracking requiring historical recall).
* **Dataset**: 500 held-out evaluation sequences (1,635 reasoning steps, seed=42).

### The Diagnostic Logit Audit (Testing the "Dead Gate" Hypothesis)
To resolve whether 0.00 writes reflected genuine "emergent parsimony" or a dead gate, we executed a post-hoc audit hooking into `write_controller.gate_net` on frozen checkpoint weights:

```
========================================================================================
  Post-Hoc Write Gate Logit Distribution (Model D Frozen Weights)
========================================================================================
  Track 1 (affordability_sequence):
    • Pre-activation Logit Range : [-20.9829, -3.1264]
    • Mean Pre-activation Logit  : -8.8300 ± 1.8732
    • Slots Producing Logit > 0  : 0.000%

  Track 2 (ext_sensitive):
    • Pre-activation Logit Range : [-24.6090, -3.8216]
    • Mean Pre-activation Logit  : -9.5707 ± 2.3408
    • Slots Producing Logit > 0  : 0.000%

  Weights Audit:
    • gate_net.2.bias            : -1.1724  (initialized at -1.0)
    • gate_net.2.weight mean     : -0.2698  (almost all output weights negative)
    • type_head.bias             : 0.0000   (untouched from initialization)
    • type_head.weight           : std=0.01 (untouched normal distribution, 0 updates)
========================================================================================
```

### Empirical Verdict: Optimization Pathology (Not Parsimony)
The audit definitively proves that the write gate was **structurally extinguished**:
1. **The Optimization Trap**: The gate was initialized with a negative bias (`-1.0`). Simultaneously, an $L_1$ write-sparsity penalty ($\lambda_{\text{write}} = 0.01$) penalized any non-zero write from epoch 1. Because the internal persistent graph (Phase 4) was already capable of achieving ~95% accuracy without external memory, the network minimized loss by driving gate logits deep into negative saturation (mean $-9.2$).
2. **Zero External Gradient**: The discrete `type_head` never received a single backpropagation update throughout 50 epochs.
3. **Out-of-Distribution Failure**: When evaluated on Track 1 where accuracy collapsed to 38.64%, the gate still wrote 0.000 slots/step (maximum logit $-3.12$). A truly adaptive gate would have explored external writes under failure; instead, it was locked shut.

### Updated Quantitative Results Table (With Corrected Metric)

| Model Architecture | Write Policy | Accuracy | Mean Writes / Step | Gate Entropy | Selectivity Score | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Heuristic Baseline** | Top-$K$ ($K=3$) | 95.80% | 3.00 slots/step | — | 0.625 | Forced Writing |
| **Model D: Full LIMINAL** | Learned Gate | **97.16%** | **0.00 slots/step** | **0.0000** | **0.000** *(Corrected)* | **COLLAPSED CLOSED** |

*(Correction: The previously reported selectivity score of 1.000 was a mathematical artifact of `1.0 - fraction` when fraction=0. Under our revised metric, zero-write gate collapse is assigned a selectivity score of 0.000 and fails the exit criterion).*

---

## 4. Code & Metric Refactorings Completed

1. **Fixed `selectivity_score` Bug** in `src/evaluation/selectivity_analysis.py`:
   * If a gate collapses closed (`mean_slots < 0.05`) or unconstrained open (`fraction > 0.95`), `selectivity_score` now evaluates to `0.000` with `gate_status = "collapsed_closed"`.
2. **Fixed `exit_criterion_met`** in `scripts/analyze_selectivity.py`:
   * The exit criterion now strictly enforces an activity floor (`mean_slots >= 0.05`) and an entropy floor (`gate_entropy >= 0.005`). A dead gate can no longer pass evaluation.
3. **Wired Up On-The-Fly Baseline Evaluation**:
   * If no baseline checkpoint is provided, `analyze_selectivity.py` now executes an on-the-fly top-K forced-write ablation using identical model weights, ensuring `baseline_evaluation` is never `null`.
4. **Prepared Stage 1 Remediation in Code**:
   * Added `write_gate_bias_init: float = 0.0` (neutral initialization).
   * Added `write_sparsity_warmup_epochs: int = 10` to `ExternalConfig` and `SequentialTrainer` to hold $\lambda_{\text{write}} = 0$ during the initial discovery phase.

---

## 5. Roadmap for Future Work (Staged Resolution)

Rather than jumping to a costly rebuild, future investigations should follow this staged progression:

```
[Current State: Dead Gate Diagnosed & Metrics Fixed]
                    │
                    ▼
   Stage 1: The Cheap Test (Same Task, Zero Added Compute Load)
   • Train with write_gate_bias_init = 0.0 (neutral init)
   • Train with write_sparsity_warmup_epochs = 10 (delayed penalty)
   • Check: Does the gate show non-zero entropy and non-zero writes?
         │                                       │
         ▼ (Yes: Gate works!)                    ▼ (No: Still saturates to 0)
   Working Adaptive Gate Proven           Confirms Task Capacity Sufficiency
                                                 │
                                                 ▼
                                  Stage 2: Capacity Starvation (Path B)
                                  • Restrict internal slots to N = 3
                                  • Increase entities to 6–8 across 8–10 turns
                                  • Forces externalisation under mathematical necessity
```

---

## 6. Generated Visual Artifacts

All plots and data files remain available at `/content/drive/MyDrive/LIMINAL_results/phase8a/`:
* `phase8a_metrics.json` — Comprehensive per-model, per-turn numerical metrics.
* `phase8a_metrics.csv` — Tabular CSV format for spreadsheet analysis.
* `phase8a_comparison_table.md` — Markdown table formatted for documentation.
* `turn_accuracy_decay.png` — Per-turn accuracy retention curves (Turns 1–5).
* `accuracy_vs_compute.png` — Parameter count vs. accuracy vs. latency scatter plot.
