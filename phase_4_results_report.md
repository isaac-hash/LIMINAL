# LIMINAL — Phase 4 Results Report (Part 1: Standalone Regime)

**Project**: Adaptive Internal–External Latent Workspace (LIMINAL)  
**Date**: September 2026  
**Hardware**: Google Colab · NVIDIA Tesla T4 (14.6 GB VRAM) · CUDA  
**Stack**: PyTorch 2.14 · NumPy · Matplotlib · PyYAML · pytest  
**Configs**: `configs/persistence_graph.yaml` vs `configs/persistence_reset.yaml`  

---

## Executive Summary (Lay Terms)

In Phase 4, we ask whether the model can **remember** across related questions. Instead of wiping its mind blank after each question, we added a **Persistence Gate** (a learnable valve) on each of its 8 mental slots that decides how much of its prior thinking to keep versus how much to replace with new input.

We tested two models head-to-head on 3-turn sequences:
1. **Model E (Persistent)**: State carries forward across turns via the gated blend.
2. **Model E-reset (Reset Baseline)**: Latent workspace is wiped clean to fresh encodings before every turn.

### What Happened in the Standalone Regime (Option A):
- **Both models solved the multi-turn problems with ~99.5% accuracy.**
- On Turn 3 (the longest chain with 3 operations), the persistent model achieved a slight edge (**99.60% vs 98.80%, $\Delta = +0.80\%$**).
- **The Persistence Gate remained effectively closed**: Mean gate activation across all 8 slots was **$0.00 - 0.03$**.

### Why the Gate Stayed Closed:
In this first experiment (Option A), every turn included the **complete history** of facts (Alice's starting balance, prior transactions, and the new transaction). Because the model had all the information right in front of it at every turn, the fresh encoder could easily solve each step from scratch. The neural network had **no gradient incentive to rely on memory**, so the persistence gate defaulted to near-zero.

This precisely confirms the hypothesis documented in our implementation plan: *if the model has all facts standalone, persistence is optional; to test the true causal power of memory, we must provide only new facts at each turn (Option B: Incremental Regime).*

---

## 1. Experimental Setup & Configurations

| Parameter | Persistent Model (Model E) | Reset Baseline (Model E-reset) |
| :--- | :---: | :---: |
| **Workspace Type** | Graph (8 slots × 32 dim) | Graph (8 slots × 32 dim) |
| **Spatial Pruning (Phase 2)** | ON (Adaptive Activity Gates) | ON (Adaptive Activity Gates) |
| **Temporal Halting (Phase 3)** | ON (ACT Halt Gates, $T_{\max}=16$) | ON (ACT Halt Gates, $T_{\max}=16$) |
| **State Persistence (Phase 4)** | **ON (PersistenceGate)** | **OFF (Fresh reset)** |
| **Detach Turns** | True | True |
| **Task Family** | `affordability_sequence` | `affordability_sequence` |
| **Turns per Sequence** | 3 turns | 3 turns |
| **Operations added per turn** | 1 operation | 1 operation |
| **Dataset Size** | 3,000 train (9,000 turns) · 500 val · 500 test | 3,000 train (9,000 turns) · 500 val · 500 test |
| **Training Schedule** | 50 epochs · Batch Size 8 · AdamW + Cosine LR | 50 epochs · Batch Size 8 · AdamW + Cosine LR |

---

## 2. Quantitative Results

### 2A — Final Test Set Evaluation

| Metric | Persistent Model (Model E) | Reset Baseline (Model E-reset) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **99.53%** | 99.33% | $+0.20\%$ |
| **Test Loss** | 0.0607 | 0.0473 | $+0.0134$ |
| **Turn 1 Accuracy** | 99.40% | 99.60% | $-0.20\%$ |
| **Turn 2 Accuracy** | 99.60% | 99.60% | $+0.00\%$ |
| **Turn 3 Accuracy** | **99.60%** | **98.80%** | **$+0.80\%$** |

*(Metrics computed from `best.pt` checkpoints on 500 held-out test sequences / 1,500 total turns).*

---

### 2B — Persistence Gate Activation Heatmap

Analysis of slot retention gates across the 500 test sequences ($0.0 = \text{re-encode fresh}$, $1.0 = \text{retain prior state}$):

| Turn | Slot 0 | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Slot 6 | Slot 7 | Mean Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Turn 2** | 0.01 | 0.03 | 0.02 | 0.01 | 0.00 | 0.01 | 0.01 | 0.01 | **0.014** |
| **Turn 3** | 0.01 | 0.03 | 0.02 | 0.02 | 0.01 | 0.00 | 0.01 | 0.01 | **0.014** |

Across all slots, gate values remained below $0.03$. The model consistently routed $>97\%$ of its representation through the fresh encoder rather than blending with $V_{\text{prior}}$.

---

## 3. Five-Way Project Evolution

| Metric | Model A (Vector) | Model B (Static Graph) | Model C (Activity Graph) | Model D (Resolution Graph) | **Model E (Persistent Graph)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test Accuracy** | 99.60% | 99.80% | 99.40% | 100.00% | **99.53%** |
| **Parameters** | 498,530 | 19,266 | 19,554 | 19,842 | **20,930** (+1,088 PersistenceGate) |
| **Multi-Turn Mode** | ❌ (Single-turn) | ❌ (Single-turn) | ❌ (Single-turn) | ❌ (Single-turn) | **✅ 3-turn sequences** |
| **State Carry-Over** | None | None | None | None | **Gated per-slot blend** |
| **Adaptive Steps** | Fixed (4) | Fixed (4) | Fixed (4) | Dynamic (~4.58) | Dynamic (~4.5) |
| **Active Slots** | 1 / 1 | 8 / 8 | 2.19 / 8 | 2.19 / 8 | 2.19 / 8 |
| **Regime Tested** | Single | Single | Single | Mixed | Multi-turn Standalone |

---

## 4. Scientific Takeaways & Next Step

1. **Confirmation of Baseline Stability**:
   - The `PersistenceGate` initialization (`init_bias = -2.0`) functions as intended: when persistence is not strictly required, the network safely falls back to standard encoding without degrading performance ($99.53\%$ accuracy).

2. **The Limit of the Standalone Regime**:
   - In a standalone setup where full history is provided at each turn, the model has sufficient capacity to recalculate answers from scratch. Memory provides a minor benefit on longer chains ($\Delta = +0.80\%$ on Turn 3), but the gate remains largely closed.

3. **Transition to Incremental Regime (Option B)**:
   - To make persistence an **existential requirement**, we now switch to the **Incremental Regime**:
     - **Turn 0**: Initial balance + first transaction + item query.
     - **Turn 1+**: Only the new transaction + item query *(initial balance and past transactions are omitted)*.
   - Under Option B, a reset model has zero knowledge of the protagonist's starting funds and must fail on Turns 2 and 3 ($\approx 50\%$ random chance). A persistent model must keep its memory gates active to succeed.
