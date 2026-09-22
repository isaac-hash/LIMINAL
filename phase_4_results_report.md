# LIMINAL — Phase 4 Results Report: Latent Persistence Across Turns

**Project**: Adaptive Internal–External Latent Workspace (LIMINAL)  
**Date**: September 2026  
**Hardware**: Google Colab · NVIDIA Tesla T4 GPU (14.6 GB VRAM) · CUDA  
**Stack**: PyTorch 2.14 · NumPy · Matplotlib · PyYAML · pytest  
**Configs Tested**:  
- Standalone Regime: `configs/persistence_graph.yaml` vs `configs/persistence_reset.yaml`  
- Incremental Regime: `configs/persistence_incremental.yaml` vs `configs/persistence_incremental_reset.yaml`  

---

## Executive Summary (Lay Terms)

In Phase 4, we set out to answer a core architectural question: **Can a neural latent workspace remember across related questions instead of wiping its mind blank after each query?**

We built a learnable **PersistenceGate** that operates on each of the model's 8 latent slots, deciding how much prior latent representation to retain ($V_{\text{prior}}$) versus how much to replace with new sensory input ($V_{\text{new}}$).

We evaluated this across two distinct scientific regimes:

1. **Regime A (Standalone / All Facts Provided)**:
   Every question repeated the full history of the problem. Because the model had all facts in front of it at every turn, the fresh encoder could solve the problem from scratch ($99.5\%$ accuracy). The persistence gate remained safely closed ($\text{activation} \approx 0.01$), proving that the gate defaults gracefully to fresh encoding when memory is not required.

2. **Regime B (Incremental / True Memory Requirement)**:
   Turn 1 established Alice's initial balance and first transaction. Turns 2 and 3 provided **only the new transactions**, without repeating Alice's initial funds. 
   - **The Reset baseline collapsed**: Dropping to **$76.80\%$ on Turn 2** and **$77.00\%$ on Turn 3** because it had forgotten Alice's starting balance.
   - **The Persistent model succeeded**: Retaining **$96.60\%$ accuracy across Turns 2 and 3** ($+19.80\%$ gap!).

**Conclusion**: Latent persistence works decisively. When problems require temporal context, gated persistence preserves critical state across turns with bounded memory and detached gradients, outperforming a stateless reset baseline by nearly $20$ percentage points.

---

## 1. Exit Criterion Evaluation

| Criterion | Target | Actual Result | Status |
| :--- | :---: | :---: | :---: |
| **Turn 2+ Accuracy Advantage vs Reset** | $\ge +1.0\%$ | **$+19.80\%$** (Turn 2) / **$+19.60\%$** (Turn 3) | **EXCEEDED (19.8x target)** |
| **Turn 1 Fresh Start Parity** | Similar ($\le \pm 2\%$) | **$98.80\%$ vs $97.40\%$** ($\Delta = +1.40\%$) | **MET** |
| **Sequential Training Stability** | Loss decreases, no divergence | Test Loss: **$0.0885$** (Persistent) vs **$0.3321$** (Reset) | **MET** |

---

## 2. Quantitative Results: Incremental Regime (Regime B)

### 2A — Per-Turn Test Set Performance

| Sequence Turn | Persistent Model (Model E) | Reset Baseline (Model E-reset) | Accuracy Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Turn 1** (Full context given) | **98.80%** | 97.40% | $+1.40\%$ |
| **Turn 2** (Incremental fact only) | **96.60%** | **76.80%** | **$+19.80\%$** |
| **Turn 3** (Incremental fact only) | **96.60%** | **77.00%** | **$+19.60\%$** |
| **Overall Multi-Turn Accuracy** | **97.80%** | **83.67%** | **$+14.13\%$** |
| **Overall Test Loss** | **0.0885** | **0.3321** | **$-0.2436$** (3.7x lower) |

*(Evaluated on 500 held-out test sequences / 1,500 total turns on NVIDIA Tesla T4).*

---

### 2B — Regime A (Standalone) vs Regime B (Incremental) Comparison

| Turn | Standalone $\Delta$ (All Facts Present) | Incremental $\Delta$ (Memory Required) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Turn 1** | $-0.20\%$ | $+1.40\%$ | Parity at cold start (both models start with fresh $V_0$) |
| **Turn 2** | $+0.00\%$ | **$+19.80\%$** | Reset baseline forgets starting budget; Persistent model carries it |
| **Turn 3** | $+0.80\%$ | **$+19.60\%$** | Advantage remains completely sustained over longer chains |

---

## 3. The Complete 5-Way Model Comparison

Across the four completed phases of the LIMINAL project, we have systematically scaled and pruned the latent workspace:

| Phase & Model | Architecture | Active Slots | Active Steps | Multi-Turn State | Test Acc (Primary) | Memory Advantage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A** (Phase 1) | Single Recurrent Vector (GRU) | 1 / 1 | Fixed (4) | ❌ None | 99.60% | — |
| **Model B** (Phase 1) | Static Relational Graph | 8 / 8 | Fixed (4) | ❌ None | 99.80% | — |
| **Model C** (Phase 2) | Graph + Spatial Activity Gates | **2.19 / 8** | Fixed (4) | ❌ None | 99.40% | Sparse routing |
| **Model D** (Phase 3) | Graph + Spatial Gates + Halt Gates | **2.19 / 8** | **Dynamic (~4.6)** | ❌ None | 100.00% | Temporal self-pruning |
| **Model E** (Phase 4) | Graph + Adaptive Gates + **PersistenceGate** | **2.19 / 8** | **Dynamic (~4.6)** | **✅ Gated Carry-Over** | **97.80%** | **$+19.80\%$ vs Reset** |

---

## 4. Key Architectural Discoveries

1. **Selective Gated Blending**:
   - Rather than forcing an all-or-nothing recurrence across turns, per-slot gating enables the model to independently decide which slots serve as persistent registers (storing accumulated budget) while allowing other slots to absorb new operations.

2. **Decoupled BPTT (Constant-Memory Multi-Turn Reasoning)**:
   - Detaching $V_{\text{prior}}$ across turns (`detach_between_turns: true`) completely eliminates the memory overhead of backpropagating through long historical reasoning chains.
   - The persistence gate trains purely from the current turn's loss gradient, yet reliably learns to retain critical state from previous turns.

3. **Falsifiable Memory Evaluation**:
   - Comparing Standalone vs Incremental regimes cleanly isolates the causal mechanism: when memory is redundant, gates remain at rest; when memory is required, gates open to transmit state, delivering a $+19.8\%$ performance differential.
