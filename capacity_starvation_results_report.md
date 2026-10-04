# LIMINAL — Capacity Starvation Analysis: Dynamic Externalisation Under Working Memory Constraints

**Project**: Adaptive Internal–External Latent Workspace (LIMINAL)  
**Date**: September 2026  
**Status**: Logit Audit & Experimental Control Roadmap  
**Stack**: PyTorch · NumPy · Matplotlib · PyYAML · pytest  

---

## 1. Executive Summary & Critical Discoveries

Following our initial multi-seed disentanglement runs on the 5-turn task, we conducted a rigorous **internal logit audit** and inspected the configuration parameters across all evaluated models. Two decisive findings emerged:

1. **The Logit Audit Differentiates Sub-Threshold Inactivity from Pathological Collapse**:
   - The dead Colab gate was pushed deeply negative (bias = $-1.172$, maximum logit = $-3.73$, 0.0% positive).
   - In contrast, the Disentangled $N=3$ models on the standard task are **healthy and near-boundary**:
     - **Seed 42**: Final bias $= \mathbf{+0.021}$, maximum logit $= \mathbf{+3.87}$ ($1.3\%$ positive writes). The gate is actively responsive.
     - **Seed 43**: Final bias $= \mathbf{-0.124}$, maximum logit $= \mathbf{-2.14}$ ($0.0\%$ positive writes). It sits directly below the decision boundary, not frozen in deep saturation.
2. **Major Training Hyperparameter Confound Identified in `capacity_starvation.yaml`**:
   - Inspecting `configs/experiments/capacity_starvation.yaml` revealed that it was formulated as a "rescue" experiment with different loss dynamics:
     - `write_sparsity_lambda: 0.0` (zero penalty on writing, vs $0.01$ in the baseline)
     - `write_gate_bias_init: 1.0` (gate pre-opened positive, vs $0.0$ neutral in the baseline)
   - Therefore, the 2.83 slots/step write rate cannot be claimed as an emergent response to $N=3$ until we run the **$N=8$ control under the exact same zero-penalty, pre-opened settings**.

---

## 2. Gate Logit Distribution Audit Across All Checkpoints

We evaluated raw pre-activation write gate logits across 100 test sequences for each checkpoint:

| Checkpoint Identifier | Logit Range $[L_{\min}, L_{\max}]$ | Mean Logit $\pm$ Std | $\% > 0$ (Writes) | Final Layer Bias | Diagnostic Status |
|---|:---:|:---:|:---:|:---:|---|
| **Colab $N=8$ (Pre-Fix)** | $[-23.36, -3.73]$ | $-9.75 \pm 2.20$ | 0.0% | $-1.172$ | **Pathologically Collapsed** (Forced shut) |
| **Local $N=8$ (Seed 42)** | $[-20.21, +2.77]$ | $-7.09 \pm 2.92$ | 1.6% | $-0.067$ | **Active Dynamic** (Selective writes) |
| **Disentangled $N=3$ (Seed 42)** | $[-60.89, +3.87]$ | $-17.08 \pm 12.52$ | 1.3% | $+0.021$ | **Active Dynamic** (Near-zero writes, responsive) |
| **Disentangled $N=3$ (Seed 43)** | $[-16.12, -2.14]$ | $-6.69 \pm 2.03$ | 0.0% | $-0.124$ | **Sub-Threshold** (Healthy, non-firing) |
| **Capacity Starvation $N=3$** | $[-3.70, +7.55]$ | $+1.80 \pm 1.63$ | 92.9% | $+1.087$ | **High Saturation** (Pre-open + 0 penalty) |

### Interpretation:
- `Disentangled N=3 (Seed 42)` reaches a maximum logit of $+3.87$, proving that the gate MLP is fully alive and capable of firing when its inputs demand it.
- In both Seed 42 and Seed 43 on the standard 5-turn task, the gate maintains a near-neutral bias ($-0.12$ to $+0.02$), confirming that the near-zero write rate on the standard task is a genuine, learned determination that external memory is unneeded.

---

## 3. The Unresolved Confounds on the Hard Task

The current hypothesis ("externalisation is driven by entity demand relative to capacity, $E > N$") rests on comparing:
- Standard task ($E \le N$): $\le 0.12$ slots written
- Hard task ($E > N$): $2.83$ slots written

However, the hard task run modified:
1. $N=8 \to 3$
2. Turns $5 \to 7$
3. Task `std` $\to$ `hard`
4. $\lambda_{\text{write}}: 0.01 \to 0.0$
5. Bias init: $0.0 \to 1.0$

### The Crucial Missing Controls:
- **Control A ($N=8$ on Hard Task, same settings)**: If $N=8$ with $\lambda_{\text{write}}=0.0$ and $\text{bias}=1.0$ also writes to all 8 slots, then writing is simply driven by the unpenalized positive bias, not capacity starvation.
- **Control B (Multi-Seed on Hard Task)**: The +8.65% advantage on the hard task currently rests on a single run (Seed 42). Running Seed 43 is required to confirm whether this advantage holds against seed variance (~2.4pp).

---

## 4. Prioritized Action Plan

| Priority | Action | Purpose |
|:---:|---|---|
| **P1** | **$N=8$ Control on `ext_sensitive_hard` (7T)** | Test whether a well-provisioned model also writes heavily under identical training dynamics. |
| **P2** | **Seed 43 on `capacity_starvation` ($N=3$, 7T, hard)** | Replicate the headline +8.65% result on a second seed to measure variance. |
| **P3** | **$N=3$ on Hard Task with $\lambda_{\text{write}} = 0.01$** | Test whether $N=3$ opens its gate even when writing incurs a cost. |
