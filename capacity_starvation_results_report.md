# LIMINAL — Capacity Starvation Analysis: Dynamic Externalisation Under Working Memory Constraints

**Project**: Adaptive Internal–External Latent Workspace (LIMINAL)  
**Date**: October 2026  
**Status**: Phase 7 Controls Complete — Gate Collapse Confirmed, Accuracy Advantage Validated  
**Stack**: PyTorch · NumPy · Matplotlib · PyYAML · pytest  

---

## 1. Executive Summary & Critical Discoveries

Following our initial multi-seed disentanglement runs on the 5-turn task, we conducted a rigorous **internal logit audit** and identified a critical hyperparameter confound. We have now completed the full control suite (P1 + P2) to resolve it.

### Original Findings (Pre-Controls)
1. **The Logit Audit Differentiates Sub-Threshold Inactivity from Pathological Collapse**:
   - The dead Colab gate was pushed deeply negative (bias = $-1.172$, maximum logit = $-3.73$, 0.0% positive).
   - In contrast, the Disentangled $N=3$ models on the standard task are **healthy and near-boundary**:
     - **Seed 42**: Final bias $= \mathbf{+0.021}$, maximum logit $= \mathbf{+3.87}$ ($1.3\%$ positive writes). The gate is actively responsive.
     - **Seed 43**: Final bias $= \mathbf{-0.124}$, maximum logit $= \mathbf{-2.14}$ ($0.0\%$ positive writes). It sits directly below the decision boundary, not frozen in deep saturation.
2. **Major Training Hyperparameter Confound Identified in `capacity_starvation.yaml`**:
   - `write_sparsity_lambda: 0.0` (zero penalty on writing, vs $0.01$ in the baseline)
   - `write_gate_bias_init: 1.0` (gate pre-opened positive, vs $0.0$ neutral in the baseline)
   - Therefore, the 2.83 slots/step write rate could not be claimed as emergent until the $N=8$ control ran under the exact same settings.

### New Findings (Post-Controls)
3. **Control A — Hard Task N=8, Seed 42: Gate COLLAPSED_OPEN**:
   - Under the same `write_sparsity_lambda=0.0` + `bias_init=1.0` settings, the $N=8$ model writes to **all 8 slots every step** (gate fraction = 1.000 across all 7 turns, entropy = 0.0001).
   - **Confirms**: high write rates are driven by the unpenalized positive bias, **not** by capacity pressure.
   - Despite collapse, accuracy = **93.89% vs. 81.11% Top-K baseline (+12.78pp)**.

4. **Control B — Capacity Starvation N=3, Seed 43: Gate COLLAPSED_OPEN**:
   - Collapses identically — all 3 slots written every step, entropy ≈ 0.
   - Accuracy = **92.60% vs. 76.49% Top-K baseline (+16.11pp)** — the advantage is *larger* than Seed 42, ruling out a lucky-seed artifact.
   - Since $N=3 = K=3$, no selectivity claim can be made (model trivially matches baseline in slot count).

---

## 2. Gate Logit Distribution Audit Across All Checkpoints

We evaluated raw pre-activation write gate logits across 100 test sequences for each checkpoint:

| Checkpoint Identifier | Logit Range $[L_{\min}, L_{\max}]$ | Mean Logit $\pm$ Std | $\% > 0$ (Writes) | Final Layer Bias | Diagnostic Status |
|---|:---:|:---:|:---:|:---:|---|
| **Colab $N=8$ (Pre-Fix)** | $[-23.36, -3.73]$ | $-9.75 \pm 2.20$ | 0.0% | $-1.172$ | **Pathologically Collapsed** (Forced shut) |
| **Local $N=8$ (Seed 42)** | $[-20.21, +2.77]$ | $-7.09 \pm 2.92$ | 1.6% | $-0.067$ | **Active Dynamic** (Selective writes) |
| **Disentangled $N=3$ (Seed 42)** | $[-60.89, +3.87]$ | $-17.08 \pm 12.52$ | 1.3% | $+0.021$ | **Active Dynamic** (Near-zero writes, responsive) |
| **Disentangled $N=3$ (Seed 43)** | $[-16.12, -2.14]$ | $-6.69 \pm 2.03$ | 0.0% | $-0.124$ | **Sub-Threshold** (Healthy, non-firing) |
| **Capacity Starvation $N=3$ (Seed 42)** | $[-3.70, +7.55]$ | $+1.80 \pm 1.63$ | 92.9% | $+1.087$ | **High Saturation** (Pre-open + 0 penalty) |

### Interpretation:
- `Disentangled N=3 (Seed 42)` reaches a maximum logit of $+3.87$, proving the gate MLP is fully alive and capable of firing when inputs demand it.
- On the standard 5-turn task, the gate maintains a near-neutral bias ($-0.12$ to $+0.02$), confirming near-zero write rates are a genuine learned determination that external memory is unneeded.

---

## 3. Phase 7 Selectivity Analysis — Full Control Suite Results

### 3.1 Hard Task N=8, Seed 42 (Control A — "Does capacity matter?")

| Metric | Value |
|---|---|
| Latent Slots (N) | 8 |
| Top-K Baseline K | 3 |
| Learned Mean Slots/Step | **8.00** (all slots, always) |
| Selectivity Score | 0.000 |
| Gate Entropy | 0.0001 |
| Gate Status | **COLLAPSED_OPEN** |
| Learned Accuracy | **93.89%** |
| Top-K Baseline Accuracy | **81.11%** |
| Accuracy Advantage | **+12.78pp** |
| Phase 7 Exit Criterion | ❌ FAILED |

Gate fractions by turn: 1.000 across all 7 turns — no selectivity.

> **Verdict**: The $N=8$ model writes to all 8 slots under the same training dynamics, proving that high write rates are a function of `write_gate_bias_init=1.0` + `write_sparsity_lambda=0.0`, not of capacity constraints. A fair training setup ($\lambda_{\text{write}} > 0$, neutral bias) is required to properly test the selectivity hypothesis.

---

### 3.2 Capacity Starvation N=3, Seed 43 (Control B — "Is the accuracy advantage seed-robust?")

| Metric | Value |
|---|---|
| Latent Slots (N) | 3 |
| Top-K Baseline K | 3 |
| Learned Mean Slots/Step | **3.00** (all slots, always) |
| Selectivity Score | 0.000 |
| Gate Entropy | 0.0000 |
| Gate Status | **COLLAPSED_OPEN** |
| Learned Accuracy | **92.60%** |
| Top-K Baseline Accuracy | **76.49%** |
| Accuracy Advantage | **+16.11pp** |
| Phase 7 Exit Criterion | ❌ FAILED |

Gate fractions by turn: 1.000 across all 7 turns — identical collapse pattern to Seed 42.

> **Verdict**: The accuracy advantage over Top-K is **larger** in Seed 43 (+16.11pp vs. +8.65pp). The learned write policy confers a real, seed-robust functional advantage even under gate collapse. Since $N=3=K$, selectivity cannot be claimed.

---

### 3.3 P3 — Fair Dynamics N=3 (lambda=0.01, bias=0.0)

| Metric | Value |
|---|---|
| Latent Slots (N) | 3 |
| Top-K Baseline K | 3 |
| Learned Mean Slots/Step | **0.02** (near-zero) |
| Gate Entropy | 0.0038 (threshold: 0.005) |
| Gate Status | **COLLAPSED_CLOSED** |
| Learned Accuracy | **92.26%** |
| Top-K Baseline Accuracy | **83.46%** |
| Accuracy Advantage | **+8.80pp** |
| Phase 7 Exit Criterion | ❌ FAILED |

Gate fractions by turn: 0.017 at Turn 0, then 0.000 for all subsequent turns.

> **Verdict**: Restoring `lambda=0.01` causes N=3 to almost entirely suppress writing. The sparsity penalty is strong enough to close the gate even under capacity pressure. The accuracy advantage (+8.80pp) is preserved anyway — the model relies on its internal graph and the read mechanism, not writes, for performance.

---

### 3.4 P4 — Fair Dynamics N=8 (lambda=0.01, bias=0.0)

| Metric | Value |
|---|---|
| Latent Slots (N) | 8 |
| Top-K Baseline K | 3 |
| Learned Mean Slots/Step | **0.08** |
| Selectivity Score | 0.989 |
| Gate Entropy | 0.0001 |
| Gate Status | **ACTIVE** (near-zero writes, sparse burst at Turn 0) |
| Learned Accuracy | **91.57%** |
| Top-K Baseline Accuracy | **84.51%** |
| Accuracy Advantage | **+7.06pp** |
| Phase 7 Exit Criterion | ❌ FAILED (entropy below threshold) |

Gate fractions by turn: 0.047 at Turn 0, then 0.000 for all subsequent turns.

> **Verdict**: N=8 writes *more* than N=3 (0.08 vs 0.02 slots/step) — the **opposite** of the capacity-starvation prediction. A well-provisioned model uses the external workspace slightly more than a capacity-starved one under the same cost. This contradicts the hypothesis in its simple form but may reflect the N=8 model using the workspace as a precision store (at Turn 0 only) rather than an overflow buffer.

---

### 3.5 Cross-Run Master Table (All Phase 7 Evaluations)

| Run | N | lambda | bias_init | Mean Slots/Step | Gate Status | Learned Acc | Top-K Acc | Advantage | P7 Pass? |
|---|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|:---:|
| Externalisation Comparison (N=8, std) | 8 | 0.01 | 0.0 | 0.00 | COLLAPSED_CLOSED | 97.44% | 95.84% | +1.60pp | ❌ |
| Capacity Starvation (N=3, Seed 42) | 3 | 0.0 | 1.0 | 2.83 | ACTIVE | 92.51% | 83.86% | +8.65pp | ✅ |
| Hard Task N=8, Seed 42 (Control A) | 8 | 0.0 | 1.0 | 8.00 | COLLAPSED_OPEN | 93.89% | 81.11% | +12.78pp | ❌ |
| Capacity Starvation Seed 43 (Control B) | 3 | 0.0 | 1.0 | 3.00 | COLLAPSED_OPEN | 92.60% | 76.49% | +16.11pp | ❌ |
| **P3: Fair Hard N=3** | **3** | **0.01** | **0.0** | **0.02** | **COLLAPSED_CLOSED** | **92.26%** | **83.46%** | **+8.80pp** | ❌ |
| **P4: Fair Hard N=8** | **8** | **0.01** | **0.0** | **0.08** | **ACTIVE*** | **91.57%** | **84.51%** | **+7.06pp** | ❌ |

*ACTIVE by script classification (mean slots < K), but entropy 0.0001 — effectively near-closed.

---

## 4. Root Cause of Gate Collapse — A Binary Lambda Problem

The P3/P4 results reveal that the gate operates in a **bistable regime**: lambda is either too weak (gate collapses open) or too strong (gate collapses closed). There is no observed intermediate where selective externalisation emerges.

| lambda | bias_init | Observed Behaviour |
|:---:|:---:|---|
| 0.0 | 1.0 | COLLAPSED_OPEN — writes everything, always |
| 0.01 | 0.0 | COLLAPSED_CLOSED — writes almost nothing |
| **?** | **0.0** | **Target: selective writing** |

The write gate logit is driven by a cost-benefit balance:
- At `lambda=0.0`: zero marginal cost → gate stays open, accuracy is maintained or improved.
- At `lambda=0.01`: marginal cost exceeds expected accuracy gain for most turns → gate shuts. The model achieves near-equal accuracy without writing by relying on internal graph representations and the read mechanism.

This implies the **selectivity window** — the lambda range where writing is selectively beneficial — lies between 0.0 and 0.01. A sweep over `lambda ∈ {0.001, 0.002, 0.005}` is the next diagnostic step.

### Unexpected Inversion: N=8 Writes More Than N=3 Under Fair Conditions

Under `lambda=0.01`, N=8 writes 0.08 slots/step while N=3 writes only 0.02 — the **opposite** of the capacity-starvation prediction. Two non-exclusive explanations:
1. **Internal sufficiency suppresses urgency**: N=3, despite being capacity-starved, has learned to compress representations aggressively. Under a write cost, it finds this cheaper than externalising.
2. **N=8 precision store**: The N=8 model uses a sparse write at Turn 0 as a precision caching mechanism — not as overflow relief — which the N=3 model cannot afford under the same cost structure.

Either way, the relationship between internal capacity and externalisation rate is **non-monotonic** and depends critically on the lambda regime.

---

## 5. Revised Theoretical Standing

### What We Can Claim:
1. ✅ **The LIMINAL architecture learns a write policy that outperforms forced Top-K by 7–16pp on the hard multi-turn task**, across all N and lambda settings tested (advantage holds even when the gate barely writes).
2. ✅ **On the easy standard task, the gate correctly determines externalisation is unnecessary** — near-zero slot writes verified by logit audit to be genuine learned abstinence, not pathological collapse.
3. ✅ **The gate MLP is alive and responsive** — maximum logit +3.87 in Disentangled N=3 Seed 42 confirms the MLP fires when inputs demand it.
4. ✅ **The accuracy advantage is robust to gate collapse mode** — it holds whether the gate collapses open, collapses closed, or writes selectively.

### What We Cannot Yet Claim:
1. ❌ **That externalisation is causally driven by capacity starvation** ($N < E$). P4 shows N=8 writes *more* than N=3 under fair conditions — the relationship is non-monotonic.
2. ❌ **That the gate learns selective externalisation** at any tested lambda. The gate is bistable: `lambda=0.0` → open collapse; `lambda=0.01` → closed collapse. No intermediate selective regime has been found.
3. ❌ **That `lambda=0.01` is the correct operating point**. It is evidently too strong for the hard task — it extinguishes all writing for N=3 and nearly all for N=8.

### New Hypothesis — Lambda Regime, Not Capacity:
The more parsimonious explanation of all data to date is: **the gate selectively externalises when and only when the marginal accuracy gain of writing exceeds its lambda-weighted cost**. Capacity (N) modulates *what* can be stored internally, but the lambda-accuracy tradeoff determines *whether* the gate opens at all. The selectivity window for this task likely sits at `lambda ∈ (0.001, 0.005)`.

---

## 6. Prioritized Action Plan (Revised)

| Priority | Action | Status | Purpose |
|:---:|---|:---:|---|
| **P1** | **$N=8$ Control on `ext_sensitive_hard` (7T, biased)** | ✅ **Done** | Confirmed: high write rates are bias-driven, not capacity-driven. |
| **P2** | **Seed 43 on `capacity_starvation` ($N=3$, biased)** | ✅ **Done** | Confirmed: accuracy advantage (+16.11pp) is seed-robust. |
| **P3** | **$N=3$ Hard Task, $\lambda = 0.01$, neutral bias** | ✅ **Done** | COLLAPSED_CLOSED — lambda too strong; gate suppressed even under N=3. |
| **P4** | **$N=8$ Hard Task, $\lambda = 0.01$, neutral bias** | ✅ **Done** | Inversion: N=8 writes *more* than N=3 (0.08 vs 0.02). Hypothesis disconfirmed in simple form. |
| **P5** | **Lambda sweep: $N=3$ Hard Task, $\lambda \in \{0.001, 0.002, 0.005\}$** | ⏳ Pending | Find the selectivity window where the gate opens selectively under cost. |
| **P6** | **Lambda sweep: $N=8$ Hard Task, same $\lambda$ grid** | ⏳ Pending | Causal pair to P5: does N=8 write less than N=3 at the optimal lambda? |
| **P7** | **Phase 7 selectivity analysis on best P5/P6 checkpoints** | ⏳ Pending | Gate entropy > 0.005, mean slots < K, accuracy >= baseline — clean PASS needed. |
