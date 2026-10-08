# LIMINAL — Lambda Sweep Report: Probing the Selectivity Window

**Project**: Adaptive Internal-External Latent Workspace (LIMINAL)
**Date**: October 2026
**Status**: Phase 8 Diagnostics Complete — Causal ablation, gate variance, loss normalization audit done ✅
**Continuation of**: `capacity_starvation_results_report.md` (Phase 7 Controls Complete)
**Stack**: PyTorch · NumPy · Matplotlib · PyYAML · pytest

---

## 1. Context & Motivation

This report is a direct continuation of the Phase 7 capacity-starvation control suite.
The previous report (`capacity_starvation_results_report.md`) established the following
critical findings:

1. **Gate bistability**: `write_sparsity_lambda=0.0` -> gate COLLAPSED_OPEN (writes always);
   `write_sparsity_lambda=0.01` -> gate COLLAPSED_CLOSED (writes never). No selective
   intermediate regime was observed at either tested lambda.

2. **N=8 > N=3 write inversion**: Under fair conditions (`lambda=0.01`, `bias=0.0`),
   N=8 wrote *more* than N=3 (0.08 vs 0.02 slots/step), directly contradicting the
   simple capacity-starvation hypothesis.

3. **Accuracy advantage is collapse-mode-agnostic**: The learned gate outperforms
   forced Top-K by 7-16pp regardless of whether the gate is open, closed, or selective.

4. **Proposed selectivity window**: `lambda in (0.001, 0.005)` — the regime where
   writing is cost-permitted but not cost-suppressed.

**Goal of this sweep**: Identify the lambda value(s) at which the gate discovers a
selective writing policy (entropy > 0.005, mean slots < K) while retaining the accuracy
advantage over Top-K.

---

## 2. Experiment Grid

All six runs share the same architecture, data, and training settings as the P3/P4
baseline pair. The only variables are `write_sparsity_lambda` and N.

| ID | Config | N | λ | bias_init | checkpoint_dir |
|---|---|:---:|:---:|:---:|---|
| P5-lam001 | `fair_hard_n3_lam001.yaml` | 3 | **0.001** | 0.0 | `results/fair_hard_n3_lam001` |
| P5-lam002 | `fair_hard_n3_lam002.yaml` | 3 | **0.002** | 0.0 | `results/fair_hard_n3_lam002` |
| P5-lam005 | `fair_hard_n3_lam005.yaml` | 3 | **0.005** | 0.0 | `results/fair_hard_n3_lam005` |
| P6-lam001 | `fair_hard_n8_lam001.yaml` | 8 | **0.001** | 0.0 | `results/fair_hard_n8_lam001` |
| P6-lam002 | `fair_hard_n8_lam002.yaml` | 8 | **0.002** | 0.0 | `results/fair_hard_n8_lam002` |
| P6-lam005 | `fair_hard_n8_lam005.yaml` | 8 | **0.005** | 0.0 | `results/fair_hard_n8_lam005` |

**Shared fixed settings** (all 6 runs):
- `write_gate_bias_init: 0.0` (neutral — no thumb on the scale)
- `write_sparsity_warmup_epochs: 5` (penalty applied from epoch 5 onward)
- `gumbel_tau`: annealed 1.0 -> 0.1 over 30 epochs
- `epochs: 60`, `batch_size: 8`, `lr: 3e-4`, task: `ext_sensitive_hard` (7 turns)

---

## 3. Phase 7 Exit Criterion

A run is considered a **Phase 7 PASS** if all three conditions are satisfied:

| Condition | Threshold | Interpretation |
|---|---|---|
| `mean_slots_written_per_step` | < K (= 3) | Gate is selective, not saturating all slots |
| `gate_entropy` | > 0.005 | Gate is stochastic/selective, not deterministically collapsed |
| Learned accuracy | >= Top-K baseline | Selective policy retains functional advantage |

---

## 4. Prior Anchor Points (from Phase 7 Controls)

For reference when interpreting sweep results:

| Run | N | λ | Mean Slots/Step | Gate Status | Learned Acc | Top-K Acc | Advantage |
|---|:---:|:---:|:---:|---|:---:|:---:|:---:|
| Externalisation Comparison | 8 | 0.010 | 0.00 | COLLAPSED_CLOSED | 97.44% | 95.84% | +1.60pp |
| Capacity Starvation S42 | 3 | 0.000 | 2.83 | ACTIVE | 92.51% | 83.86% | +8.65pp |
| Control A (N=8, biased) | 8 | 0.000 | 8.00 | COLLAPSED_OPEN | 93.89% | 81.11% | +12.78pp |
| Control B (N=3 S43, biased) | 3 | 0.000 | 3.00 | COLLAPSED_OPEN | 92.60% | 76.49% | +16.11pp |
| P3: Fair N=3 | 3 | 0.010 | 0.02 | COLLAPSED_CLOSED | 92.26% | 83.46% | +8.80pp |
| P4: Fair N=8 | 8 | 0.010 | 0.08 | ACTIVE* | 91.57% | 84.51% | +7.06pp |

*ACTIVE by classifier (mean < K) but entropy = 0.0001 — effectively near-closed.

---

## 5. Training Results Summary

All training runs converged cleanly. Final test accuracies:

| Config | N | λ | Final Test Acc | Best Val Acc (epoch) |
|---|:---:|:---:|:---:|:---:|
| fair_hard_n3_lam001 | 3 | 0.001 | **94.09%** | 94.14% (ep55) |
| fair_hard_n3_lam002 | 3 | 0.002 | **93.29%** | 93.20% (ep36) |
| fair_hard_n3_lam005 | 3 | 0.005 | **92.14%** | 91.80% (ep53) |
| fair_hard_n8_lam001 | 8 | 0.001 | **94.03%** | 94.31% (ep36) |
| fair_hard_n8_lam002 | 8 | 0.002 | **93.37%** | 93.43% (ep50) |
| fair_hard_n8_lam005 | 8 | 0.005 | **92.57%** | 92.83% (ep54) |

**Trend**: Higher lambda lowers final accuracy slightly but monotonically for both N.
N=3 and N=8 track closely — the gap is small (<2pp) across the full lambda range tested.

---

## 6. P5 Results — N=3 Lambda Sweep

### 6.1 P5-lam001 — N=3, λ=0.001

```
python scripts/analyze_selectivity.py \
  --config configs/experiments/fair_hard_n3_lam001.yaml \
  --checkpoint results/fair_hard_n3_lam001/best.pt \
  --output-dir results/phase7_eval/fair_hard_n3_lam001
```

| Metric | Value |
|---|---|
| Mean Slots/Step | **0.99** |
| Selectivity Score | 0.670 |
| Gate Entropy | **0.3193** |
| Gate Status | **ACTIVE** |
| Learned Accuracy | **93.86%** |
| Top-K Baseline Accuracy | 86.77% |
| Accuracy Advantage | **+7.09pp** |
| Phase 7 Exit Criterion | **PASSED** ✅ |

Gate fractions by turn:

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Write fraction | 0.505 | 0.243 | 0.389 | 0.328 | 0.494 | 0.231 | 0.071 |

> **Verdict**: Gate is genuinely selective — writes ~1 slot/step on average vs 3 forced
> by Top-K. Turn fractions oscillate between 0.07 and 0.50, indicating turn-sensitive
> write decisions rather than uniform sparsity. Entropy (0.3193) is high — gate is
> confident but not deterministic; it fires situationally. **Phase 7 PASS confirmed.**

---

### 6.2 P5-lam002 — N=3, λ=0.002

```
python scripts/analyze_selectivity.py \
  --config configs/experiments/fair_hard_n3_lam002.yaml \
  --checkpoint results/fair_hard_n3_lam002/best.pt \
  --output-dir results/phase7_eval/fair_hard_n3_lam002
```

| Metric | Value |
|---|---|
| Mean Slots/Step | **0.54** |
| Selectivity Score | 0.821 |
| Gate Entropy | **0.1726** |
| Gate Status | **ACTIVE** |
| Learned Accuracy | **93.29%** |
| Top-K Baseline Accuracy | 90.40% |
| Accuracy Advantage | **+2.89pp** |
| Phase 7 Exit Criterion | **PASSED** ✅ |

Gate fractions by turn:

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Write fraction | 0.362 | 0.115 | 0.265 | 0.124 | 0.184 | 0.091 | 0.004 |

> **Verdict**: Sparser than lam001 (0.54 vs 0.99 slots/step), with lower entropy
> (0.1726) — gate is more confident about *not* writing. Turn 6 write fraction collapses
> to 0.004, suggesting the model has learned that late-sequence externalisation is almost
> never needed. Accuracy advantage narrows (+2.89pp) as the Top-K baseline improves
> under lambda. **Phase 7 PASS confirmed.**

---

### 6.3 P5-lam005 — N=3, λ=0.005

```
python scripts/analyze_selectivity.py \
  --config configs/experiments/fair_hard_n3_lam005.yaml \
  --checkpoint results/fair_hard_n3_lam005/best.pt \
  --output-dir results/phase7_eval/fair_hard_n3_lam005
```

| Metric | Value |
|---|---|
| Mean Slots/Step | **0.10** |
| Selectivity Score | 0.968 |
| Gate Entropy | **0.0267** |
| Gate Status | **ACTIVE** |
| Learned Accuracy | **91.63%** |
| Top-K Baseline Accuracy | 85.77% |
| Accuracy Advantage | **+5.86pp** |
| Phase 7 Exit Criterion | **PASSED** ✅ |

Gate fractions by turn:

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Write fraction | 0.043 | 0.047 | 0.052 | 0.000 | 0.000 | 0.000 | 0.000 |

> **Verdict**: Strikingly sparse — only 0.10 slots/step, with writing confined entirely
> to Turns 0–2 and zero writes in Turns 3–6. The gate has learned a sharp early-burst
> policy: externalise only at the start of the sequence when context is freshest, then
> rely on internal representations for the remainder. This is the highest selectivity
> score in the N=3 sweep (0.968). Entropy (0.0267) is very low — the gate is nearly
> deterministic in its sparse decisions. At lam005, N=3 is approaching the COLLAPSED_CLOSED
> boundary seen at lam=0.01 (P3: 0.02 slots). **Phase 7 PASS confirmed.**

---

## 7. P6 Results — N=8 Lambda Sweep

### 7.1 P6-lam001 — N=8, λ=0.001

```
python scripts/analyze_selectivity.py \
  --config configs/experiments/fair_hard_n8_lam001.yaml \
  --checkpoint results/fair_hard_n8_lam001/best.pt \
  --output-dir results/phase7_eval/fair_hard_n8_lam001
```

| Metric | Value |
|---|---|
| Mean Slots/Step | **1.37** |
| Selectivity Score | 0.828 |
| Gate Entropy | **0.1601** |
| Gate Status | **ACTIVE** |
| Learned Accuracy | **94.34%** |
| Top-K Baseline Accuracy | 70.94% |
| Accuracy Advantage | **+23.40pp** |
| Phase 7 Exit Criterion | **PASSED** ✅ |

Gate fractions by turn:

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Write fraction | 0.225 | 0.201 | 0.154 | 0.098 | 0.183 | 0.107 | 0.047 |

> **Verdict**: Gate is selective across all 7 turns with a gradual write-fraction
> decline from Turn 0 to Turn 6. The +23.40pp accuracy advantage is the largest seen
> across the entire experiment series — at lambda=0.001 the Top-K baseline is severely
> degraded (70.94%), revealing how much the forced K=3 policy hurts N=8 when better
> selective strategies exist. **Phase 7 PASS confirmed.**

---

### 7.2 P6-lam002 — N=8, λ=0.002

```
python scripts/analyze_selectivity.py \
  --config configs/experiments/fair_hard_n8_lam002.yaml \
  --checkpoint results/fair_hard_n8_lam002/best.pt \
  --output-dir results/phase7_eval/fair_hard_n8_lam002
```

| Metric | Value |
|---|---|
| Mean Slots/Step | **1.02** |
| Selectivity Score | 0.873 |
| Gate Entropy | **0.1342** |
| Gate Status | **ACTIVE** |
| Learned Accuracy | **93.06%** |
| Top-K Baseline Accuracy | 75.49% |
| Accuracy Advantage | **+17.57pp** |
| Phase 7 Exit Criterion | **PASSED** ✅ |

Gate fractions by turn:

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Write fraction | 0.179 | 0.088 | 0.147 | 0.091 | 0.195 | 0.137 | 0.054 |

> **Verdict**: Sparser than lam001 (1.02 vs 1.37 slots/step), highest selectivity
> score in the sweep (0.873). Turn fractions are more uniform than lam001 — the gate
> writes slightly across all turns rather than concentrating at Turn 0. Entropy is the
> lowest of all clean N=8 runs (0.1342), indicating the gate is more confident in its
> sparse decisions. **Phase 7 PASS confirmed.**

---

### 7.3 P6-lam005 — N=8, λ=0.005

```
python scripts/analyze_selectivity.py \
  --config configs/experiments/fair_hard_n8_lam005.yaml \
  --checkpoint results/fair_hard_n8_lam005/best.pt \
  --output-dir results/phase7_eval/fair_hard_n8_lam005
```

| Metric | Value |
|---|---|
| Mean Slots/Step | **0.51** |
| Selectivity Score | 0.936 |
| Gate Entropy | **0.0516** |
| Gate Status | **ACTIVE** |
| Learned Accuracy | **92.57%** |
| Top-K Baseline Accuracy | 74.29% |
| Accuracy Advantage | **+18.28pp** |
| Phase 7 Exit Criterion | **PASSED** ✅ |

Gate fractions by turn:

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Write fraction | 0.071 | 0.080 | 0.023 | 0.066 | 0.071 | 0.087 | 0.025 |

> **Verdict**: Sparser than lam002 (0.51 vs 1.02 slots/step) with a distinctive
> near-uniform turn distribution — unlike N=3 lam005 which concentrates writes in
> Turns 0–2, N=8 lam005 spreads its sparse writes roughly evenly across all turns
> (0.02–0.09 range). This suggests the N=8 model uses the workspace as a precision
> supplement at any turn where a marginal gain exists, rather than front-loading. The
> +18.28pp advantage over Top-K is the second-largest in the entire study. **Phase 7
> PASS confirmed.**

---

## 8. Cross-Run Sweep Master Table

Anchor rows from prior Phase 7 report shown in italics for context.

| Run | N | λ | Mean Slots | Sel. Score | Gate Entropy | Gate Status | Learned Acc | Top-K Acc | Advantage | P7 Pass? |
|---|:---:|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|:---:|
| *P3: Fair N=3 (anchor)* | *3* | *0.010* | *0.02* | *—* | *0.0038* | *COLLAPSED_CLOSED* | *92.26%* | *83.46%* | *+8.80pp* | ❌ |
| P5-lam001 | 3 | 0.001 | 0.99 | 0.670 | 0.3193 | ACTIVE | 93.86% | 86.77% | +7.09pp | ✅ |
| P5-lam002 | 3 | 0.002 | 0.54 | 0.821 | 0.1726 | ACTIVE | 93.29% | 90.40% | +2.89pp | ✅ |
| **P5-lam005** | **3** | **0.005** | **0.10** | **0.968** | **0.0267** | **ACTIVE** | **91.63%** | **85.77%** | **+5.86pp** | ✅ |
| *P4: Fair N=8 (anchor)* | *8* | *0.010* | *0.08* | *—* | *0.0001* | *ACTIVE†* | *91.57%* | *84.51%* | *+7.06pp* | ❌ |
| P6-lam001 | 8 | 0.001 | 1.37 | 0.828 | 0.1601 | ACTIVE | 94.34% | 70.94% | +23.40pp | ✅ |
| P6-lam002 | 8 | 0.002 | 1.02 | 0.873 | 0.1342 | ACTIVE | 93.06% | 75.49% | +17.57pp | ✅ |
| **P6-lam005** | **8** | **0.005** | **0.51** | **0.936** | **0.0516** | **ACTIVE** | **92.57%** | **74.29%** | **+18.28pp** | ✅ |

†P4 ACTIVE by classifier only; entropy effectively zero — excluded from selectivity window claim.

---

## 9. Key Findings — Complete 6-Run Analysis

### 9.1 Selectivity Window Confirmed Across Full Lambda Range

All six runs PASSED Phase 7. The selectivity window is **broader than originally
hypothesised** — it spans the full tested range `lambda in {0.001, 0.002, 0.005}`:

| λ | N=3 Mean Slots | N=3 Entropy | N=8 Mean Slots | N=8 Entropy |
|:---:|:---:|:---:|:---:|:---:|
| 0.001 | 0.99 | 0.3193 | 1.37 | 0.1601 |
| 0.002 | 0.54 | 0.1726 | 1.02 | 0.1342 |
| 0.005 | **0.10** | **0.0267** | **0.51** | **0.0516** |

All values satisfy: `mean_slots < K=3` AND `entropy > 0.005`. This is the first
confirmed evidence that LIMINAL's learned write gate discovers genuine *selective*
externalisation under a fair cost-benefit regime.

### 9.2 N=8 Write Inversion Is Stable Across the Entire Selectivity Window

At every tested lambda, N=8 writes **more** than N=3:

| λ | N=3 Slots | N=8 Slots | Ratio (N8/N3) |
|:---:|:---:|:---:|:---:|
| 0.001 | 0.99 | 1.37 | 1.38× |
| 0.002 | 0.54 | 1.02 | 1.89× |
| 0.005 | 0.10 | 0.51 | **5.10×** |

The inversion not only persists — it **widens** as lambda increases. At lam005, N=8
writes 5× more than N=3 per step. This is a decisive, lambda-stable disconfirmation
of the simple capacity-starvation hypothesis.

**Revised interpretation (divergent compression strategies)**:
- Under write cost, N=3 has learned to compress so aggressively that external writes
  become nearly unnecessary — internal latent representations absorb all task-relevant
  state, even at the cost of some accuracy (~0.5pp below N=8).
- N=8 maintains a complementary strategy: it uses the workspace as a steady precision
  supplement across all turns, writing a small fraction of slots when the expected
  accuracy gain exceeds the lambda cost.

### 9.3 N=3 lam005: Early-Burst Gating — A New Qualitative Regime

The N=3 lam005 turn profile is qualitatively distinct from all other runs:

| Turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| N=3 lam001 | 0.505 | 0.243 | 0.389 | 0.328 | 0.494 | 0.231 | 0.071 |
| N=3 lam002 | 0.362 | 0.115 | 0.265 | 0.124 | 0.184 | 0.091 | 0.004 |
| **N=3 lam005** | **0.043** | **0.047** | **0.052** | **0.000** | **0.000** | **0.000** | **0.000** |

At lam005, the N=3 gate writes only in Turns 0–2 and is exactly zero for Turns 3–6.
This suggests the model has learned: *"if I must pay to write, only do so at the start
when the context is new — mid-to-late sequence writes never pay back"*. This is
economically rational: reads at Turn t can access any prior write, so an early write
serves the maximum number of subsequent turns.

In contrast, N=8 lam005 spreads its 0.51 slots/step nearly uniformly (0.02–0.09 range)
— consistent with the precision-store model writing selectively but without temporal bias.

### 9.4 Accuracy Advantage: N=8 Dominates, N=3 Narrows at lam002, Recovers at lam005

| λ | N=3 Learned Acc | N=3 Adv. | N=8 Learned Acc | N=8 Adv. |
|:---:|:---:|:---:|:---:|:---:|
| 0.001 | 93.86% | +7.09pp | 94.34% | **+23.40pp** |
| 0.002 | 93.29% | +2.89pp | 93.06% | **+17.57pp** |
| 0.005 | 91.63% | +5.86pp | 92.57% | **+18.28pp** |

N=8 maintains a large advantage (17–23pp) across all lambda. For N=3, the minimum
advantage is at lam002 (+2.89pp), not lam005 — this is because at lam002 the Top-K
baseline improves faster (to 90.40%) than the learned model's accuracy decline. At
lam005, the baseline retreats again (85.77%) while learned accuracy holds at 91.63%,
so the advantage widens back to +5.86pp.

### 9.5 Monotonic Trends Confirmed Across Full Sweep

All key metrics are monotonic in lambda within each N:

| Metric | Trend as λ increases |
|---|---|
| Mean slots/step | Decreases monotonically (both N) |
| Gate entropy | Decreases monotonically (both N) |
| Selectivity score | Increases monotonically (both N) |
| Training accuracy | Decreases monotonically (both N) |
| N=8 / N=3 slots ratio | **Increases** (1.38× → 1.89× → 5.10×) |

### 9.6 Late-Sequence Suppression Is Universal

Turn 6 write fraction is the lowest (or near-lowest) in every run. The pattern is most
pronounced in N=3 where it drops to zero at lam002 and lam005. The gate has correctly
inferred that terminal-turn writes have zero future-read value. This is consistent with
backpropagated temporal credit assignment through the gating mechanism.

---

## 10. Theoretical Standing Update

### What Can Now Be Claimed

1. ✅ **LIMINAL's learned write gate discovers genuine selective externalisation** under
   a fair cost-benefit regime (neutral bias, write penalty). All 6 sweep runs PASSED
   Phase 7 across N=3 and N=8, lambda=0.001–0.005.

2. ✅ **The selectivity window spans at least lambda in {0.001, 0.002, 0.005}** — broader
   than the original (0.001, 0.005) hypothesis. The lower bound likely lies below 0.001
   (gate transitions from open-collapse to selective somewhere < 0.001).

3. ✅ **The accuracy advantage over forced Top-K is robust and large**: +2.89pp to
   +23.40pp across all 6 selective runs. Even the minimum (+2.89pp) is a consistent
   positive result.

4. ✅ **The gate exhibits temporal credit assignment**: it suppresses late-sequence writes
   (especially Turn 6) universally, and at high lambda (N=3 lam005) restricts itself
   to early-burst writing only.

### What Cannot Be Claimed

1. ❌ **That capacity starvation causes externalisation**. The N=8>N=3 inversion is
   stable across 3 lambda values in the selective regime (ratios 1.4× to 5.1×). The
   causal driver appears to be the write cost/benefit ratio operating on each model's
   learned compression strategy, not N relative to entity count.

2. ❌ **That lam=0.001 is the lower bound of the selective window**. The transition from
   COLLAPSED_OPEN to SELECTIVE has not been located — it likely lies below 0.001.

---

## 11. Action Plan

| Priority | Action | Status | Key Result |
|:---:|---|:---:|---|
| **P1** | N=8 Control (biased) | ✅ Done | Gate-open artefact confirmed as bias-driven |
| **P2** | Seed 43 N=3 (biased) | ✅ Done | Accuracy advantage (+16.11pp) seed-robust |
| **P3** | Fair N=3, λ=0.01 | ✅ Done | COLLAPSED_CLOSED |
| **P4** | Fair N=8, λ=0.01 | ✅ Done | Near-CLOSED; N=8>N=3 inversion first observed |
| **P5-lam001** | N=3, λ=0.001 | ✅ PASSED | 0.99 slots/step, entropy=0.319, +7.09pp |
| **P5-lam002** | N=3, λ=0.002 | ✅ PASSED | 0.54 slots/step, entropy=0.173, +2.89pp |
| **P5-lam005** | N=3, λ=0.005 | ✅ PASSED | 0.10 slots/step, entropy=0.027, +5.86pp |
| **P6-lam001** | N=8, λ=0.001 | ✅ PASSED | 1.37 slots/step, entropy=0.160, +23.40pp |
| **P6-lam002** | N=8, λ=0.002 | ✅ PASSED | 1.02 slots/step, entropy=0.134, +17.57pp |
| **P6-lam005** | N=8, λ=0.005 | ✅ PASSED | 0.51 slots/step, entropy=0.052, +18.28pp |
| **P8-A** | Loss normalization audit | ✅ Done | N=8 gets 2.67× per-slot discount at same λ — inversion artifact |
| **P8-B** | Zero-write ablation (N=3, N=8 λ=0.001) | ✅ Done | Writes causally necessary; content active |
| **P8-C** | Within-turn gate variance (N=3, N=8 λ=0.001) | ✅ Done | HIGH content selectivity; saturation at Turn 1–2 |
| **P8-D** | Second seed runs (N=8 λ=0.001, N=3 λ=0.005) | ⏳ Pending | Seed robustness check |
| **P8-E** | Matched per-slot cost rerun (N=3 cell) | ⏳ Pending | Confirm normalization artifact explains inversion |

---

## 12. Phase 8 Diagnostic Suite

### 12.1 Loss Normalization Audit

The write sparsity penalty in [`losses.py`](src/training/losses.py) is computed as:

```python
gate_mean = info["write_gate_mean"]          # = gate_soft.mean()  over [B, N]
loss_write_sparse = lambda * gate_mean
```

Where `gate_soft.mean()` divides by `B × N`. The denominator scales with N,
making the **effective per-slot cost inversely proportional to N**:

| N | λ | Effective cost per slot written |
|:---:|:---:|:---:|
| 3 | 0.001 | λ/3 = **0.000333** |
| 8 | 0.001 | λ/8 = **0.000125** |

**N=8 pays 2.67× less per slot than N=3 at the same λ.** This is a normalization
artifact that directly and mechanically explains the N=8 > N=3 write inversion.
The nominal per-slot discounts at each tested lambda:

| λ | N=3 cost/slot | N=8 cost/slot | N=8 discount |
|:---:|:---:|:---:|:---:|
| 0.001 | 0.000333 | 0.000125 | **2.67×** |
| 0.002 | 0.000667 | 0.000250 | **2.67×** |
| 0.005 | 0.001667 | 0.000625 | **2.67×** |

The 2.67× ratio is constant across all lambda values (it's N=8/N=3 = 8/3), which
explains why the inversion is stable across the sweep — it is baked into the loss
normalization, not discovered empirically.

> [!IMPORTANT]
> This does **not** invalidate the selectivity finding: all 6 runs still achieve
> genuine selective writing (mean_slots < K, entropy > 0.005). What it invalidates
> is the **comparison** of slot counts between N=3 and N=8 as a test of the
> capacity-starvation hypothesis. The two models are running under different
> effective cost regimes — the comparison is confounded.

**Fix**: Replace `gate_soft.mean()` with `gate_soft.sum(dim=-1).mean()` in the
externaliser. This makes the penalty scale with total slots written (absolute),
not fraction written, giving N=3 and N=8 equal cost per slot. A matched-per-slot
rerun of at least one N=3 cell is needed to separate the artifact from genuine
architectural effects.

---

### 12.2 Zero-Write Ablation Results

Script: `scripts/ablate_zero_write.py`

Three evaluation conditions on the same checkpoint:
- **NORMAL**: learned gate runs as trained
- **ZERO_WRITE**: `gate_net` replaced with constant −∞ output (no writes possible)
- **CORRUPT**: gate fires normally, written records zeroed before reads (Phase-6-style)

#### 12.2.1 N=3, λ=0.001

| Condition | Overall Acc | Turn 3 | Turn 4 | Turn 5 | Turn 6 | Turn 7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| NORMAL | **93.86%** | 92.40% | 90.20% | 87.40% | 91.00% | 97.40% |
| ZERO_WRITE | 72.97% | 52.80% | 67.40% | 62.20% | 71.80% | 94.00% |
| CORRUPT | 73.29% | 54.80% | 67.20% | 65.40% | 71.80% | 94.00% |
| Δ ZERO_WRITE | **−20.89pp** | −39.60pp | −22.80pp | −25.20pp | −19.20pp | −3.40pp |
| Δ CORRUPT | **−20.57pp** | −37.60pp | −23.00pp | −22.00pp | −19.20pp | −3.40pp |

**Verdict**: Writes are causally necessary (−20.89pp). CORRUPT ≈ ZERO_WRITE
(difference: 0.32pp) → the **content** of written records is what the reader
uses, not just the mask-guided attention pattern. Turn 7 is largely unaffected
(−3.40pp) — consistent with minimal writes at the final turn across all lambda
values (late-sequence suppression confirmed causally).

#### 12.2.2 N=8, λ=0.001

| Condition | Overall Acc | Turn 1 | Turn 3 | Turn 6 | Turn 7 |
|---|:---:|:---:|:---:|:---:|:---:|
| NORMAL | **94.34%** | 100.00% | 97.80% | 86.80% | 96.40% |
| ZERO_WRITE | 40.37% | 40.20% | 31.80% | 28.20% | 20.60% |
| CORRUPT | 40.43% | 40.20% | 31.80% | 28.20% | 22.20% |
| Δ ZERO_WRITE | **−53.97pp** | −59.80pp | −66.00pp | −58.60pp | −75.80pp |
| Δ CORRUPT | **−53.91pp** | −59.80pp | −66.00pp | −58.60pp | −74.20pp |

**Verdict**: N=8 is dramatically more workspace-dependent. Removing writes drops
accuracy by 53.97pp — to near-random-guess levels on several turns. CORRUPT ≈
ZERO_WRITE (0.06pp gap) → record content is completely causally active; the
reader gets nothing useful from corrupted records.

#### 12.2.3 Cross-Model Comparison

| | N=3 λ=0.001 | N=8 λ=0.001 |
|---|:---:|:---:|
| Workspace dependency (Δ zero-write) | −20.89pp | **−53.97pp** |
| Content dependency (Δ corrupt) | −20.57pp | **−53.91pp** |
| CORRUPT − ZERO_WRITE | +0.32pp | +0.06pp |
| Mask-only effect | negligible | negligible |

The N=8 model is **2.58× more workspace-dependent** than N=3 (53.97pp vs 20.89pp
drop). Combined with the normalization artifact (N=8 runs at 2.67× lower cost per
slot), this creates a consistent picture: N=8 has learned a qualitatively different,
workspace-heavy strategy *partly because it was incentivised to do so by the
cost structure*. Whether this strategy is architectural or cost-driven cannot be
determined until the matched-per-slot rerun (P8-E) is complete.

---

### 12.3 Within-Turn Gate Variance Analysis

Script: `scripts/analyze_gate_variance.py`

For each turn, across 500 test examples, measures: mean and std dev of gate fraction,
coefficient of variation (CV), and Spearman r vs entity count.

#### 12.3.1 N=3, λ=0.001

| Turn | n | Mean | Std | CV | r_entity |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 500 | 1.000 | 0.000 | 0.000 | (degenerate) |
| 2 | 500 | 0.973 | 0.123 | 0.126 | 0.525 |
| 3 | 500 | 0.683 | 0.313 | 0.459 | 0.010 |
| 4 | 500 | 0.519 | 0.171 | 0.330 | 0.015 |
| 5 | 500 | 0.756 | 0.237 | 0.313 | 0.042 |
| 6 | 500 | 0.334 | 0.177 | 0.530 | 0.395 |
| 7 | 500 | 0.303 | 0.331 | 1.091 | 0.026 |

#### 12.3.2 N=8, λ=0.001

| Turn | n | Mean | Std | CV | r_entity |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 500 | 1.000 | 0.000 | 0.000 | (degenerate) |
| 2 | 500 | 1.000 | 0.000 | 0.000 | (degenerate) |
| 3 | 500 | 0.273 | 0.343 | **1.257** | 0.036 |
| 4 | 500 | 0.153 | 0.072 | 0.472 | 0.165 |
| 5 | 500 | 0.289 | 0.310 | **1.073** | 0.151 |
| 6 | 500 | 0.132 | 0.094 | 0.712 | 0.571 |
| 7 | 500 | 0.243 | 0.298 | **1.226** | 0.137 |

#### 12.3.3 Findings

**1. Both gates are content-selective.** Mean within-turn std > 0.15 for both models
(well above the positional schedule threshold). For N=8, CV > 1.0 at Turns 3, 5, 7
— the standard deviation exceeds the mean, meaning examples span a bimodal
distribution (either write or don't write, decided per-content).

**2. Turn 1–2 saturation** is uninformative. Both models write to all examples at
Turns 1–2 (std = 0), so no content discrimination occurs. This is likely because
initial context is always worth externalising; the gate has not yet seen enough
sequence context to be selective.

**3. Spearman r values are unreliable.** The `r_turns_remaining` column exactly
equals `r_entity` in every row — a guaranteed artifact when `turns_remaining` is
constant within a turn (a scalar identical for all examples), making Spearman rank
degenerate. The `r_entity` values for Turn 2 (0.525) should also be treated with
caution: when std is near-zero, small numerical differences dominate rank ordering.
The reliable signal is Turns 3–7 for N=3 and Turns 3–7 for N=8, where entity-count
correlation is low (0.01–0.57) but the pattern is not interpretable without
controlling for batch-ordering effects.

**4. Entity count has low explanatory power.** Where r_entity is meaningful
(Turns 3–7), values are mostly near 0 (0.01–0.17) for both models, with one
exception: N=8 Turn 6 r=0.571. This suggests entity count is not the primary
driver of write decisions — the gate is responding to something else (query type,
task state, or internal representation quality). The nature of that signal requires
a richer intervention, e.g., stratified analysis by query type.

**5. N=8 has higher CV at the same turns.** At Turn 3, N=8 CV=1.257 vs N=3
CV=0.459. This aligns with N=8's write decisions being more binary (write/no-write
sharp threshold) while N=3's gate is smoother in its fraction. Consistent with N=8
having a lower-entropy gate overall (from §9.4).

---

## 13. Revised Theoretical Standing

### 13.1 What Can Now Be Claimed (Strengthened)

1. ✅ **Writes are causally necessary and content-active.** Zero-write ablation
   drops N=3 by 20.89pp and N=8 by 53.97pp. CORRUPT ≈ ZERO_WRITE confirms the
   reader uses record content, not just mask-guided attention.

2. ✅ **The learned gate is genuinely content-selective**, not a positional schedule.
   HIGH within-turn variance (mean std > 0.15, CV > 1.0 at several turns) across
   both models.

3. ✅ **The accuracy advantage over Top-K is real and causal.** The large zero-write
   drop (20–54pp) confirms the workspace contributes substantially; the Top-K
   baseline that writes *wrong records* (forced rather than selective) pays a
   genuine functional cost.

4. ✅ **The selectivity window exists and spans λ ∈ {0.001, 0.002, 0.005}** for
   both N=3 and N=8 under the current (fractional) loss normalization.

### 13.2 What Is Now Retracted or Qualified

1. ❌ **The N=8 > N=3 write inversion is not a genuine architectural finding.**
   The loss normalization gives N=8 a 2.67× per-slot cost discount at every tested
   lambda. This mechanically explains why N=8 writes more under the same nominal λ.
   The precision-store / aggressive-compression interpretation must be suspended
   pending the matched-per-slot rerun (P8-E).

2. ⚠️ **The Top-K advantage numbers partially reflect training-time cost asymmetry.**
   N=8's large advantages (+17–23pp) over Top-K partly reflect that the Top-K
   *baseline was not retrained* under the same cost structure — it uses the learned
   model's weights with `learned_gate=False`. The advantage is real in the sense
   that forced Top-K uses the wrong records; but the magnitude is inflated by the
   normalization artifact if N=8 has learned a workspace-heavy strategy *because*
   it was cheap to do so.

3. ⚠️ **Spearman r(gate, entity_count) results are unreliable** due to degenerate
   `turns_remaining` constant and possibly low entity-count variance per turn.
   Content-selectivity is confirmed by std/CV metrics; the *driver* of selectivity
   (entity count, query type, or other) remains uncharacterised.

---

## 14. Next Steps (P8-D, P8-E)

### P8-D: Second Seed Runs

Before committing to the matched-per-slot cost fix, confirm that the current
selectivity finding is seed-robust on the two most informative checkpoints:

```powershell
# Configs to create: same as lam001/lam005 but with seed: 43
# (AI to generate YAMLs on request)
python scripts/train_sequential.py --config configs/experiments/fair_hard_n8_lam001_seed43.yaml
python scripts/train_sequential.py --config configs/experiments/fair_hard_n3_lam005_seed43.yaml
```

### P8-E: Matched Per-Slot Cost (N=3 rerun with corrected normalization)

Patch [`src/models/externaliser.py`](src/models/externaliser.py) line 321:

```python
# Current (fractional — penalises N=3 2.67× more per slot):
"write_gate_mean": gate_soft.mean(),

# Fixed (absolute — equal cost per slot regardless of N):
"write_gate_mean": gate_soft.sum(dim=-1).mean(),
```

Then retrain N=3 and N=8 at λ=0.001 with the fix. If the inversion disappears
(N=3 writes >= N=8), the artifact hypothesis is confirmed. If it persists, the
precision-store interpretation is genuine.