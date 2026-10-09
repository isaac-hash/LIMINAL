"""Three-Way Demand-Sensitivity Analysis with Bootstrap CIs.

Fixes the design flaw in analyze_demand_sensitivity.py (§18.3 of the report)
where Group 0 ("not mattered") mixed two distinct populations:
  (a) both-correct: NORMAL correct AND ZERO_WRITE also correct (true counterfactual)
  (b) normal-wrong: NORMAL itself wrong (uninformative about workspace demand)

This script computes the valid three-way split and compares gate fractions only
between helped vs both-correct groups, with bootstrap confidence intervals.

Three groups (defined per-example, sequence-level):
  helped       = NORMAL correct at ALL valid turns, ZERO_WRITE wrong at ANY valid turn
  both-correct = NORMAL correct at ALL valid turns, ZERO_WRITE correct at ALL valid turns
  normal-wrong = NORMAL wrong at ANY valid turn

Within each group, per-turn gate fraction is extracted from the NORMAL forward pass.

The demand-selectivity question is:
  E[gate_frac | helped] > E[gate_frac | both-correct] at each turn?

Bootstrap CIs (2000 resamples) are computed on the group difference at each turn.
A difference is considered significant if the 95% CI excludes 0.

Output: JSON report + per-example group tags + 2-panel bar chart.

Usage:
    python scripts/analyze_three_way_demand.py \\
        --config  configs/experiments/fair_hard_n3_lam001.yaml \\
        --checkpoint results/fair_hard_n3_lam001/best.pt \\
        --output-dir results/three_way_demand/fair_hard_n3_lam001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.device import print_hardware_info
from src.utils.config import load_config, set_seed
from src.utils.checkpoint import load_checkpoint, resolve_checkpoint_path
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary
from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
from src.models.sequential_model import SequentialReasoningModel


# ---------------------------------------------------------------------------
# Hook: force gate closed
# ---------------------------------------------------------------------------

def _find_controller(model: SequentialReasoningModel):
    try:
        return model.base_model.workspace.write_controller
    except AttributeError:
        return None


class _ZeroWriteHook:
    """Replace gate_net with one that always returns -1e9, suppressing all writes."""

    def __init__(self, model):
        self._ctrl = _find_controller(model)

    def __enter__(self):
        if self._ctrl is None:
            raise RuntimeError("LearnedWriteController not found in model.")
        self._orig_gate_net = self._ctrl.gate_net

        class _NegInfNet(nn.Module):
            def forward(self, x):
                shape = (x.shape[0], x.shape[1], 1) if x.dim() == 3 else (x.shape[0], 1)
                return torch.full(shape, fill_value=-1e9, device=x.device, dtype=x.dtype)

        self._ctrl.gate_net = _NegInfNet()
        return self

    def __exit__(self, *_):
        self._ctrl.gate_net = self._orig_gate_net


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, rng=None):
    """Bootstrap 95% CI for mean(a) - mean(b). Returns (lower, upper, observed_diff)."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    obs = float(a.mean() - b.mean())
    diffs = np.array([
        rng.choice(a, size=len(a), replace=True).mean() -
        rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return lo, hi, obs


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def run_forward(model, loader, device, num_batches=None):
    """Collect per-example per-turn correctness and gate fractions."""
    model.eval()
    results = []
    global_idx = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if num_batches is not None and b_idx >= num_batches:
                break

            facts_seq = batch["facts"].to(device)
            fact_mask  = batch["fact_mask"].to(device)
            turn_mask  = batch["turn_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            B, max_turns = labels.shape[0], labels.shape[1]
            all_logits, all_infos = model(facts_seq, fact_mask, turn_mask)

            preds_all   = all_logits.argmax(dim=-1)
            correct_all = (preds_all == labels)

            gate_frac_all = torch.full((B, max_turns), float("nan"))
            for t in range(max_turns):
                info_t = all_infos[t]
                gate_hard = None
                traj = info_t.get("write_gate_trajectory", [])
                if traj:
                    gates = torch.stack(
                        [g.cpu() for g in traj if isinstance(g, Tensor)], dim=0
                    )
                    gate_hard = (gates.sum(0) > 0).float()
                elif info_t.get("write_gate_hard") is not None:
                    gate_hard = info_t["write_gate_hard"].cpu()

                if gate_hard is not None:
                    gate_frac_all[:, t] = gate_hard.mean(dim=1)

            for b in range(B):
                valid = (turn_mask[b] > 0).nonzero(as_tuple=True)[0].tolist()
                results.append({
                    "seq_idx": global_idx,
                    "turn_correct": [bool(correct_all[b, t].item()) for t in range(max_turns)],
                    "turn_gate_frac": [float(gate_frac_all[b, t].item()) for t in range(max_turns)],
                    "valid_turns": valid,
                })
                global_idx += 1

    return results


# ---------------------------------------------------------------------------
# Three-way classification
# ---------------------------------------------------------------------------

GROUP_HELPED       = "helped"         # NORMAL correct all turns, ZERO wrong at >= 1 turn
GROUP_BOTH_CORRECT = "both-correct"   # NORMAL correct all turns, ZERO correct all turns
GROUP_NORMAL_WRONG = "normal-wrong"   # NORMAL wrong at >= 1 turn

def classify_examples(normal_results, zero_results):
    """Assign each example to one of three groups."""
    groups = []
    for i in range(len(normal_results)):
        nr = normal_results[i]
        zr = zero_results[i]
        valid = nr["valid_turns"]

        normal_all_correct = all(nr["turn_correct"][t] for t in valid)

        if not normal_all_correct:
            groups.append(GROUP_NORMAL_WRONG)
            continue

        # NORMAL is correct on all valid turns
        zero_any_wrong = any(not zr["turn_correct"][t] for t in valid)
        if zero_any_wrong:
            groups.append(GROUP_HELPED)
        else:
            groups.append(GROUP_BOTH_CORRECT)

    return groups


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(normal_results, zero_results, groups, max_turns, n_boot=2000):
    n = len(normal_results)
    rng = np.random.default_rng(42)

    counts = {GROUP_HELPED: 0, GROUP_BOTH_CORRECT: 0, GROUP_NORMAL_WRONG: 0}
    for g in groups:
        counts[g] += 1

    print(f"\n  Group counts:")
    for g, c in counts.items():
        print(f"    {g:>15} : {c:4d} ({c/n*100:.1f}%)")

    # Per-turn, per-group gate fracs (from NORMAL pass)
    turn_gf: dict[int, dict[str, list]] = {
        t: {GROUP_HELPED: [], GROUP_BOTH_CORRECT: [], GROUP_NORMAL_WRONG: []}
        for t in range(max_turns)
    }

    for i in range(n):
        g = groups[i]
        for t in normal_results[i]["valid_turns"]:
            gf = normal_results[i]["turn_gate_frac"][t]
            if not np.isnan(gf):
                turn_gf[t][g].append(gf)

    turn_stats = {}
    for t in range(max_turns):
        helped      = np.array(turn_gf[t][GROUP_HELPED])
        both_ok     = np.array(turn_gf[t][GROUP_BOTH_CORRECT])
        norm_wrong  = np.array(turn_gf[t][GROUP_NORMAL_WRONG])

        if len(helped) + len(both_ok) == 0:
            continue

        m_helped   = float(helped.mean())   if len(helped)    > 0 else float("nan")
        m_both_ok  = float(both_ok.mean())  if len(both_ok)   > 0 else float("nan")
        m_nw       = float(norm_wrong.mean()) if len(norm_wrong) > 0 else float("nan")

        ci_lo, ci_hi, obs_diff = bootstrap_mean_diff(helped, both_ok, n_boot=n_boot, rng=rng)
        significant = (not np.isnan(ci_lo)) and (ci_lo > 0)

        turn_stats[t] = {
            "turn": t + 1,
            "n_helped":       len(helped),
            "n_both_correct": len(both_ok),
            "n_normal_wrong": len(norm_wrong),
            "mean_helped":       round(m_helped, 4),
            "mean_both_correct": round(m_both_ok, 4),
            "mean_normal_wrong": round(m_nw, 4),
            "delta_helped_minus_both_correct": round(obs_diff, 4) if not np.isnan(obs_diff) else None,
            "ci95_lo": round(ci_lo, 4) if not np.isnan(ci_lo) else None,
            "ci95_hi": round(ci_hi, 4) if not np.isnan(ci_hi) else None,
            "significant_positive": significant,
        }

    # Verdict
    sig_turns = sum(1 for s in turn_stats.values() if s["significant_positive"])
    total_turns = len(turn_stats)

    if sig_turns >= total_turns * 0.6:
        verdict = (
            f"DEMAND-SELECTIVE: gate_frac[helped] > gate_frac[both-correct] "
            f"at {sig_turns}/{total_turns} turns (95% CI excludes 0)."
        )
    elif sig_turns >= 1:
        verdict = (
            f"WEAKLY DEMAND-SELECTIVE: significant positive lift at "
            f"{sig_turns}/{total_turns} turns only."
        )
    else:
        verdict = (
            "NOT DEMAND-SELECTIVE: no turn shows a significant positive lift "
            "in gate_frac for helped vs both-correct. Gate may be positional."
        )

    return {
        "n_examples":    n,
        "group_counts":  counts,
        "turn_stats":    {str(t + 1): v for t, v in turn_stats.items()},
        "n_sig_turns":   sig_turns,
        "n_total_turns": total_turns,
        "verdict":       verdict,
        "group_labels":  groups,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(analysis, out_path: Path):
    ts = analysis["turn_stats"]
    turns = sorted(ts.keys(), key=int)
    labels = [f"T{t}" for t in turns]

    m_helped = [ts[t]["mean_helped"]        for t in turns]
    m_both   = [ts[t]["mean_both_correct"]  for t in turns]
    delta    = [ts[t]["delta_helped_minus_both_correct"] for t in turns]
    ci_lo    = [ts[t]["ci95_lo"] for t in turns]
    ci_hi    = [ts[t]["ci95_hi"] for t in turns]
    sig      = [ts[t]["significant_positive"] for t in turns]

    x = np.arange(len(labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Three-Way Demand Sensitivity: gate_frac by group", fontsize=13)

    # Left: grouped bar
    ax = axes[0]
    ax.bar(x - w/2, m_helped, w, label="helped",       color="#e07b54", alpha=0.85)
    ax.bar(x + w/2, m_both,   w, label="both-correct", color="#5b7fa6", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean gate_fraction"); ax.set_title("Gate Fraction by Group")
    ax.legend(); ax.set_ylim(0, 1.1)

    # Right: delta + CIs
    ax2 = axes[1]
    bar_colors = ["#2e7d32" if s else "#c62828" for s in sig]
    delta_vals = [d if d is not None else 0.0 for d in delta]
    ax2.bar(x, delta_vals, color=bar_colors, alpha=0.85)

    # Error bars for CI
    valid_mask = [ci_lo[i] is not None and ci_hi[i] is not None for i in range(len(turns))]
    x_valid  = [x[i] for i in range(len(turns)) if valid_mask[i]]
    d_valid  = [delta_vals[i] for i in range(len(turns)) if valid_mask[i]]
    lo_valid = [delta_vals[i] - ci_lo[i] for i in range(len(turns)) if valid_mask[i]]
    hi_valid = [ci_hi[i] - delta_vals[i] for i in range(len(turns)) if valid_mask[i]]
    if x_valid:
        ax2.errorbar(x_valid, d_valid, yerr=[lo_valid, hi_valid],
                     fmt="none", color="black", capsize=4, linewidth=1.5)

    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Delta (helped - both-correct)")
    ax2.set_title("Demand Lift (95% Bootstrap CI)")

    # Legend for significance
    from matplotlib.patches import Patch
    legend_elems = [Patch(fc="#2e7d32", label="sig. positive"),
                    Patch(fc="#c62828", label="not sig.")]
    ax2.legend(handles=legend_elems)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--num-batches",  type=int, default=None)
    p.add_argument("--n-bootstrap",  type=int, default=2000)
    p.add_argument("--seed",         type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = print_hardware_info()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    config = load_config(Path("configs/base.yaml"), overrides=overrides)
    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    generator = ArithmeticGenerator(config=config.data, seed=seed)
    splits    = generator.generate_dataset()
    vocab     = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"] + splits["test"])

    loader = DataLoader(
        SequenceReasoningDataset(splits["test"], vocab),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )
    max_turns = config.data.sequence_turns
    print(f"Test set: {len(splits['test'])} sequences | max_turns={max_turns}")

    model = SequentialReasoningModel(
        config=config.model, vocab=vocab,
        activity_config=config.activity,
        resolution_config=config.resolution,
        persistence_config=config.persistence,
        external_config=config.external,
    ).to(device)

    ckpt = resolve_checkpoint_path(args.checkpoint, None)
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    load_checkpoint(ckpt, model=model, device=device)

    width = 72
    print("\n" + "=" * width)
    print("  Three-Way Demand-Sensitivity Analysis")
    print("=" * width)

    print("\n[1/2] NORMAL forward pass...")
    normal_results = run_forward(model, loader, device, args.num_batches)

    print("[2/2] ZERO_WRITE forward pass...")
    with _ZeroWriteHook(model):
        zero_results = run_forward(model, loader, device, args.num_batches)

    print("\nClassifying examples into three groups...")
    groups = classify_examples(normal_results, zero_results)

    print(f"\nRunning bootstrap CIs (n={args.n_bootstrap})...")
    analysis = analyse(normal_results, zero_results, groups, max_turns, n_boot=args.n_bootstrap)

    # Print turn table
    print(f"\n{'T':>3} {'N_hlp':>6} {'N_ok':>6} {'N_nw':>6} "
          f"{'hlp':>7} {'ok':>7} {'delta':>8} {'95% CI':>18} {'sig':>5}")
    print("-" * 72)
    for tstr in sorted(analysis["turn_stats"].keys(), key=int):
        s = analysis["turn_stats"][tstr]
        d = s["delta_helped_minus_both_correct"]
        lo, hi = s["ci95_lo"], s["ci95_hi"]
        d_str  = f"{d:+.4f}" if d is not None else "     N/A"
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "          N/A"
        sig_str = " *" if s["significant_positive"] else ""
        print(f"  {tstr:>2} {s['n_helped']:>6} {s['n_both_correct']:>6} "
              f"{s['n_normal_wrong']:>6} "
              f"{s['mean_helped']:>7.4f} {s['mean_both_correct']:>7.4f} "
              f"{d_str} {ci_str:>18}{sig_str}")

    print(f"\n  Significant turns: {analysis['n_sig_turns']}/{analysis['n_total_turns']}")
    print(f"\n  Verdict: {analysis['verdict']}")
    print("=" * width)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in analysis.items() if k != "group_labels"}
    summary["config"] = args.config
    summary["checkpoint"] = args.checkpoint
    summary["n_bootstrap"] = args.n_bootstrap

    json_out = out_dir / "three_way_demand.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved report : {json_out}")

    tags_out = out_dir / "group_labels.json"
    with open(tags_out, "w", encoding="utf-8") as f:
        json.dump(analysis["group_labels"], f)
    print(f"Saved labels : {tags_out}")

    try:
        plot_results(analysis, out_dir / "three_way_demand.png")
    except Exception as e:
        print(f"  [Warning] Plot failed: {e}")


if __name__ == "__main__":
    main()
