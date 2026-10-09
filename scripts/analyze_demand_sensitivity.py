"""Demand-Sensitivity Analysis (LM-Transition Condition 4).

Tests whether the write gate fires *more* on examples where writing actually
changes the prediction, i.e. where the workspace is demanded by the task.

Per-example tagging:
    write_mattered[i] = 1  if  NORMAL correct  AND  ZERO_WRITE wrong  at any turn

Per-turn comparison (at each turn t):
    mean gate_fraction  for  write_mattered == 1  vs  write_mattered == 0

Expected finding (demand-selective gate):
    E[gate_frac | write_mattered=1] > E[gate_frac | write_mattered=0]  at most turns.

A gate that is merely positional / structural will show no difference between groups.

Output: JSON report + bar chart of per-turn gate_frac by group.

Usage:
    python scripts/analyze_demand_sensitivity.py \\
        --config  configs/experiments/fair_hard_n3_lam001.yaml \\
        --checkpoint results/fair_hard_n3_lam001/best.pt \\
        --output-dir results/demand_sensitivity/fair_hard_n3_lam001
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
# Hook: force gate closed (reused from ablate_zero_write.py)
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
        orig = self._ctrl.gate_net
        self._orig_gate_net = orig

        class _NegInfNet(nn.Module):
            def forward(self, x):
                shape = (x.shape[0], x.shape[1], 1) if x.dim() == 3 else (x.shape[0], 1)
                return torch.full(shape, fill_value=-1e9, device=x.device, dtype=x.dtype)

        self._ctrl.gate_net = _NegInfNet()
        return self

    def __exit__(self, *_):
        self._ctrl.gate_net = self._orig_gate_net


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def run_forward(model, loader, device, num_batches=None):
    """Return per-example predictions and gate fractions across turns.

    Returns
    -------
    results : list of dicts, one per example:
        {
            "seq_idx": int,
            "turn_correct": [bool ...],
            "turn_gate_frac": [float...],
            "valid_turns": [int ...]
        }
    """
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

            preds_all   = all_logits.argmax(dim=-1)     # [B, T]
            correct_all = (preds_all == labels)          # [B, T] bool

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
                valid_turns = (turn_mask[b] > 0).nonzero(as_tuple=True)[0].tolist()
                results.append({
                    "seq_idx": global_idx,
                    "turn_correct": [bool(correct_all[b, t].item()) for t in range(max_turns)],
                    "turn_gate_frac": [float(gate_frac_all[b, t].item()) for t in range(max_turns)],
                    "valid_turns": valid_turns,
                })
                global_idx += 1

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(normal_results, zero_results, max_turns):
    """Compute per-example write_mattered tag and per-turn group comparisons."""
    n_examples = len(normal_results)
    assert len(zero_results) == n_examples, "Example count mismatch between conditions."

    write_mattered = []
    for i in range(n_examples):
        nr = normal_results[i]
        zr = zero_results[i]
        mattered = any(
            nr["turn_correct"][t] and not zr["turn_correct"][t]
            for t in nr["valid_turns"]
        )
        write_mattered.append(mattered)

    n_mattered     = sum(write_mattered)
    n_not_mattered = n_examples - n_mattered
    print(f"\n  write_mattered=1 : {n_mattered} examples ({n_mattered/n_examples*100:.1f}%)")
    print(f"  write_mattered=0 : {n_not_mattered} examples ({n_not_mattered/n_examples*100:.1f}%)")

    turn_group_fracs: dict[int, dict[int, list]] = {
        t: {0: [], 1: []} for t in range(max_turns)
    }

    for i in range(n_examples):
        g = 1 if write_mattered[i] else 0
        for t in normal_results[i]["valid_turns"]:
            gf = normal_results[i]["turn_gate_frac"][t]
            if not np.isnan(gf):
                turn_group_fracs[t][g].append(gf)

    turn_stats = {}
    for t in range(max_turns):
        g0 = np.array(turn_group_fracs[t][0])
        g1 = np.array(turn_group_fracs[t][1])
        if len(g0) == 0 and len(g1) == 0:
            continue

        mean0 = float(g0.mean()) if len(g0) > 0 else float("nan")
        mean1 = float(g1.mean()) if len(g1) > 0 else float("nan")
        delta = mean1 - mean0 if not (np.isnan(mean0) or np.isnan(mean1)) else float("nan")

        turn_stats[t] = {
            "turn": t + 1,
            "n_not_mattered": len(g0),
            "n_mattered":     len(g1),
            "mean_gate_frac_not_mattered": round(mean0, 4),
            "mean_gate_frac_mattered":     round(mean1, 4),
            "delta_mattered_minus_not":    round(delta, 4) if not np.isnan(delta) else None,
        }

    deltas = [
        turn_stats[t]["delta_mattered_minus_not"]
        for t in turn_stats
        if turn_stats[t]["delta_mattered_minus_not"] is not None
    ]
    positive_turns = sum(1 for d in deltas if d > 0.02)
    total_turns    = len(deltas)

    if positive_turns >= total_turns * 0.6:
        verdict = (
            "DEMAND-SELECTIVE: gate fires more when writing changes the answer "
            f"({positive_turns}/{total_turns} turns show >2pp lift)."
        )
    elif positive_turns >= 1:
        verdict = (
            f"WEAKLY DEMAND-SELECTIVE: gate shows >2pp lift on "
            f"{positive_turns}/{total_turns} turns only."
        )
    else:
        verdict = (
            "NOT DEMAND-SELECTIVE: gate shows no consistent lift on write_mattered examples. "
            "Gate schedule may be positional or structural."
        )

    return {
        "n_examples":          n_examples,
        "n_mattered":          n_mattered,
        "n_not_mattered":      n_not_mattered,
        "pct_mattered":        round(n_mattered / n_examples * 100, 2),
        "turn_stats":          {str(t + 1): v for t, v in turn_stats.items()},
        "n_positive_turns":    positive_turns,
        "n_total_turns":       total_turns,
        "verdict":             verdict,
        "write_mattered_list": [int(w) for w in write_mattered],
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(analysis, out_path: Path):
    turn_stats = analysis["turn_stats"]
    turns_sorted = sorted(turn_stats.keys(), key=int)
    turns_labels = [f"Turn {t}" for t in turns_sorted]

    mean0 = [turn_stats[t]["mean_gate_frac_not_mattered"] for t in turns_sorted]
    mean1 = [turn_stats[t]["mean_gate_frac_mattered"]     for t in turns_sorted]
    delta = [turn_stats[t]["delta_mattered_minus_not"]    for t in turns_sorted]

    x = np.arange(len(turns_labels))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Demand-Sensitivity Analysis: Gate Fraction by Group", fontsize=13)

    ax = axes[0]
    ax.bar(x - w/2, mean0, w, label="write_mattered=0", color="#5b7fa6", alpha=0.85)
    ax.bar(x + w/2, mean1, w, label="write_mattered=1", color="#e07b54", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(turns_labels, rotation=30, ha="right")
    ax.set_ylabel("Mean gate_fraction")
    ax.set_title("Gate Fraction by Group (per turn)")
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax2 = axes[1]
    colors = ["#2e7d32" if d is not None and d > 0 else "#c62828" for d in delta]
    delta_vals = [d if d is not None else 0.0 for d in delta]
    ax2.bar(x, delta_vals, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(0.02, color="#2e7d32", linewidth=0.8, linestyle=":", label="+2pp threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(turns_labels, rotation=30, ha="right")
    ax2.set_ylabel("Delta gate_frac (mattered - not mattered)")
    ax2.set_title("Demand Lift per Turn")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--num-batches", type=int, default=None)
    p.add_argument("--seed",        type=int, default=None)
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
    print(f"Test set: {len(splits['test'])} sequences  |  max_turns={max_turns}")

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

    width = 70
    print("\n" + "=" * width)
    print("  Demand-Sensitivity Analysis (LM-Transition Condition 4)")
    print("=" * width)

    print("\n[1/2] Running NORMAL (learned gate) forward pass...")
    normal_results = run_forward(model, loader, device, args.num_batches)

    print("[2/2] Running ZERO_WRITE (gate forced closed) forward pass...")
    with _ZeroWriteHook(model):
        zero_results = run_forward(model, loader, device, args.num_batches)

    print("\nTagging examples and computing per-turn group statistics...")
    analysis = analyse(normal_results, zero_results, max_turns)

    print(f"\n{'Turn':>6}  {'N(0)':>6}  {'N(1)':>6}  "
          f"{'gfrac(0)':>10}  {'gfrac(1)':>10}  {'delta':>8}")
    print("-" * 58)
    for tstr in sorted(analysis["turn_stats"].keys(), key=int):
        s = analysis["turn_stats"][tstr]
        d = s["delta_mattered_minus_not"]
        d_str = f"{d:+.4f}" if d is not None else "    N/A"
        flag = " >" if d is not None and d > 0.02 else ""
        print(f"  {tstr:>4}  {s['n_not_mattered']:>6}  {s['n_mattered']:>6}  "
              f"{s['mean_gate_frac_not_mattered']:>10.4f}  "
              f"{s['mean_gate_frac_mattered']:>10.4f}  {d_str}{flag}")

    print(f"\n  Positive turns (>2pp lift): "
          f"{analysis['n_positive_turns']}/{analysis['n_total_turns']}")
    print(f"\n  Verdict: {analysis['verdict']}")
    print("=" * width)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in analysis.items() if k != "write_mattered_list"}
    summary["config"]     = args.config
    summary["checkpoint"] = args.checkpoint

    json_out = out_dir / "demand_sensitivity.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved report : {json_out}")

    tags_out = out_dir / "write_mattered_tags.json"
    with open(tags_out, "w", encoding="utf-8") as f:
        json.dump(analysis["write_mattered_list"], f)
    print(f"Saved tags   : {tags_out}")

    plot_path = out_dir / "demand_sensitivity.png"
    try:
        plot_results(analysis, plot_path)
    except Exception as e:
        print(f"  [Warning] Plot failed: {e}")


if __name__ == "__main__":
    main()
