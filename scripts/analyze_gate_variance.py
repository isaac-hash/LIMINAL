"""Within-Turn Gate Variance Analysis.

Tests whether the learned write gate varies on *content* or only on *position*.

For each turn t, across all test examples, this script records:
  - gate_fraction_t   (per example: fraction of slots written at turn t)
  - entity_count      (number of active entities in the example)
  - query_overlap     (fraction of written entities that are queried at a later turn)

Then computes:
  - Std dev of gate_fraction across examples at each fixed turn  (within-turn variance)
    A positional schedule has ~0 within-turn variance.
    A content-selective gate has high within-turn variance.

  - Spearman correlation of gate_fraction vs entity_count (per turn)
    If the gate responds to capacity pressure, r > 0 for capacity-starved N.

  - Spearman correlation of gate_fraction vs query_overlap (per turn)
    If the gate is forward-looking, r > 0 (write more when you will need it later).

Output: JSON report + bar charts of within-turn std dev.

Usage:
    python scripts/analyze_gate_variance.py \
        --config  configs/experiments/fair_hard_n3_lam001.yaml \
        --checkpoint results/fair_hard_n3_lam001/best.pt \
        --output-dir results/variance/fair_hard_n3_lam001
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
# Spearman correlation (no scipy dependency)
# ---------------------------------------------------------------------------

def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    if len(x) < 3:
        return float("nan")
    def rank(a):
        idx = np.argsort(a)
        r = np.empty_like(idx, dtype=float)
        r[idx] = np.arange(len(a))
        return r
    rx, ry = rank(x.astype(float)), rank(y.astype(float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_gate_data(
    model: SequentialReasoningModel,
    loader: DataLoader,
    device: torch.device,
    num_batches: int | None = None,
) -> dict[int, dict]:
    """Return per-turn lists of (gate_fraction, entity_count, ...)."""
    model.eval()

    # per turn: lists of scalar values across examples
    turn_gate_fracs:  dict[int, list] = defaultdict(list)
    turn_entity_cnts: dict[int, list] = defaultdict(list)
    # query_overlap requires knowing which entities are queried later;
    # we approximate with the number of turns remaining (later-turn query proxy)
    turn_turns_remaining: dict[int, list] = defaultdict(list)

    # Also collect per-example gate vectors for std computation
    turn_gate_vectors: dict[int, list] = defaultdict(list)   # list of [N] tensors

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if num_batches is not None and b_idx >= num_batches:
                break

            facts_seq  = batch["facts"].to(device)     # [B, T, F, 5]
            fact_mask  = batch["fact_mask"].to(device)  # [B, T, F]
            turn_mask  = batch["turn_mask"].to(device)  # [B, T]

            B, max_turns, max_facts, _ = facts_seq.shape

            # Entity count per (batch, turn): count non-zero fact rows
            # fact_mask: 1 where a fact exists
            entity_counts = fact_mask.sum(dim=-1).float()  # [B, T]

            all_logits, all_infos = model(facts_seq, fact_mask, turn_mask)

            for t in range(max_turns):
                valid = (turn_mask[:, t] > 0).nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue

                info_t = all_infos[t]
                turns_remaining = max_turns - 1 - t   # scalar int

                # Extract hard gate decisions
                gate_hard = None
                traj = info_t.get("write_gate_trajectory", [])
                if traj:
                    # sum across reasoning steps: any-write per slot
                    gates = torch.stack([g[valid].cpu() for g in traj if isinstance(g, Tensor)], dim=0)
                    gate_hard = (gates.sum(0) > 0).float()   # [valid, N]
                elif info_t.get("write_gate_hard") is not None:
                    gate_hard = info_t["write_gate_hard"][valid].cpu()   # [valid, N]

                if gate_hard is None:
                    continue

                N = gate_hard.shape[1]
                gate_frac = gate_hard.mean(dim=1).numpy()      # [valid] fraction of slots written per example
                ent_cnt   = entity_counts[valid, t].cpu().numpy()  # [valid]

                for i, vi in enumerate(valid):
                    turn_gate_fracs[t].append(float(gate_frac[i]))
                    turn_entity_cnts[t].append(float(ent_cnt[i]))
                    turn_turns_remaining[t].append(float(turns_remaining))
                    turn_gate_vectors[t].append(gate_hard[i].numpy())  # [N]

    return {
        "gate_fracs":        dict(turn_gate_fracs),
        "entity_cnts":       dict(turn_entity_cnts),
        "turns_remaining":   dict(turn_turns_remaining),
        "gate_vectors":      dict(turn_gate_vectors),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(data: dict, config) -> dict:
    gate_fracs   = data["gate_fracs"]
    entity_cnts  = data["entity_cnts"]
    turns_rem    = data["turns_remaining"]
    gate_vectors = data["gate_vectors"]

    N = config.model.latent_slots
    results_by_turn = {}

    for t in sorted(gate_fracs.keys()):
        gf  = np.array(gate_fracs[t])
        ec  = np.array(entity_cnts[t])
        tr  = np.array(turns_rem[t])
        gv  = np.array(gate_vectors[t])  # [n_examples, N]

        n = len(gf)
        mean_frac  = float(gf.mean())
        std_frac   = float(gf.std())
        cv_frac    = std_frac / (mean_frac + 1e-8)   # coefficient of variation

        # Slot-level agreement: do the same slots tend to fire across examples?
        # High slot-agreement -> positional/structural schedule
        # Low slot-agreement  -> content-driven (different slots for different inputs)
        if n > 1:
            slot_fire_rate = gv.mean(axis=0)  # [N] fraction of examples that fire each slot
            slot_agreement = float(slot_fire_rate.std())  # high = uneven = some slots always fire
        else:
            slot_agreement = float("nan")

        r_entity  = spearman_r(gf, ec)
        r_turns   = spearman_r(gf, tr)

        results_by_turn[t] = {
            "turn": t,
            "n_examples":       n,
            "mean_gate_frac":   round(mean_frac, 4),
            "std_gate_frac":    round(std_frac, 4),
            "cv_gate_frac":     round(cv_frac, 4),
            "slot_fire_std":    round(slot_agreement, 4),
            "spearman_r_entity_count":  round(r_entity, 4),
            "spearman_r_turns_remaining": round(r_turns, 4),
        }

    return results_by_turn


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_within_turn_variance(results_by_turn: dict, output_dir: Path):
    turns  = [t+1 for t in sorted(results_by_turn)]
    means  = [results_by_turn[t-1]["mean_gate_frac"] for t in turns]
    stds   = [results_by_turn[t-1]["std_gate_frac"]  for t in turns]
    r_ent  = [results_by_turn[t-1]["spearman_r_entity_count"] for t in turns]
    r_trn  = [results_by_turn[t-1]["spearman_r_turns_remaining"] for t in turns]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    # Panel 1: mean ± std gate fraction
    ax = axes[0]
    ax.bar(turns, means, yerr=stds, color="#4C9BE8", edgecolor="#1F4E79", alpha=0.85,
           capsize=5, width=0.55)
    ax.set_title("Gate Fraction by Turn\n(mean ± std across examples)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Turn"); ax.set_ylabel("Fraction written")
    ax.set_ylim(0, min(1.1, max(means) + max(stds) * 1.5 + 0.1))
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 2: Spearman r vs entity count
    colors2 = ["#E84C4C" if v < 0 else "#2E7D32" for v in r_ent]
    ax = axes[1]
    ax.bar(turns, r_ent, color=colors2, edgecolor="#333", alpha=0.85, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman r(gate_frac, entity_count)\nper turn", fontsize=10, fontweight="bold")
    ax.set_xlabel("Turn"); ax.set_ylabel("Spearman r")
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel 3: Spearman r vs turns remaining
    colors3 = ["#E84C4C" if v < 0 else "#7B2D8B" for v in r_trn]
    ax = axes[2]
    ax.bar(turns, r_trn, color=colors3, edgecolor="#333", alpha=0.85, width=0.55)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Spearman r(gate_frac, turns_remaining)\nper turn", fontsize=10, fontweight="bold")
    ax.set_xlabel("Turn"); ax.set_ylabel("Spearman r")
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out = output_dir / "gate_variance_analysis.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")
    return out


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
    splits = generator.generate_dataset()
    vocab  = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"] + splits["test"])

    loader = DataLoader(
        SequenceReasoningDataset(splits["test"], vocab),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )
    print(f"Test set: {len(splits['test'])} sequences.")

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

    print("\n" + "=" * 60)
    print("  Within-Turn Gate Variance Analysis")
    print("=" * 60)
    print("Collecting gate decisions across test set...")

    data = collect_gate_data(model, loader, device, args.num_batches)
    results = analyse(data, config)

    print(f"\n{'Turn':>5} {'N':>6} {'Mean':>7} {'Std':>7} {'CV':>7} "
          f"{'r_entity':>10} {'r_turns_rem':>12}")
    print("-" * 65)
    for t in sorted(results):
        r = results[t]
        print(f"  {t+1:>3}  {r['n_examples']:>6}  {r['mean_gate_frac']:>6.3f}  "
              f"{r['std_gate_frac']:>6.3f}  {r['cv_gate_frac']:>6.3f}  "
              f"{r['spearman_r_entity_count']:>9.3f}  {r['spearman_r_turns_remaining']:>11.3f}")

    print()
    # Interpretation hints
    all_stds = [results[t]["std_gate_frac"] for t in results]
    mean_std = np.mean(all_stds)
    if mean_std < 0.05:
        print("  Interpretation: LOW within-turn variance (mean std < 0.05) -- gate "
              "behaves like a positional schedule, not content-selective.")
    elif mean_std < 0.15:
        print("  Interpretation: MODERATE within-turn variance -- gate has some content "
              "sensitivity but may also be partially positional.")
    else:
        print("  Interpretation: HIGH within-turn variance (mean std > 0.15) -- gate is "
              "content-selective: different examples trigger different write decisions.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_within_turn_variance(results, out_dir)

    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "latent_slots": config.model.latent_slots,
        "write_sparsity_lambda": getattr(config.external, "write_sparsity_lambda", None),
        "mean_within_turn_std": round(float(mean_std), 4),
        "per_turn": {str(t): v for t, v in results.items()},
    }
    out = out_dir / "gate_variance_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()