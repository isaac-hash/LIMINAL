"""Per-Turn Workspace Ablation (Condition 2 gap).

Tests which turns' writes are causally downstream-useful, by ablating the workspace
at the entry to each turn t and measuring accuracy at turns t, t+1, ..., T.

This is a finer-grained version of the whole-sequence ZERO_WRITE ablation. Instead of
removing all writes (which breaks the reader entirely), this removes writes at a single
turn boundary, keeping the model in-distribution for all turns except those that depend
on turn-t writes.

Protocol:
  For each ablation turn t in {1, ..., T-1}:
    - Run NORMAL for turns 0..t-1, accumulating workspace normally.
    - At turn t: reset workspace to empty before the forward pass of turn t.
    - Continue normally for turns t..T.
    - Measure accuracy at turns t, t+1, ..., T.

  Compare turn-level accuracy against NORMAL baseline at those turns.

Output: JSON report + heatmap of accuracy delta[ablation_turn, eval_turn].

Usage:
    python scripts/analyze_per_turn_ablation.py \\
        --config  configs/experiments/fair_hard_n3_lam001.yaml \\
        --checkpoint results/fair_hard_n3_lam001/best.pt \\
        --output-dir results/per_turn_ablation/fair_hard_n3_lam001
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
# Workspace reset hook
# ---------------------------------------------------------------------------

class _WorkspaceResetAtTurnHook:
    """Zero out the workspace state at the START of a specific turn.

    Works by monkey-patching the model's step method so that before turn
    `target_turn`, it empties workspace.records, .mask, .step_written, etc.
    """

    def __init__(self, model: SequentialReasoningModel, target_turn: int):
        self._model = model
        self._target_turn = target_turn
        self._orig_forward = None

    def __enter__(self):
        target_turn = self._target_turn
        model = self._model

        # Find the sequential forward to patch
        orig_forward = model.forward

        def patched_forward(facts_seq, fact_mask, turn_mask):
            # We need to intercept the model's internal turn loop.
            # Strategy: run normal forward but reset workspace before turn t.
            # Since the sequential model loops internally, we patch the workspace
            # object so that when turn = target_turn is entered, we zero it.
            return _run_with_reset(model, facts_seq, fact_mask, turn_mask,
                                   target_turn, orig_forward)

        self._orig_forward = orig_forward
        model.forward = patched_forward
        return self

    def __exit__(self, *_):
        self._model.forward = self._orig_forward


def _empty_workspace_like(ws):
    """Return a workspace with the same shape/device but all slots empty."""
    import copy
    ws_empty = copy.copy(ws)
    ws_empty.records      = torch.zeros_like(ws.records)
    ws_empty.mask         = torch.zeros_like(ws.mask)
    ws_empty.step_written = torch.zeros_like(ws.step_written)
    if hasattr(ws, "types"):
        ws_empty.types = torch.zeros_like(ws.types)
    if hasattr(ws, "wrote_this_pass"):
        ws_empty.wrote_this_pass = torch.zeros_like(ws.wrote_this_pass)
    return ws_empty


def _run_with_reset(model: SequentialReasoningModel,
                    facts_seq, fact_mask, turn_mask,
                    reset_at_turn: int,
                    orig_forward):
    """Run the model's sequential forward, clearing workspace before reset_at_turn."""
    # We use the model's step-by-step interface if available, otherwise
    # we fall back to hooking the workspace object directly.
    #
    # Primary strategy: patch the workspace object to zero itself on the
    # correct turn. We intercept the workspace's write_controller.forward
    # to do the reset on the first call in the target turn.

    # Walk into workspace
    try:
        ws_module = model.base_model.workspace
    except AttributeError:
        # Fall back to unpatched forward if we can't find the workspace
        return orig_forward(facts_seq, fact_mask, turn_mask)

    write_ctrl = getattr(ws_module, "write_controller", None)
    if write_ctrl is None:
        return orig_forward(facts_seq, fact_mask, turn_mask)

    # Track which turn we're in via a mutable counter
    state = {"turn": 0, "step_in_turn": 0, "reset_done": False}

    orig_ws_forward = write_ctrl.forward

    def _patched_ws_forward(workspace, V, A, step):
        # step is the reasoning step index (0..R-1) within the current turn.
        # We use step==0 to detect the start of a new turn.
        if step == 0 and not state["reset_done"]:
            state["step_in_turn"] = 0
            if state["turn"] > 0:
                state["turn"] += 0  # already incremented outside
        elif step == 0:
            pass

        # Detect turn transition: when step resets to 0 after step > 0
        if step == 0 and state["step_in_turn"] > 0:
            state["turn"] += 1
            state["reset_done"] = False

        state["step_in_turn"] = step

        # Reset workspace at the start of target turn (step=0)
        if state["turn"] == reset_at_turn and step == 0 and not state["reset_done"]:
            ws_empty = _empty_workspace_like(workspace)
            # Copy the emptied fields back into workspace in-place
            workspace.records = ws_empty.records
            workspace.mask = ws_empty.mask
            workspace.step_written = ws_empty.step_written
            if hasattr(workspace, "types"):
                workspace.types = ws_empty.types
            if hasattr(workspace, "wrote_this_pass"):
                workspace.wrote_this_pass = ws_empty.wrote_this_pass
            state["reset_done"] = True

        return orig_ws_forward(workspace, V, A, step)

    write_ctrl.forward = _patched_ws_forward
    try:
        result = orig_forward(facts_seq, fact_mask, turn_mask)
    finally:
        write_ctrl.forward = orig_ws_forward

    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_normal(model, loader, device, num_batches=None):
    """Run NORMAL forward, return per-turn accuracy dicts."""
    model.eval()
    turn_correct: dict[int, int] = {}
    turn_counts:  dict[int, int] = {}

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if num_batches is not None and b_idx >= num_batches:
                break
            facts_seq = batch["facts"].to(device)
            fact_mask  = batch["fact_mask"].to(device)
            turn_mask  = batch["turn_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            all_logits, _ = model(facts_seq, fact_mask, turn_mask)
            B, T, _ = all_logits.shape

            for t in range(T):
                valid = (turn_mask[:, t] > 0).nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue
                preds = all_logits[valid, t].argmax(dim=-1)
                c = (preds == labels[valid, t]).sum().item()
                turn_correct[t] = turn_correct.get(t, 0) + c
                turn_counts[t]  = turn_counts.get(t, 0) + len(valid)

    return {t: turn_correct[t] / max(1, turn_counts[t]) for t in sorted(turn_counts)}


def evaluate_with_reset(model, loader, device, reset_at_turn, num_batches=None):
    """Run forward with workspace zeroed at the start of reset_at_turn."""
    model.eval()
    turn_correct: dict[int, int] = {}
    turn_counts:  dict[int, int] = {}

    # We need to apply the reset logic on each batch; use _run_with_reset directly
    try:
        ws_module = model.base_model.workspace
        write_ctrl = ws_module.write_controller
    except AttributeError:
        print(f"  [Warning] Cannot find workspace write_controller. Skipping turn {reset_at_turn}.")
        return {}

    orig_ws_forward = write_ctrl.forward

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if num_batches is not None and b_idx >= num_batches:
                break

            facts_seq = batch["facts"].to(device)
            fact_mask  = batch["fact_mask"].to(device)
            turn_mask  = batch["turn_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            B, T = labels.shape[0], labels.shape[1]

            # Reset state for each batch
            state = {"turn": 0, "step_in_turn": 0, "reset_done": False}

            def _patched(workspace, V, A, step):
                if step == 0 and state["step_in_turn"] > 0:
                    state["turn"] += 1
                    state["reset_done"] = False
                state["step_in_turn"] = step

                if state["turn"] == reset_at_turn and step == 0 and not state["reset_done"]:
                    workspace.records      = torch.zeros_like(workspace.records)
                    workspace.mask         = torch.zeros_like(workspace.mask)
                    workspace.step_written = torch.zeros_like(workspace.step_written)
                    if hasattr(workspace, "types"):
                        workspace.types = torch.zeros_like(workspace.types)
                    if hasattr(workspace, "wrote_this_pass"):
                        workspace.wrote_this_pass = torch.zeros_like(workspace.wrote_this_pass)
                    state["reset_done"] = True

                return orig_ws_forward(workspace, V, A, step)

            write_ctrl.forward = _patched
            all_logits, _ = model(facts_seq, fact_mask, turn_mask)
            write_ctrl.forward = orig_ws_forward

            for t in range(T):
                valid = (turn_mask[:, t] > 0).nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue
                preds = all_logits[valid, t].argmax(dim=-1)
                c = (preds == labels[valid, t]).sum().item()
                turn_correct[t] = turn_correct.get(t, 0) + c
                turn_counts[t]  = turn_counts.get(t, 0) + len(valid)

    return {t: turn_correct[t] / max(1, turn_counts[t]) for t in sorted(turn_counts)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_heatmap(delta_matrix, ablation_turns, eval_turns, out_path: Path):
    """Heatmap of delta[ablation_turn, eval_turn]. Negative = workspace was helping."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = np.array([[delta_matrix.get((a, e), float("nan"))
                      for e in eval_turns]
                     for a in ablation_turns])

    im = ax.imshow(data * 100, cmap="RdYlGn", aspect="auto",
                   vmin=-30, vmax=5)
    ax.set_xticks(range(len(eval_turns)))
    ax.set_xticklabels([f"Eval T{e+1}" for e in eval_turns], rotation=45, ha="right")
    ax.set_yticks(range(len(ablation_turns)))
    ax.set_yticklabels([f"Reset T{a+1}" for a in ablation_turns])
    ax.set_title("Accuracy Delta (Ablated - Normal, pp)\nNegative = workspace writes at reset turn were causally useful downstream")
    fig.colorbar(im, ax=ax, label="pp change")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--num-batches",  type=int, default=None)
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
    print("  Per-Turn Workspace Ablation (Condition 2 Gap)")
    print("=" * width)

    # NORMAL baseline
    print("\n[Baseline] Running NORMAL forward pass...")
    normal_acc = evaluate_normal(model, loader, device, args.num_batches)
    print("  Normal per-turn accuracy:")
    for t, a in normal_acc.items():
        print(f"    Turn {t+1}: {a*100:.2f}%")

    # Per-turn ablations
    ablation_turns = list(range(1, max_turns))  # reset at turns 1..T-1 (0-indexed)
    eval_turns     = list(range(max_turns))
    delta_matrix: dict[tuple, float] = {}

    results_by_ablation = {}
    for abl_t in ablation_turns:
        print(f"\n[Ablation] Resetting workspace at start of Turn {abl_t+1}...")
        abl_acc = evaluate_with_reset(model, loader, device, abl_t, args.num_batches)

        turn_results = {}
        for e_t in eval_turns:
            if e_t in normal_acc and e_t in abl_acc:
                delta = abl_acc[e_t] - normal_acc[e_t]
                delta_matrix[(abl_t, e_t)] = delta
                turn_results[str(e_t + 1)] = {
                    "normal_acc": round(normal_acc[e_t], 4),
                    "ablated_acc": round(abl_acc[e_t], 4),
                    "delta_pp": round(delta * 100, 2),
                }
        results_by_ablation[str(abl_t + 1)] = turn_results

        # Print summary row for turns >= ablation turn
        downstream_deltas = [
            delta_matrix[(abl_t, e_t)] * 100
            for e_t in range(abl_t, max_turns)
            if (abl_t, e_t) in delta_matrix
        ]
        if downstream_deltas:
            print(f"  Downstream turns {abl_t+1}–{max_turns}: "
                  f"mean delta = {np.mean(downstream_deltas):+.2f}pp, "
                  f"min = {min(downstream_deltas):+.2f}pp")

    # Print summary table
    print(f"\n{'':8}", end="")
    for e_t in eval_turns:
        print(f"{'EvalT'+str(e_t+1):>10}", end="")
    print()
    print("-" * (8 + 10 * len(eval_turns)))
    for abl_t in ablation_turns:
        print(f"ResetT{abl_t+1:>2}", end="")
        for e_t in eval_turns:
            key = (abl_t, e_t)
            if key in delta_matrix:
                d = delta_matrix[key] * 100
                marker = "*" if d < -5 else " "
                print(f"  {d:+6.2f}{marker}", end="")
            else:
                print(f"{'   N/A':>10}", end="")
        print()

    print("\n  (* = delta < -5pp, indicating a causal write at that turn)")
    print("=" * width)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "max_turns": max_turns,
        "normal_per_turn_acc": {str(t+1): round(a, 4) for t, a in normal_acc.items()},
        "ablation_results": results_by_ablation,
    }
    json_out = out_dir / "per_turn_ablation.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report : {json_out}")

    try:
        plot_heatmap(delta_matrix, ablation_turns, eval_turns,
                     out_dir / "per_turn_ablation_heatmap.png")
    except Exception as e:
        print(f"  [Warning] Plot failed: {e}")


if __name__ == "__main__":
    main()
