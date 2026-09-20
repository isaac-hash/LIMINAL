"""Phase 4 — Persistence Analysis Script.

Evaluates trained persistent models on multi-turn reasoning tasks:
  1. Per-turn accuracy breakdown (Turn 1 vs Turn 2 vs Turn 3+)
  2. Persistence vs Reset baseline comparison (if reset checkpoint provided)
  3. Slot persistence gate activations (which slots retain prior state)
  4. Gate trajectories across consecutive sequence turns

Usage:
    python scripts/analyze_persistence.py \\
        --config configs/persistence_graph.yaml \\
        --checkpoint results/persistence_graph/best.pt \\
        [--reset-checkpoint results/persistence_reset/best.pt] \\
        --output-dir results/persistence_graph/analysis
"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.device import print_hardware_info
from src.utils.config import load_config, set_seed
from src.utils.checkpoint import load_checkpoint
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary
from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
from src.models.sequential_model import SequentialReasoningModel


def parse_args():
    parser = argparse.ArgumentParser(description="LIMINAL Phase 4 Persistence Analysis")
    parser.add_argument("--config", type=str, default="configs/persistence_graph.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to persistent model checkpoint (e.g. best.pt)")
    parser.add_argument("--reset-checkpoint", type=str, default=None, help="Path to reset baseline checkpoint (optional)")
    parser.add_argument("--output-dir", type=str, default="results/persistence_graph/analysis")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def collect_sequential_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Run model on test loader and return per-turn accuracy and gate metrics."""
    model.eval()
    turn_correct: dict[int, int] = {}
    turn_counts: dict[int, int] = {}
    # slot_gates[t] is list of arrays [N]
    turn_gates: dict[int, list[np.ndarray]] = {}

    with torch.no_grad():
        for batch in loader:
            facts_seq = batch["facts"].to(device)
            fact_mask_seq = batch["fact_mask"].to(device)
            turn_mask = batch["turn_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            all_logits, all_infos = model(facts_seq, fact_mask_seq, turn_mask)
            B, max_turns, _ = all_logits.shape

            for t in range(max_turns):
                mask_t = turn_mask[:, t]
                valid_idx = (mask_t > 0).nonzero(as_tuple=True)[0]
                if len(valid_idx) == 0:
                    continue

                logits_t = all_logits[valid_idx, t]
                labels_t = labels[valid_idx, t]
                preds_t = logits_t.argmax(dim=-1)

                corr = (preds_t == labels_t).sum().item()
                n_valid = len(valid_idx)

                turn_correct[t] = turn_correct.get(t, 0) + corr
                turn_counts[t] = turn_counts.get(t, 0) + n_valid

                info_t = all_infos[t]
                if "persistence_gates" in info_t:
                    pg = info_t["persistence_gates"][valid_idx].cpu().numpy()  # [B_valid, N]
                    if t not in turn_gates:
                        turn_gates[t] = []
                    turn_gates[t].append(pg)

    accuracies = {t: turn_correct[t] / max(1, turn_counts[t]) for t in turn_counts}
    
    mean_slot_gates: dict[int, np.ndarray] = {}
    for t, gate_list in turn_gates.items():
        all_pg = np.concatenate(gate_list, axis=0)  # [total_valid, N]
        mean_slot_gates[t] = np.mean(all_pg, axis=0)

    return accuracies, mean_slot_gates


def plot_accuracy_comparison(
    persistent_accs: dict[int, float],
    reset_accs: dict[int, float] | None,
    output_dir: Path,
):
    """Plot per-turn accuracy comparing persistent vs reset models."""
    turns = sorted(persistent_accs.keys())
    turn_labels = [f"Turn {t + 1}" for t in turns]
    p_vals = [persistent_accs[t] * 100 for t in turns]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(turns))
    width = 0.35

    r_vals: list[float] = []
    if reset_accs is not None:
        r_vals = [reset_accs.get(t, 0.0) * 100 for t in turns]
        ax.bar(x - width / 2, p_vals, width, label="Persistent (Model E)", color="#2b5c8f", alpha=0.9)
        ax.bar(x + width / 2, r_vals, width, label="Reset Baseline (Model E-reset)", color="#d95f02", alpha=0.8)
    else:
        ax.bar(x, p_vals, width, label="Persistent Model", color="#2b5c8f", alpha=0.9)

    ax.set_xlabel("Sequence Turn", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Multi-Turn Accuracy: Persistence vs Reset", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(turn_labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    # Annotate values
    for i, v in enumerate(p_vals):
        offset = -width / 2 if reset_accs is not None else 0
        ax.text(i + offset, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    if reset_accs is not None:
        for i, v in enumerate(r_vals):
            ax.text(i + width / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "per_turn_accuracy.png", dpi=150)
    plt.close(fig)


def plot_gate_heatmap(mean_slot_gates: dict[int, np.ndarray], output_dir: Path):
    """Plot heatmap of mean persistence gate values per slot across turns."""
    if not mean_slot_gates:
        return

    turns = sorted(mean_slot_gates.keys())
    matrix = np.array([mean_slot_gates[t] for t in turns])  # [num_turns_with_gate, N]
    num_slots = matrix.shape[1]

    fig, ax = plt.subplots(figsize=(8, 4))
    cax = ax.matshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, label="Mean Gate Activation (0=Fresh, 1=Retain)")

    ax.set_xticks(np.arange(num_slots))
    ax.set_xticklabels([f"Slot {i}" for i in range(num_slots)])
    ax.set_yticks(np.arange(len(turns)))
    ax.set_yticklabels([f"Turn {t + 1}" for t in turns])
    ax.set_title("Persistence Gate Activation by Slot and Turn", fontsize=13, pad=15)

    for t_idx in range(len(turns)):
        for s_idx in range(num_slots):
            val = matrix[t_idx, s_idx]
            color = "white" if val > 0.5 else "black"
            ax.text(s_idx, t_idx, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "persistence_gate_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = print_hardware_info()

    base_config_path = Path("configs/base.yaml")
    config_path = Path(args.config)
    if config_path != base_config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = load_config(base_config_path, overrides=overrides)
    else:
        config = load_config(base_config_path)

    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate test dataset splits
    generator = ArithmeticGenerator(config=config.data, seed=seed)
    dataset_splits = generator.generate_dataset()
    test_records = dataset_splits["test"]

    vocab = Vocabulary()
    vocab.build_from_records(dataset_splits["train"] + dataset_splits["val"] + test_records)

    test_dataset = SequenceReasoningDataset(test_records, vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    # 2. Evaluate persistent model
    print(f"Loading persistent checkpoint: {args.checkpoint}")
    model = SequentialReasoningModel(
        config=config.model,
        vocab=vocab,
        activity_config=config.activity,
        resolution_config=config.resolution,
        persistence_config=config.persistence,
    ).to(device)
    load_checkpoint(args.checkpoint, model=model, device=device)

    p_accs, slot_gates = collect_sequential_metrics(model, test_loader, device)

    # 3. Optionally evaluate reset baseline
    r_accs = None
    if args.reset_checkpoint:
        print(f"Loading reset baseline checkpoint: {args.reset_checkpoint}")
        import copy
        reset_persistence_config = copy.deepcopy(config.persistence)
        # Force persistence disabled for reset model
        from src.utils.config import PersistenceConfig
        reset_config_obj = PersistenceConfig(enabled=False)

        reset_model = SequentialReasoningModel(
            config=config.model,
            vocab=vocab,
            activity_config=config.activity,
            resolution_config=config.resolution,
            persistence_config=reset_config_obj,
        ).to(device)
        load_checkpoint(args.reset_checkpoint, model=reset_model, device=device)
        r_accs, _ = collect_sequential_metrics(reset_model, test_loader, device)

    # 4. Generate plots
    print(f"Saving diagnostic plots to {output_dir}...")
    plot_accuracy_comparison(p_accs, r_accs, output_dir)
    plot_gate_heatmap(slot_gates, output_dir)

    # 5. Output summary metrics JSON
    summary = {
        "persistent_accuracies": {f"turn_{t + 1}": float(acc) for t, acc in p_accs.items()},
        "persistent_overall_accuracy": float(np.mean(list(p_accs.values()))),
    }
    if r_accs is not None:
        summary["reset_accuracies"] = {f"turn_{t + 1}": float(acc) for t, acc in r_accs.items()}
        summary["reset_overall_accuracy"] = float(np.mean(list(r_accs.values())))
        summary["accuracy_deltas"] = {
            f"turn_{t + 1}": float(p_accs[t] - r_accs[t]) for t in p_accs if t in r_accs
        }

    with open(output_dir / "persistence_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Persistence Analysis Summary ---")
    for t in sorted(p_accs.keys()):
        p_str = f"Turn {t + 1} Acc: {p_accs[t] * 100:.2f}%"
        if r_accs is not None and t in r_accs:
            p_str += f"  (Reset: {r_accs[t] * 100:.2f}%, Δ = {(p_accs[t] - r_accs[t]) * 100:+.2f}%)"
        print(f"  {p_str}")
    print(f"Detailed results and plots written to {output_dir}\n")


if __name__ == "__main__":
    main()
