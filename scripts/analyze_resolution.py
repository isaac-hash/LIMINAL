"""Phase 3 — Resolution Analysis Script.

Loads a trained resolution model checkpoint and produces diagnostic plots:

  1. Step count histogram     — distribution of effective steps across test set
  2. Step count by task family — affordability (easy) vs multi_step (hard)
  3. Halt probability trajectory — mean h_t at each step (should peak early)
  4. Step count vs accuracy  — do harder examples use more steps?
  5. Activity × Resolution heatmap — per-slot gate activity at each step

Usage:
    python scripts/analyze_resolution.py \\
        --config configs/resolution_graph.yaml \\
        --checkpoint results/resolution_graph/best.pt \\
        --output-dir results/resolution_graph/analysis
"""

import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.utils.device import print_hardware_info
from src.utils.config import load_config, set_seed
from src.utils.checkpoint import load_checkpoint
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary, ReasoningDataset, collate_reasoning_batch
from src.models.model import ReasoningModel


def parse_args():
    parser = argparse.ArgumentParser(description="LIMINAL Phase 3 Resolution Analysis")
    parser.add_argument("--config", type=str, default="configs/resolution_graph.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--output-dir", type=str, default="results/resolution_graph/analysis")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def collect_test_outputs(model, loader, device):
    """Run model on test set and collect per-example diagnostics."""
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            facts = batch["facts"].to(device)
            fact_mask = batch["fact_mask"].to(device)
            labels = batch["label"].to(device)
            task_families = batch.get("task_family", ["unknown"] * facts.shape[0])

            logits, info = model(facts, fact_mask)
            preds = logits.argmax(dim=-1)
            correct = (preds == labels)

            # --- Per-example fields ---
            n_steps = info.get("n_steps", None)
            halt_probs = info.get("halt_probs", None)
            ponder_weights = info.get("ponder_weights", None)
            activity_traj = info.get("activity_trajectory", None)

            B = facts.shape[0]
            for b in range(B):
                rec = {
                    "correct": correct[b].item(),
                    "label": labels[b].item(),
                    "task_family": task_families[b] if isinstance(task_families, list) else task_families,
                }
                if n_steps is not None:
                    rec["n_steps"] = n_steps[b].item()
                if halt_probs is not None:
                    rec["halt_probs"] = [h[b].item() for h in halt_probs]
                if ponder_weights is not None:
                    rec["ponder_weights"] = ponder_weights[b].cpu().numpy()
                if activity_traj is not None:
                    rec["activity_traj"] = [a[b].cpu().numpy() for a in activity_traj]
                records.append(rec)
    return records


def plot_step_histogram(records, out_dir: Path):
    """1. Distribution of effective step counts across full test set."""
    steps = [r["n_steps"] for r in records if "n_steps" in r]
    if not steps:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(steps, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Effective steps (n_steps)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Effective Reasoning Steps (Test Set)", fontsize=13)
    ax.axvline(np.mean(steps), color="#C44E52", linestyle="--", label=f"Mean = {np.mean(steps):.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "step_histogram.png", dpi=150)
    plt.close(fig)
    print(f"[plot] step_histogram.png  (mean={np.mean(steps):.2f}, std={np.std(steps):.2f})")


def plot_steps_by_family(records, out_dir: Path):
    """2. Step count broken down by task family."""
    families = sorted(set(r["task_family"] for r in records if "task_family" in r))
    if len(families) < 2:
        return

    steps_by_family = {f: [r["n_steps"] for r in records if r.get("task_family") == f and "n_steps" in r]
                       for f in families}

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for idx, (family, steps) in enumerate(steps_by_family.items()):
        ax.hist(steps, bins=20, alpha=0.7, label=f"{family} (μ={np.mean(steps):.2f})",
                color=colors[idx % len(colors)], edgecolor="white")
    ax.set_xlabel("Effective steps", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Effective Steps by Task Family", fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "steps_by_family.png", dpi=150)
    plt.close(fig)
    for f, s in steps_by_family.items():
        print(f"[stats] {f:20s} mean_steps={np.mean(s):.2f}  std={np.std(s):.2f}  n={len(s)}")


def plot_halt_probability_trajectory(records, out_dir: Path):
    """3. Mean halt probability at each step index."""
    # Collect halt_probs lists aligned by step index
    max_len = max((len(r["halt_probs"]) for r in records if "halt_probs" in r), default=0)
    if max_len == 0:
        return

    step_means = []
    step_stds = []
    for t in range(max_len):
        vals = [r["halt_probs"][t] for r in records if "halt_probs" in r and t < len(r["halt_probs"])]
        step_means.append(np.mean(vals))
        step_stds.append(np.std(vals))

    xs = list(range(1, max_len + 1))
    means = np.array(step_means)
    stds = np.array(step_stds)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, means, marker="o", color="#4C72B0", linewidth=2, label="Mean h_t")
    ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="#4C72B0", label="±1 std")
    ax.set_xlabel("Reasoning step t", fontsize=12)
    ax.set_ylabel("Halt probability h_t", fontsize=12)
    ax.set_title("Halt Probability Trajectory (Mean ± Std)", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "halt_trajectory.png", dpi=150)
    plt.close(fig)
    print(f"[plot] halt_trajectory.png  (max_steps={max_len})")


def plot_steps_vs_accuracy(records, out_dir: Path):
    """4. Bin examples by step count and compute accuracy per bin."""
    steps = [r["n_steps"] for r in records if "n_steps" in r]
    correct = [r["correct"] for r in records if "n_steps" in r]
    if not steps:
        return

    bins = np.linspace(min(steps), max(steps) + 0.01, 8)
    bin_indices = np.digitize(steps, bins)
    bin_accs = {}
    bin_counts = {}
    for idx, (s, c) in enumerate(zip(steps, correct)):
        b = bin_indices[idx]
        bin_accs.setdefault(b, []).append(c)
        bin_counts[b] = bin_counts.get(b, 0) + 1

    xs = [bins[b - 1] for b in sorted(bin_accs)]
    ys = [np.mean(bin_accs[b]) for b in sorted(bin_accs)]
    ns = [bin_counts[b] for b in sorted(bin_accs)]

    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(xs, ys, s=[n * 2 for n in ns], color="#4C72B0", alpha=0.8, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Effective steps (bin lower bound)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Step Count (bubble size = n examples)", fontsize=13)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "steps_vs_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"[plot] steps_vs_accuracy.png")


def plot_activity_resolution_heatmap(records, out_dir: Path):
    """5. Activity × Resolution: mean per-slot gate at each reasoning step."""
    records_with_traj = [r for r in records if "activity_traj" in r and len(r["activity_traj"]) > 0]
    if not records_with_traj:
        return

    max_steps = max(len(r["activity_traj"]) for r in records_with_traj)
    n_slots = records_with_traj[0]["activity_traj"][0].shape[0]

    # Average across examples: shape [max_steps, n_slots]
    heatmap = np.zeros((max_steps, n_slots))
    counts = np.zeros(max_steps)
    for r in records_with_traj:
        for t, a in enumerate(r["activity_traj"]):
            heatmap[t] += a
            counts[t] += 1
    counts = np.maximum(counts, 1)
    heatmap = heatmap / counts[:, None]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap.T, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("Reasoning step t", fontsize=12)
    ax.set_ylabel("Slot index", fontsize=12)
    ax.set_title("Mean Activity Gate per Slot × Step (Activity × Resolution Heatmap)", fontsize=12)
    ax.set_xticks(range(max_steps))
    ax.set_xticklabels([str(t + 1) for t in range(max_steps)])
    ax.set_yticks(range(n_slots))
    plt.colorbar(im, ax=ax, label="Gate activation")
    fig.tight_layout()
    fig.savefig(out_dir / "activity_resolution_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"[plot] activity_resolution_heatmap.png  ({max_steps} steps × {n_slots} slots)")


def main():
    args = parse_args()
    device = print_hardware_info()

    # Load config
    base_config_path = Path("configs/base.yaml")
    config_path = Path(args.config)
    if config_path != base_config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = load_config(base_config_path, overrides=overrides)
    else:
        config = load_config(base_config_path)

    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    print(f"\nConfig: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resolution enabled: {config.resolution.enabled}")
    print(f"Task family: {config.data.task_family}")

    # Generate test data
    generator = ArithmeticGenerator(config=config.data, seed=seed)
    splits = generator.generate_dataset()
    all_records = splits["train"] + splits["val"] + splits["test"]

    vocab = Vocabulary()
    vocab.build_from_records(all_records)

    test_dataset = ReasoningDataset(splits["test"], vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
    )

    # Build model
    model = ReasoningModel(
        config=config.model,
        vocab=vocab,
        activity_config=config.activity,
        resolution_config=config.resolution,
    ).to(device)

    # Load checkpoint
    load_checkpoint(args.checkpoint, model=model, device=device)
    print(f"\nLoaded checkpoint from {args.checkpoint}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Collect test outputs
    print("\nRunning inference on test set...")
    records = collect_test_outputs(model, test_loader, device)
    overall_acc = sum(r["correct"] for r in records) / len(records)
    print(f"Test accuracy: {overall_acc * 100:.2f}%  ({len(records)} examples)")

    # Summary stats
    if any("n_steps" in r for r in records):
        steps = [r["n_steps"] for r in records if "n_steps" in r]
        print(f"Effective steps:  mean={np.mean(steps):.3f}  std={np.std(steps):.3f}  "
              f"min={np.min(steps):.3f}  max={np.max(steps):.3f}")

    # Output plots
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {out_dir}")

    plot_step_histogram(records, out_dir)
    plot_steps_by_family(records, out_dir)
    plot_halt_probability_trajectory(records, out_dir)
    plot_steps_vs_accuracy(records, out_dir)
    plot_activity_resolution_heatmap(records, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
