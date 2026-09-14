"""Phase 2B: Activity Gate Analysis Script.

Loads a trained activity-graph checkpoint, runs it over the test set,
and produces three diagnostic plots plus a printed statistics report.

Usage:
    python scripts/analyze_activity.py --config configs/activity_graph.yaml

Outputs (saved to the checkpoint directory):
    activity_heatmap.png       -- [T x N] mean gate values per step
    activity_per_class.png     -- mean gate value per slot, split by predicted class
    activity_over_steps.png    -- per-slot mean activity across T reasoning steps
    activity_stats.json        -- numerical summary of activity statistics
"""

import sys
# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.utils.config import load_config, set_seed
from src.utils.checkpoint import load_checkpoint
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary, ReasoningDataset, collate_reasoning_batch
from src.models.model import ReasoningModel
from src.evaluation.activity_analysis import (
    collect_activity_data,
    compute_activity_stats,
    plot_activity_heatmap,
    plot_per_class_activity,
    plot_activity_over_steps,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LIMINAL Activity Gate Analyser")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/activity_graph.yaml",
        help="Path to activity-graph config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (default: <checkpoint_dir>/best.pt)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit analysis to N batches (useful for quick iteration)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config (overlay approach, same as train.py)
    base_cfg_path = Path("configs/base.yaml")
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    config = load_config(base_cfg_path, overrides=overrides)

    if not config.activity.enabled:
        print("[ERROR] Activity gates are not enabled in this config.")
        print("        Run with a config that has 'activity.enabled: true'.")
        sys.exit(1)

    set_seed(config.seed)
    device = "cpu"

    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "best.pt"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("        Train the model first: python scripts/train.py --config configs/activity_graph.yaml")
        sys.exit(1)

    print(f"\n--- Activity Gate Analysis ---")
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Slots:      {config.model.latent_slots}")
    print(f"Steps:      {config.model.reasoning_steps}")
    print(f"Gate dim:   {config.activity.gate_hidden_dim}")

    # Rebuild dataset and vocab (identical seed = identical splits)
    print("\nRebuilding test split...")
    generator = ArithmeticGenerator(config=config.data, seed=config.seed)
    splits = generator.generate_dataset()

    vocab = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"] + splits["test"])

    test_loader = DataLoader(
        ReasoningDataset(splits["test"], vocab),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
    )

    # Build and restore model
    model = ReasoningModel(
        config=config.model,
        vocab=vocab,
        activity_config=config.activity,
    ).to(device)

    load_checkpoint(ckpt_path, model)
    print(f"Checkpoint loaded from {ckpt_path}")

    # Collect activity data
    print(f"\nCollecting activity trajectories over test set ({len(splits['test'])} samples)...")
    data = collect_activity_data(
        model, test_loader, device=device, max_batches=args.max_batches
    )
    print(f"Collected {data['A_finals'].shape[0]} samples. "
          f"Slots: {data['n_slots']}, Steps: {data['n_steps']}")

    # Compute statistics
    stats = compute_activity_stats(data)

    print("\n--- Slot Activity Statistics ---")
    print(f"  Effective active slots : {stats['effective_slots']:.2f} / {data['n_slots']}")
    print(f"  Slot utilization (>0.5): {stats['utilization'] * 100:.1f}%")
    print(f"  Mean activity per slot :")
    for i, (m, s) in enumerate(zip(stats["mean_per_slot"], stats["std_per_slot"])):
        bar = "#" * int(m * 20)
        print(f"    Slot {i}: {m:.3f} +/- {s:.3f}  [{bar:<20}]")
    print(f"  Mean activity per step :")
    for t, m in enumerate(stats["mean_per_step"]):
        print(f"    Step {t}: {m:.3f}")

    # Save numerical stats to JSON
    stats_path = ckpt_dir / "activity_stats.json"
    json_safe = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in stats.items()
    }
    json_safe["n_samples"] = int(data["A_finals"].shape[0])
    json_safe["n_slots"] = int(data["n_slots"])
    json_safe["n_steps"] = int(data["n_steps"])
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\nStats saved -> {stats_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_activity_heatmap(
        data,
        output_path=str(ckpt_dir / "activity_heatmap.png"),
        title=f"Slot Activity Heatmap (test set, {data['A_finals'].shape[0]} samples)",
    )
    plot_per_class_activity(
        data,
        class_names=["Not affordable", "Affordable"],
        output_path=str(ckpt_dir / "activity_per_class.png"),
        title="Mean Slot Activity by Predicted Class",
    )
    plot_activity_over_steps(
        data,
        output_path=str(ckpt_dir / "activity_over_steps.png"),
        title="Per-Slot Mean Activity Across Reasoning Steps",
    )

    print(f"\n[DONE] All outputs saved to: {ckpt_dir}/")
    print("  activity_heatmap.png")
    print("  activity_per_class.png")
    print("  activity_over_steps.png")
    print("  activity_stats.json")


if __name__ == "__main__":
    main()
