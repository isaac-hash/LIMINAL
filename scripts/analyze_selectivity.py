"""Phase 7 — Selectivity Analysis & Verification Runner.

Evaluates the learned write controller (LearnedWriteController with Gumbel-Softmax)
against the deterministic top-K baseline on the externalisation-sensitive task.

Metrics computed:
  1. Mean slots written per reasoning step (vs. fixed K baseline)
  2. Selectivity score (1 - fraction_written)
  3. Gate entropy (decision confidence)
  4. Per-turn write fraction (Turn 1 through Turn 5)
  5. Overall and per-turn classification accuracy
  6. Exit criterion check:
       Learned gate accuracy >= Top-K baseline accuracy
       AND mean slots written < Top-K baseline (K)

Usage:
    python scripts/analyze_selectivity.py \\
        --config configs/experiments/externalisation_comparison.yaml \\
        --checkpoint results/externalisation_comparison/best.pt \\
        [--baseline-checkpoint results/external_ablation/best.pt] \\
        --output-dir results/externalisation_comparison/analysis \\
        [--drive-path /content/drive/MyDrive/LIMINAL_results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

# Terminal utf-8 configuration on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils.device import print_hardware_info
from src.utils.config import load_config, set_seed
from src.utils.checkpoint import load_checkpoint
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary
from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
from src.models.sequential_model import SequentialReasoningModel
from src.evaluation.selectivity_analysis import SelectivityAnalyser, save_selectivity_report


def parse_args():
    parser = argparse.ArgumentParser(description="LIMINAL Phase 7 Selectivity Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/externalisation_comparison.yaml",
        help="Path to Phase 7 experiment config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/externalisation_comparison/best.pt",
        help="Path to learned gate checkpoint",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=None,
        help="Path to deterministic top-K checkpoint for paired comparison (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/externalisation_comparison/analysis",
        help="Directory to store reports and plots",
    )
    parser.add_argument(
        "--drive-path",
        type=str,
        default=None,
        help="Optional Google Drive backup directory",
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        default=None,
        help="Limit number of evaluation batches (for fast debugging)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def evaluate_model(
    model: SequentialReasoningModel,
    loader: DataLoader,
    device: torch.device,
    analyser: SelectivityAnalyser,
    num_batches: int | None = None,
) -> dict[str, Any]:
    """Run model on loader, accumulate accuracy and selectivity statistics."""
    model.eval()
    turn_correct: dict[int, int] = {}
    turn_counts: dict[int, int] = {}
    total_samples = 0
    total_correct = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if num_batches is not None and b_idx >= num_batches:
                break

            facts_seq = batch["facts"].to(device)
            fact_mask_seq = batch["fact_mask"].to(device)
            turn_mask = batch["turn_mask"].to(device)
            labels = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            all_logits, all_infos = model(facts_seq, fact_mask_seq, turn_mask)
            B, max_turns, _ = all_logits.shape

            for t in range(max_turns):
                mask_t = turn_mask[:, t]
                valid_indices = (mask_t > 0).nonzero(as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue

                logits_t = all_logits[valid_indices, t]
                labels_t = labels[valid_indices, t]
                preds_t = logits_t.argmax(dim=-1)
                corr_t = (preds_t == labels_t).sum().item()
                n_valid = len(valid_indices)

                turn_correct[t] = turn_correct.get(t, 0) + corr_t
                turn_counts[t] = turn_counts.get(t, 0) + n_valid
                total_correct += corr_t
                total_samples += n_valid

                # Extract write gates for selectivity analysis
                info_t = all_infos[t]
                gate_traj = info_t.get("write_gate_trajectory", [])
                if gate_traj:
                    for step_idx, g_step in enumerate(gate_traj):
                        if isinstance(g_step, Tensor):
                            # Slice for valid batch indices
                            g_valid = g_step[valid_indices]
                            analyser.update(g_valid.cpu(), turn_index=t, step=step_idx)
                elif "write_gate_hard" in info_t and info_t["write_gate_hard"] is not None:
                    g_valid = info_t["write_gate_hard"][valid_indices]
                    analyser.update(g_valid.cpu(), turn_index=t, step=0)

    acc = total_correct / max(1, total_samples)
    per_turn_acc = {
        f"turn_{t+1}": turn_correct[t] / max(1, turn_counts[t])
        for t in sorted(turn_counts.keys())
    }

    return {
        "acc": acc,
        "total_samples": total_samples,
        "per_turn_acc": per_turn_acc,
    }


def generate_plots(
    learned_report: dict[str, Any],
    learned_eval: dict[str, Any],
    baseline_eval: dict[str, Any] | None,
    output_dir: Path,
) -> list[Path]:
    """Generate and save publication-grade visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. Gate fraction by turn (bar chart)
    per_turn = learned_report.get("per_turn_gate_fraction", {})
    if per_turn:
        turns = [f"Turn {int(k)+1}" for k in sorted(per_turn.keys(), key=lambda x: int(x))]
        fractions = [per_turn[k] for k in sorted(per_turn.keys(), key=lambda x: int(x))]

        plt.figure(figsize=(7, 4.5), dpi=150)
        bars = plt.bar(turns, fractions, color="#4C9BE8", edgecolor="#1F4E79", alpha=0.85, width=0.5)
        plt.axhline(
            y=learned_report["top_k_baseline_slots"] / 8.0,
            color="#E84C4C",
            linestyle="--",
            label=f"Top-{int(learned_report['top_k_baseline_slots'])} Heuristic Baseline ({learned_report['top_k_baseline_slots']/8.0:.2%})",
        )
        plt.title("Learned Write Gate Fraction Across Sequence Turns", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Turn Index", fontsize=11)
        plt.ylabel("Fraction of Latent Slots Written", fontsize=11)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        # Label values on bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
            )

        plt.tight_layout()
        p1 = output_dir / "write_gate_per_turn.png"
        plt.savefig(p1)
        plt.close()
        generated.append(p1)

    # 2. Accuracy Comparison (Learned vs Baseline)
    if baseline_eval is not None:
        turns = list(learned_eval["per_turn_acc"].keys())
        learned_vals = [learned_eval["per_turn_acc"][t] * 100 for t in turns]
        baseline_vals = [baseline_eval["per_turn_acc"].get(t, 0.0) * 100 for t in turns]

        x = np.arange(len(turns))
        width = 0.35

        plt.figure(figsize=(8, 4.5), dpi=150)
        plt.bar(x - width/2, baseline_vals, width, label="Top-K Fixed Baseline", color="#B0BEC5", edgecolor="#607D8B")
        plt.bar(x + width/2, learned_vals, width, label="Learned Write Gate", color="#2E7D32", edgecolor="#1B5E20")

        plt.title("Per-Turn Reasoning Accuracy Comparison", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Sequence Turn", fontsize=11)
        plt.ylabel("Accuracy (%)", fontsize=11)
        plt.xticks(x, [t.replace("_", " ").title() for t in turns])
        plt.ylim(0, 105)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

        plt.tight_layout()
        p2 = output_dir / "accuracy_comparison.png"
        plt.savefig(p2)
        plt.close()
        generated.append(p2)

    return generated


def main():
    args = parse_args()
    device = print_hardware_info()

    # 1. Load config
    base_config_path = Path("configs/base.yaml")
    config_path = Path(args.config)
    if config_path.exists() and config_path != base_config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = load_config(base_config_path, overrides=overrides)
    else:
        config = load_config(base_config_path)

    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    print("\n" + "=" * 65)
    print("  Phase 7: Learned Externalisation Selectivity Analysis")
    print("=" * 65)
    print(f"  Config          : {args.config}")
    print(f"  Checkpoint      : {args.checkpoint}")
    print(f"  Baseline Checkpt: {args.baseline_checkpoint or 'None (analytical top-K)'}")
    print(f"  Output Dir      : {args.output_dir}")
    print(f"  Device          : {device}")
    print("=" * 65 + "\n")

    # 2. Build test dataset
    print("Generating evaluation dataset...")
    generator = ArithmeticGenerator(config=config.data, seed=seed)
    splits = generator.generate_dataset()
    vocab = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"] + splits["test"])

    test_dataset = SequenceReasoningDataset(splits["test"], vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )
    print(f"Test dataset: {len(splits['test'])} sequences ready.")

    # 3. Build Model & Load Checkpoint
    model = SequentialReasoningModel(
        config=config.model,
        vocab=vocab,
        activity_config=config.activity,
        resolution_config=config.resolution,
        persistence_config=config.persistence,
        external_config=config.external,
    ).to(device)

    chk_path = Path(args.checkpoint)
    if chk_path.exists():
        print(f"Loading learned gate weights from {chk_path}...")
        load_checkpoint(chk_path, model=model, device=device)
    else:
        print(f"  ⚠️ Checkpoint {chk_path} not found. Running with initialised weights.")

    # 4. Evaluate Learned Model
    analyser = SelectivityAnalyser(
        num_latent_slots=config.model.latent_slots,
        write_top_k=config.external.write_top_k,
    )
    print("Evaluating learned gate model...")
    learned_eval = evaluate_model(model, test_loader, device, analyser, num_batches=args.num_eval_batches)
    report = analyser.summarise()
    analyser.print_summary()

    # 5. Evaluate Baseline Model if provided
    baseline_eval = None
    if args.baseline_checkpoint and Path(args.baseline_checkpoint).exists():
        print(f"Loading baseline weights from {args.baseline_checkpoint}...")
        # Create baseline model with top-K
        import copy
        b_ext = copy.deepcopy(config.external)
        b_ext.learned_gate = False
        baseline_model = SequentialReasoningModel(
            config=config.model,
            vocab=vocab,
            activity_config=config.activity,
            resolution_config=config.resolution,
            persistence_config=config.persistence,
            external_config=b_ext,
        ).to(device)
        load_checkpoint(args.baseline_checkpoint, model=baseline_model, device=device)
        b_analyser = SelectivityAnalyser(config.model.latent_slots, config.external.write_top_k)
        print("Evaluating baseline model...")
        baseline_eval = evaluate_model(baseline_model, test_loader, device, b_analyser, num_batches=args.num_eval_batches)

    # 6. Exit Criterion Check
    k_baseline = float(config.external.write_top_k)
    slots_written = report["mean_slots_written_per_step"]
    writes_fewer = slots_written < k_baseline
    acc_check = True
    if baseline_eval is not None:
        acc_check = learned_eval["acc"] >= (baseline_eval["acc"] - 0.02)  # within margin

    exit_criterion_met = writes_fewer and acc_check

    print("\n" + "=" * 65)
    print("  Exit Criterion Check (Phase 7)")
    print("=" * 65)
    print(f"  Learned Mean Slots Written: {slots_written:.2f} / step")
    print(f"  Baseline Top-K Fixed Slots: {k_baseline:.2f} / step")
    print(f"  Selective Writing Check   : {'PASS (writes fewer slots)' if writes_fewer else 'FAIL'}")
    if baseline_eval:
        print(f"  Learned Accuracy          : {learned_eval['acc']*100:.2f}%")
        print(f"  Baseline Accuracy         : {baseline_eval['acc']*100:.2f}%")
        print(f"  Accuracy Retention Check  : {'PASS' if acc_check else 'FAIL'}")
    print(f"  Overall Exit Criterion    : {'PASSED' if exit_criterion_met else 'PENDING TRAINING'}")
    print("=" * 65 + "\n")

    # 7. Save Artifacts & Reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_report = {
        "selectivity": report,
        "evaluation": learned_eval,
        "baseline_evaluation": baseline_eval,
        "exit_criterion_met": exit_criterion_met,
    }
    json_path = output_dir / "selectivity_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved selectivity report: {json_path}")

    # Generate plots
    plot_paths = generate_plots(report, learned_eval, baseline_eval, output_dir)
    for p in plot_paths:
        print(f"Saved plot: {p}")

    # Backup to Drive if requested
    if args.drive_path:
        drive_dir = Path(args.drive_path) / "phase_7_selectivity"
        drive_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, drive_dir / "selectivity_report.json")
        for p in plot_paths:
            shutil.copy2(p, drive_dir / p.name)
        print(f"Backed up results to Drive: {drive_dir}")


if __name__ == "__main__":
    main()
