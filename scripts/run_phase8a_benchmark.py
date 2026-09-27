#!/usr/bin/env python
"""Phase 8A Benchmark Runner — CLI entry point.

Usage (run from the project root, e.g. in Colab or locally)::

    python scripts/run_phase8a_benchmark.py \\
        --vector-ckpt  results/vector_baseline/best.pt \\
        --static-ckpt  results/static_graph/best.pt \\
        --phase4-ckpt  LIMINAL_results/persistence_incremental/best.pt \\
        --phase7-ckpt  LIMINAL_results/externalisation_comparison/best.pt \\
        --output-dir   results/phase8a \\
        --device       cpu

Outputs written to ``--output-dir``:
    phase8a_metrics.json          — Full numeric results for all 4 models.
    phase8a_metrics.csv           — Flat CSV for spreadsheet import.
    phase8a_comparison_table.md   — Markdown comparison table (copy-paste ready).
    turn_accuracy_decay.png       — Per-turn accuracy curves for all models.
    accuracy_vs_compute.png       — Parameters vs. accuracy vs. latency scatter.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running from project root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 8A: Unified 4-model comparative evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vector-ckpt",
        default="results/vector_baseline/best.pt",
        metavar="PATH",
        help="Checkpoint for Model A — Vector Baseline.",
    )
    p.add_argument(
        "--static-ckpt",
        default="results/static_graph/best.pt",
        metavar="PATH",
        help="Checkpoint for Model B — Static Graph.",
    )
    p.add_argument(
        "--phase4-ckpt",
        default="LIMINAL_results/persistence_incremental/best.pt",
        metavar="PATH",
        help="Checkpoint for Model C — Persistent Graph (Phase 4).",
    )
    p.add_argument(
        "--phase7-ckpt",
        default="LIMINAL_results/externalisation_comparison/best.pt",
        metavar="PATH",
        help="Checkpoint for Model D — Full LIMINAL (Phase 7).",
    )
    p.add_argument(
        "--output-dir",
        default="results/phase8a",
        metavar="DIR",
        help="Directory for all output files.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="PyTorch device string (cpu / cuda / mps).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Mini-batch size during evaluation.",
    )
    p.add_argument(
        "--n-sequences",
        type=int,
        default=500,
        metavar="N",
        help="Number of evaluation sequences (holdout, seed=999).",
    )
    p.add_argument(
        "--n-turns",
        type=int,
        default=5,
        metavar="T",
        help="Turns per evaluation sequence.",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib plot generation (useful in headless environments).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------

def _save_json(results: list[dict], out_dir: Path) -> Path:
    """Serialise full results to JSON (numpy scalars → native Python)."""
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if hasattr(obj, "item"):          # numpy / torch scalar
            return obj.item()
        return obj

    path = out_dir / "phase8a_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"  JSON  → {path}")
    return path


def _save_csv(results: list[dict], out_dir: Path) -> Path:
    """Export a flat CSV with one row per model."""
    path = out_dir / "phase8a_metrics.csv"
    n_turns = max(len(r["per_turn_accuracy"]) for r in results)

    fieldnames = (
        ["model_id", "label", "overall_accuracy", "mean_loss",
         "mean_halt_steps", "mean_active_slots", "mean_ext_writes",
         "latency_ms_per_seq", "params_total", "params_trainable",
         "has_persistence", "has_external"]
        + [f"turn_{t+1}_acc" for t in range(n_turns)]
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict = {
                "model_id":           r["model_id"],
                "label":              r["label"],
                "overall_accuracy":   f"{r['overall_accuracy']:.6f}",
                "mean_loss":          f"{r['mean_loss']:.6f}",
                "mean_halt_steps":    f"{r['mean_halt_steps']:.4f}",
                "mean_active_slots":  f"{r['mean_active_slots']:.4f}",
                "mean_ext_writes":    f"{r['mean_ext_writes']:.4f}",
                "latency_ms_per_seq": f"{r['latency_ms_per_seq']:.4f}",
                "params_total":       r["param_count"]["total"],
                "params_trainable":   r["param_count"]["trainable"],
                "has_persistence":    r["has_persistence"],
                "has_external":       r["has_external"],
            }
            for t in range(n_turns):
                acc = r["per_turn_accuracy"][t] if t < len(r["per_turn_accuracy"]) else ""
                row[f"turn_{t+1}_acc"] = f"{acc:.6f}" if acc != "" else ""
            writer.writerow(row)

    print(f"  CSV   → {path}")
    return path


def _save_markdown_table(results: list[dict], out_dir: Path) -> Path:
    """Write a Markdown comparison table suitable for the Phase 8 report."""
    path = out_dir / "phase8a_comparison_table.md"
    n_turns = max(len(r["per_turn_accuracy"]) for r in results)

    lines: list[str] = []
    lines.append("# Phase 8A — Four-Model Comparative Evaluation\n")
    lines.append(
        "> Shared evaluation set: `affordability_sequence`, "
        f"{n_turns} turns, 500 sequences, seed=999 (held-out).\n"
    )

    # Header
    headers = (
        ["Metric"]
        + [f"**{r['label']}**" for r in results]
    )
    sep = [":---"] + [":---:" for _ in results]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    def _row(label: str, values: list[str]) -> str:
        return "| " + " | ".join([label] + values) + " |"

    # Overall accuracy
    lines.append(_row(
        "**Overall Accuracy**",
        [f"{r['overall_accuracy']*100:.2f}%" for r in results],
    ))

    # Per-turn accuracy
    for t in range(n_turns):
        vals = []
        for r in results:
            if t < len(r["per_turn_accuracy"]):
                vals.append(f"{r['per_turn_accuracy'][t]*100:.2f}%")
            else:
                vals.append("—")
        lines.append(_row(f"Turn {t+1} Accuracy", vals))

    # Test loss
    lines.append(_row(
        "**Cross-Entropy Loss**",
        [f"{r['mean_loss']:.4f}" for r in results],
    ))

    # Halt steps
    lines.append(_row(
        "Mean Halt Steps",
        [
            f"{r['mean_halt_steps']:.2f}" if (r["has_persistence"] or r["mean_halt_steps"] > 0)
            else "Fixed (4)"
            for r in results
        ],
    ))

    # Active slots
    lines.append(_row(
        "Mean Active Slots",
        [
            f"{r['mean_active_slots']:.3f}" if r["mean_halt_steps"] > 0
            else ("1/1" if r["model_id"] == "A" else "8/8")
            for r in results
        ],
    ))

    # External writes
    lines.append(_row(
        "Ext. Writes / Step",
        [f"{r['mean_ext_writes']:.3f}" if r["has_external"] else "N/A" for r in results],
    ))

    # Parameters
    lines.append(_row(
        "Trainable Parameters",
        [f"{r['param_count']['trainable']:,}" for r in results],
    ))

    # Latency
    lines.append(_row(
        "Latency (ms/seq)",
        [f"{r['latency_ms_per_seq']:.2f}" for r in results],
    ))

    # Persistence / External flags
    lines.append(_row(
        "Persistent State",
        ["✓" if r["has_persistence"] else "✗" for r in results],
    ))
    lines.append(_row(
        "External Workspace",
        ["✓" if r["has_external"] else "✗" for r in results],
    ))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  MD    → {path}")
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_plots(results: list[dict], out_dir: Path) -> None:
    """Generate and save publication-quality comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("  [plots] matplotlib not available — skipping.")
        return

    colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b"]
    markers = ["o", "s", "^", "D"]
    labels = [r["label"] for r in results]

    n_turns = max(len(r["per_turn_accuracy"]) for r in results)
    turns = list(range(1, n_turns + 1))

    # ── Plot 1: Per-turn accuracy decay ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        accs = [a * 100 for a in r["per_turn_accuracy"]]
        ax.plot(
            turns[:len(accs)], accs,
            label=r["label"],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=2.0,
            markersize=7,
        )

    ax.set_xlabel("Sequence Turn", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Turn Accuracy across LIMINAL Architecture Stages",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(turns)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, n_turns + 0.2)
    fig.tight_layout()
    p1 = out_dir / "turn_accuracy_decay.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot  → {p1}")

    # ── Plot 2: Parameters vs. Accuracy vs. Latency scatter ────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        params = r["param_count"]["trainable"] / 1_000  # in K
        acc = r["overall_accuracy"] * 100
        lat = r["latency_ms_per_seq"]
        # Bubble size ~ latency
        size = max(80, lat * 60)
        sc = ax.scatter(
            params, acc,
            s=size,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=r["label"],
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
            zorder=3,
        )
        ax.annotate(
            r["model_id"],
            (params, acc),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=10,
            fontweight="bold",
            color=colors[i % len(colors)],
        )

    ax.set_xlabel("Trainable Parameters (K)", fontsize=12)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
    ax.set_title(
        "Accuracy vs. Model Complexity\n(bubble size ∝ inference latency)",
        fontsize=13, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "accuracy_vs_compute.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot  → {p2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    import torch
    from src.evaluation.phase8a_benchmark import Phase8Evaluator

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available — falling back to CPU.")
        device = "cpu"
    print(f"  Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        ("A", args.vector_ckpt),
        ("B", args.static_ckpt),
        ("C", args.phase4_ckpt),
        ("D", args.phase7_ckpt),
    ]

    # Validate checkpoint paths before starting
    missing = [(mid, p) for mid, p in checkpoints if not Path(p).exists()]
    if missing:
        print("\n  [WARNING] The following checkpoints were not found:")
        for mid, p in missing:
            print(f"    Model {mid}: {p}")
        print("  Evaluation will proceed for available checkpoints only.\n")

    print("\n" + "=" * 64)
    print("  Phase 8A: Four-Model Comparative Benchmark")
    print("=" * 64)

    evaluator = Phase8Evaluator(
        device=device,
        batch_size=args.batch_size,
        n_eval_sequences=args.n_sequences,
        n_turns=args.n_turns,
    )

    for model_id, ckpt_path in checkpoints:
        if Path(ckpt_path).exists():
            evaluator.register(model_id, ckpt_path)
        else:
            print(f"  Skipping Model {model_id} (checkpoint not found).")

    results = evaluator.run()

    if not results:
        print("\n  No results — check checkpoint paths.")
        return

    print("\n" + "=" * 64)
    print("  Exporting results …")
    print("=" * 64)

    _save_json(results, out_dir)
    _save_csv(results, out_dir)
    _save_markdown_table(results, out_dir)

    if not args.skip_plots:
        _save_plots(results, out_dir)

    # Final summary table to stdout
    print("\n" + "=" * 64)
    print("  ✓ Phase 8A Complete — Summary")
    print("=" * 64)
    print(f"  {'Model':<30}  {'Accuracy':>10}  {'Loss':>8}  {'Params':>10}  {'ms/seq':>8}")
    print("  " + "-" * 72)
    for r in results:
        print(
            f"  {r['label']:<30}  "
            f"{r['overall_accuracy']*100:>9.2f}%  "
            f"{r['mean_loss']:>8.4f}  "
            f"{r['param_count']['trainable']:>10,}  "
            f"{r['latency_ms_per_seq']:>7.2f}ms"
        )
    print()
    print(f"  All outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
