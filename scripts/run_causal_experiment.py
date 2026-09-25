"""Phase 6 — Causal Intervention Experiment Runner.

Usage (from repo root):
    python scripts/run_causal_experiment.py \\
        --config configs/experiments/causal_intervention.yaml \\
        --checkpoint results/external_ablation/best.pt \\
        --output results/causal_intervention/

What this script does
---------------------
1.  Loads the trained Phase 5 Model F from a checkpoint.
2.  Generates a held-out evaluation batch (or loads from an existing dataset).
3.  Runs a full forward pass to obtain the step-by-step workspace trajectory.
4.  At step t₁ (configurable), snapshots (V_{t₁}, E_{t₁}, W_{t₁}).
5.  For each of the four corruption types, resumes the model from the
    identical latent state (V_{t₁}) with a corrupted workspace (W̃_{t₁}).
6.  Measures and reports:
      - L2 trajectory distance
      - Accuracy delta
      - KL divergence of output distributions
      - Iteration delta
      - Resolution-time delta
7.  Saves per-corruption JSON results and a summary CSV.

Exit criterion (Level 3 ★):
    Relevant corruptions  → significant downstream changes (L2 >> 0, |Acc Δ| > 5%)
    Irrelevant corruption → no significant effect (L2 ≈ 0, |Acc Δ| ≈ 0)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

# Ensure src/ is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.device import print_hardware_info
from src.utils.config import load_config, set_seed
from src.utils.checkpoint import load_checkpoint
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary
from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
from src.models.sequential_model import SequentialReasoningModel
from src.evaluation.causal_intervention import (
    CorruptionType,
    InterventionConfig,
    run_all_corruption_types,
    print_intervention_summary,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LIMINAL Phase 6 — Causal Intervention Experiment"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/experiments/causal_intervention.yaml",
        help="Path to the causal intervention YAML config.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/external_ablation/best.pt",
        help="Path to a trained Phase 5 checkpoint (best.pt).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/causal_intervention",
        help="Directory to write JSON results and summary CSV.",
    )
    p.add_argument(
        "--intervention-step",
        type=int,
        default=2,
        help="Reasoning step t₁ at which to snapshot and branch.",
    )
    p.add_argument(
        "--relevant-slot",
        type=int,
        default=0,
        help="External workspace slot index considered task-relevant.",
    )
    p.add_argument(
        "--irrelevant-slot",
        type=int,
        default=15,
        help="External workspace slot index considered task-irrelevant (negative control).",
    )
    p.add_argument(
        "--swap-noise-std",
        type=float,
        default=2.0,
        help="Std-dev of Gaussian noise for SWAP_VALUE corruptions.",
    )
    p.add_argument(
        "--randomise-std",
        type=float,
        default=1.0,
        help="Std-dev for RANDOMISE_RECORD corruptions.",
    )
    p.add_argument(
        "--num-eval-batches",
        type=int,
        default=50,
        help="Number of evaluation batches to average results over.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    p.add_argument(
        "--drive-path",
        type=str,
        default=None,
        help="Optional Google Drive path to copy results to.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_occupied_slots_batch0(W_at_step: list, step: int) -> list[int]:
    """Return indices of occupied external workspace slots at a given step (batch item 0)."""
    if step >= len(W_at_step) or W_at_step[step] is None:
        return []
    ws = W_at_step[step]
    mask = ws.mask[0]  # [M]
    return mask.nonzero(as_tuple=False).squeeze(-1).tolist()


def _maybe_copy_to_drive(src: Path, drive_path: str | None) -> None:
    if drive_path is None:
        return
    import shutil
    dst = Path(drive_path) / src.name
    shutil.copy2(src, dst)
    print(f"  Copied to Drive: {dst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = print_hardware_info()

    # ── Config ───────────────────────────────────────────────────────────────
    base_cfg_path = Path("configs/base.yaml")
    cfg_path = Path(args.config)

    if cfg_path != base_cfg_path and cfg_path.exists():
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = load_config(base_cfg_path, overrides=overrides)
    else:
        config = load_config(base_cfg_path)

    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n─── Phase 6: Causal Intervention Experiment ───────────────────────")
    print(f"  Checkpoint       : {args.checkpoint}")
    print(f"  Config           : {args.config}")
    print(f"  Intervention step: t₁ = {args.intervention_step}")
    print(f"  Relevant slot    : {args.relevant_slot}")
    print(f"  Irrelevant slot  : {args.irrelevant_slot} (negative control)")
    print(f"  Swap noise std   : {args.swap_noise_std}")
    print(f"  Randomise std    : {args.randomise_std}")
    print(f"  Eval batches     : {args.num_eval_batches}")
    print(f"  Output dir       : {output_dir}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Generating evaluation dataset...")
    generator = ArithmeticGenerator(config=config.data, seed=seed)
    splits = generator.generate_dataset()

    vocab = Vocabulary()
    vocab.build_from_records(
        splits["train"] + splits["val"] + splits["test"]
    )

    from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
    test_dataset = SequenceReasoningDataset(splits["test"], vocab)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Building model...")
    model = SequentialReasoningModel(
        config=config.model,
        vocab=vocab,
        activity_config=config.activity,
        resolution_config=config.resolution,
        persistence_config=config.persistence,
        external_config=config.external,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        epoch, _metrics, _cfg = load_checkpoint(str(ckpt_path), model=model, device=device)
        print(f"  Loaded weights from {ckpt_path} (epoch {epoch})\n")
    else:
        print(f"  ⚠️  Checkpoint not found at {ckpt_path}. Running with random weights.\n")

    model.eval()
    workspace_model = model.base_model.workspace
    decoder_fn = model.base_model.decoder

    # ── Intervention sweep ───────────────────────────────────────────────────

    # Accumulators for aggregated metrics
    agg: dict[str, list[float]] = {
        ct.name + "_l2":   [] for ct in CorruptionType
    }
    agg.update({ct.name + "_acc_delta": [] for ct in CorruptionType})
    agg.update({ct.name + "_kl":        [] for ct in CorruptionType})
    agg.update({ct.name + "_step_delta":[] for ct in CorruptionType})
    agg.update({ct.name + "_halt_delta":[] for ct in CorruptionType})

    batch_results: list[dict] = []
    n_batches_done = 0

    print(f"Running {args.num_eval_batches} evaluation batches...\n")

    with torch.no_grad():
        for batch in test_loader:
            if n_batches_done >= args.num_eval_batches:
                break

            facts_seq     = batch["facts"].to(device)
            fact_mask_seq = batch["fact_mask"].to(device)
            turn_mask     = batch["turn_mask"].to(device)
            labels_seq    = batch["labels"].to(device)  # [B, T_turns]

            # We evaluate on the final turn's labels for the intervention
            # (the turn where all facts have been accumulated)
            B = facts_seq.shape[0]
            last_turn = facts_seq.shape[1] - 1
            labels = labels_seq[:, last_turn]  # [B]

            # Full forward pass to get the step-by-step trajectory
            stacked_logits, all_infos = model(facts_seq, fact_mask_seq, turn_mask)
            last_info = all_infos[last_turn]

            # Extract per-step snapshots emitted by the workspace
            V_at_step: list = last_info.get("V_at_step", [])
            E_at_step: list = last_info.get("E_at_step", [])
            W_at_step: list = last_info.get("W_at_step", [])

            t1 = args.intervention_step
            if t1 >= len(V_at_step):
                t1 = max(0, len(V_at_step) - 1)

            V_snap = V_at_step[t1]
            E_snap = E_at_step[t1]
            W_snap = W_at_step[t1]

            if W_snap is None:
                # External workspace not active; skip this batch for now
                continue

            # Auto-select occupied relevant slot if possible
            occupied = _find_occupied_slots_batch0(W_at_step, t1)
            relevant_slot = args.relevant_slot
            irrelevant_slot = args.irrelevant_slot
            # Clamp to valid range
            M = W_snap.records.shape[1]
            relevant_slot   = min(relevant_slot,   M - 1)
            irrelevant_slot = min(irrelevant_slot, M - 1)

            # Define the continuation function: workspace model starting from snapshot
            def _forward_fn(
                V_0: torch.Tensor,
                E_0: torch.Tensor | None,
                workspace_0,
                _ws_model=workspace_model,
            ):
                return _ws_model(V_0, E_0=E_0, workspace=workspace_0)

            results = run_all_corruption_types(
                model_forward_fn=_forward_fn,
                decoder_fn=decoder_fn,
                V_snapshot=V_snap,
                E_snapshot=E_snap,
                workspace_snapshot=W_snap,
                labels=labels,
                relevant_slot=relevant_slot,
                irrelevant_slot=irrelevant_slot,
                base_intervention_step=t1,
                swap_noise_std=args.swap_noise_std,
                randomise_std=args.randomise_std,
            )

            batch_summary = {"batch": n_batches_done, "t1": t1, "corruptions": []}
            for r in results:
                d = r.as_dict()
                batch_summary["corruptions"].append(d)
                ct_name = r.corruption_type.name
                agg[ct_name + "_l2"].append(r.l2_trajectory_distance)
                agg[ct_name + "_acc_delta"].append(r.accuracy_delta)
                agg[ct_name + "_kl"].append(r.answer_kl)
                agg[ct_name + "_step_delta"].append(r.iteration_delta)
                agg[ct_name + "_halt_delta"].append(r.resolution_time_delta)

            batch_results.append(batch_summary)
            n_batches_done += 1

            if n_batches_done % 10 == 0:
                print(f"  ... {n_batches_done}/{args.num_eval_batches} batches done")

    if n_batches_done == 0:
        print("⚠️  No batches processed — check that the external workspace is enabled "
              "in the config and that the checkpoint has a trained workspace.")
        return

    # ── Aggregate ────────────────────────────────────────────────────────────

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    aggregated: dict[str, dict] = {}
    for ct in CorruptionType:
        name = ct.name
        aggregated[name] = {
            "l2_trajectory_distance": _mean(agg[name + "_l2"]),
            "accuracy_delta":         _mean(agg[name + "_acc_delta"]),
            "answer_kl":              _mean(agg[name + "_kl"]),
            "iteration_delta":        _mean(agg[name + "_step_delta"]),
            "resolution_time_delta":  _mean(agg[name + "_halt_delta"]),
        }

    # ── Console summary ───────────────────────────────────────────────────────

    print(f"\n─── Aggregated results over {n_batches_done} batches ───────────────────\n")

    # Build fake InterventionResult objects for the printer
    from src.evaluation.causal_intervention import InterventionResult
    fake_results = []
    for ct in CorruptionType:
        d = aggregated[ct.name]
        r = InterventionResult(
            corruption_type=ct,
            target_slot=args.relevant_slot if ct != CorruptionType.IRRELEVANT_SWAP else args.irrelevant_slot,
            l2_trajectory_distance=d["l2_trajectory_distance"],
            accuracy_delta=d["accuracy_delta"],
            answer_kl=d["answer_kl"],
            iteration_delta=d["iteration_delta"],
            resolution_time_delta=d["resolution_time_delta"],
            clean_logits=torch.zeros(1),
            corrupted_logits=torch.zeros(1),
            clean_info={},
            corrupted_info={},
        )
        fake_results.append(r)

    print_intervention_summary(fake_results)

    # ── Exit criterion check ─────────────────────────────────────────────────

    relevant_l2_vals = [
        aggregated[ct.name]["l2_trajectory_distance"]
        for ct in CorruptionType
        if ct != CorruptionType.IRRELEVANT_SWAP
    ]
    irr_l2 = aggregated[CorruptionType.IRRELEVANT_SWAP.name]["l2_trajectory_distance"]
    mean_rel_l2 = _mean(relevant_l2_vals)

    print("─── Level 3 Exit Criterion Check ─────────────────────────────────")
    relevant_acc_deltas = [
        abs(aggregated[ct.name]["accuracy_delta"])
        for ct in CorruptionType
        if ct != CorruptionType.IRRELEVANT_SWAP
    ]
    mean_rel_acc_delta = _mean(relevant_acc_deltas)
    irr_acc_delta = abs(aggregated[CorruptionType.IRRELEVANT_SWAP.name]["accuracy_delta"])

    l2_ratio = mean_rel_l2 / irr_l2 if irr_l2 > 1e-8 else float("inf")

    print(f"  Mean relevant corruption L2    : {mean_rel_l2:.4f}")
    print(f"  Irrelevant control L2          : {irr_l2:.4f}")
    print(f"  Relevant / Irrelevant L2 ratio : {l2_ratio:.2f}×")
    print(f"  Mean |Acc Δ| relevant          : {mean_rel_acc_delta:.4f}")
    print(f"  |Acc Δ| irrelevant             : {irr_acc_delta:.4f}")

    l2_pass  = l2_ratio > 2.0
    acc_pass = mean_rel_acc_delta > 0.05 and irr_acc_delta < 0.05

    if l2_pass and acc_pass:
        print("\n✅  Level 3 ACHIEVED — External workspace causally affects reasoning.")
    elif l2_pass:
        print("\n⚠️  L2 criterion met, accuracy delta borderline — check task difficulty.")
    else:
        print("\n❌  Level 3 NOT YET met — consider increasing noise std or verifying "
              "slot selection.")

    # ── Save results ──────────────────────────────────────────────────────────

    # Full per-batch JSON
    batches_path = output_dir / "causal_experiment_batches.json"
    batches_path.write_text(json.dumps(batch_results, indent=2), encoding="utf-8")
    print(f"\nPer-batch results → {batches_path}")

    # Aggregated JSON
    agg_path = output_dir / "causal_experiment_aggregated.json"
    agg_payload = {
        "num_batches": n_batches_done,
        "intervention_step": args.intervention_step,
        "relevant_slot": args.relevant_slot,
        "irrelevant_slot": args.irrelevant_slot,
        "aggregated": {
            ct.name: aggregated[ct.name] for ct in CorruptionType
        },
        "exit_criterion": {
            "l2_ratio": l2_ratio,
            "mean_rel_acc_delta": mean_rel_acc_delta,
            "irr_acc_delta": irr_acc_delta,
            "l2_criterion_met": l2_pass,
            "acc_criterion_met": acc_pass,
            "level_3_achieved": l2_pass and acc_pass,
        },
    }
    agg_path.write_text(json.dumps(agg_payload, indent=2), encoding="utf-8")
    print(f"Aggregated results → {agg_path}")

    # Summary CSV
    csv_path = output_dir / "causal_experiment_summary.csv"
    csv_fields = [
        "corruption_type", "l2_trajectory_distance", "accuracy_delta",
        "answer_kl", "iteration_delta", "resolution_time_delta",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for ct in CorruptionType:
            row = {"corruption_type": ct.name}
            row.update(aggregated[ct.name])
            writer.writerow(row)
    print(f"Summary CSV        → {csv_path}")

    # Optional Drive copy
    for fpath in [batches_path, agg_path, csv_path]:
        _maybe_copy_to_drive(fpath, args.drive_path)

    print("\nPhase 6 causal intervention experiment complete.\n")


if __name__ == "__main__":
    main()
