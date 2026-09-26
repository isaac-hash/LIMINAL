"""Phase 7 — Selectivity Analysis for Learned Write Gate.

Measures how selectively the model uses its external workspace write gate
when trained with LearnedWriteController vs. the deterministic top-K policy.

Key metrics
-----------
mean_slots_written_per_step : float
    Mean number of latent slots that fired (gate=1) per reasoning step.
    Lower = more selective.  For top-K, this always equals write_top_k.
    For learned gate, this should be < write_top_k on easy turns.

selectivity_score : float
    1 - (mean_slots_written / num_latent_slots).
    0 = all slots always write.  1 = no slots ever write.

per_turn_gate_fraction : list[float]
    Fraction of slots that fired, broken down by sequence turn index.
    Reveals whether the model learns to write less aggressively in early
    turns (when there is less to externalise) and more in later turns.

gate_entropy : float
    Binary entropy of the gate distribution over slots.
    Low entropy => confident selective decisions.
    High entropy => uncertain / diffuse writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def compute_gate_stats(
    gate_hard: Tensor,
) -> dict[str, float]:
    """Compute selectivity statistics from a hard write gate tensor.

    Args:
        gate_hard: Tensor[B, N] binary gate values (0 or 1) for one step.

    Returns:
        dict with keys:
            'mean_slots_written'  — mean number of slots that fired.
            'fraction_written'    — mean fraction of slots that fired [0,1].
            'gate_entropy'        — mean binary entropy of gate per slot.
    """
    B, N = gate_hard.shape
    slots_written = gate_hard.sum(dim=1)       # [B]
    mean_slots = slots_written.mean().item()
    fraction = mean_slots / max(N, 1)

    # Binary entropy: H(p) = -p log p - (1-p) log(1-p), mean over slots
    p = gate_hard.mean(dim=0).clamp(1e-6, 1 - 1e-6)  # [N]
    entropy = (-p * p.log() - (1 - p) * (1 - p).log()).mean().item()

    return {
        "mean_slots_written": mean_slots,
        "fraction_written": fraction,
        "gate_entropy": entropy,
    }


class SelectivityAnalyser:
    """Accumulates per-step gate statistics across a full evaluation run.

    Usage::

        analyser = SelectivityAnalyser(num_latent_slots=8, write_top_k=3)
        for batch in eval_loader:
            # ... forward pass ...
            for t, info_t in enumerate(all_infos):
                gate = info_t.get("write_gate_hard")  # [B, N] or None
                if gate is not None:
                    analyser.update(gate, turn_index=batch_turn_index, step=t)
        report = analyser.summarise()

    Args:
        num_latent_slots: N — total number of latent slots.
        write_top_k:      K — baseline for deterministic top-K (for comparison).
    """

    def __init__(self, num_latent_slots: int, write_top_k: int = 3) -> None:
        self.N = num_latent_slots
        self.write_top_k = write_top_k
        self._steps: list[dict[str, Any]] = []
        self._per_turn: dict[int, list[float]] = {}

    def update(
        self,
        gate_hard: Tensor,
        turn_index: int,
        step: int,
    ) -> None:
        """Record gate statistics for one reasoning step of one batch.

        Args:
            gate_hard:   Tensor[B, N] binary gate (0/1) for the current step.
            turn_index:  Sequence turn index (0-indexed).
            step:        Reasoning step index within the turn (0-indexed).
        """
        stats = compute_gate_stats(gate_hard)
        stats["turn_index"] = turn_index
        stats["step"] = step
        self._steps.append(stats)

        if turn_index not in self._per_turn:
            self._per_turn[turn_index] = []
        self._per_turn[turn_index].append(stats["fraction_written"])

    def summarise(self) -> dict[str, Any]:
        """Compute aggregate selectivity metrics over all recorded steps.

        Returns:
            dict with keys:
                'mean_slots_written_per_step' float
                'selectivity_score'           float  (1 - fraction_written)
                'gate_entropy'                float
                'top_k_baseline_slots'        int    (always write_top_k)
                'per_turn_gate_fraction'      dict[int, float]
                'num_steps_recorded'          int
        """
        if not self._steps:
            return {
                "mean_slots_written_per_step": 0.0,
                "selectivity_score": 0.0,
                "gate_entropy": 0.0,
                "top_k_baseline_slots": self.write_top_k,
                "per_turn_gate_fraction": {},
                "num_steps_recorded": 0,
            }

        mean_slots = sum(s["mean_slots_written"] for s in self._steps) / len(self._steps)
        fraction = mean_slots / max(self.N, 1)
        entropy = sum(s["gate_entropy"] for s in self._steps) / len(self._steps)

        per_turn = {
            turn: sum(fracs) / len(fracs)
            for turn, fracs in sorted(self._per_turn.items())
        }

        return {
            "mean_slots_written_per_step": mean_slots,
            "selectivity_score": 1.0 - fraction,
            "gate_entropy": entropy,
            "top_k_baseline_slots": float(self.write_top_k),
            "per_turn_gate_fraction": per_turn,
            "num_steps_recorded": len(self._steps),
        }

    def print_summary(self) -> None:
        """Print a human-readable selectivity report to stdout."""
        report = self.summarise()
        print("\n\u2500\u2500\u2500 Phase 7 Selectivity Analysis \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
        print(f"  Latent slots (N)          : {self.N}")
        print(f"  Top-K baseline            : {self.write_top_k:.0f} slots/step")
        print(f"  Learned mean slots/step   : {report['mean_slots_written_per_step']:.2f}")
        print(f"  Selectivity score         : {report['selectivity_score']:.3f}  (higher = more selective)")
        print(f"  Gate entropy              : {report['gate_entropy']:.4f}  (lower = more confident)")
        print(f"  Steps recorded            : {report['num_steps_recorded']}")
        if report["per_turn_gate_fraction"]:
            print("  Gate fraction by turn     :")
            for turn, frac in report["per_turn_gate_fraction"].items():
                bar = "#" * int(frac * 20)
                print(f"    Turn {turn}: {frac:.3f}  |{bar:<20}|")
        print()


def save_selectivity_report(
    report: dict[str, Any],
    output_dir: str | Path,
    filename: str = "selectivity_report.json",
) -> Path:
    """Write a selectivity report dict to JSON.

    Args:
        report:     Output of SelectivityAnalyser.summarise().
        output_dir: Directory to write the file into.
        filename:   JSON filename.

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path
