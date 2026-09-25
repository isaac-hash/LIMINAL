"""Phase 6 — Causal Intervention Engine.

Protocol
--------
1. Run the model up to reasoning step t₁ → capture (G_{t₁}, W₁).
2. Apply a targeted corruption to W₁ → W̃₁.
3. Continue from *identical* G_{t₁} with W̃₁, recording the new trajectory.
4. Compare the clean vs corrupted trajectories.

Corruption types (matching the implementation plan):
    swap_value       – Replace relevant record payload with a perturbed vector.
    delete_record    – Mark a relevant slot as empty (information loss).
    randomise_record – Overwrite payload with unit Gaussian noise (content destruction).
    irrelevant_swap  – Same as swap_value but on a slot the model doesn't rely on
                       (negative control).

Metrics:
    l2_trajectory_distance  – Mean L2 distance between clean and corrupted latent
                               trajectories.
    accuracy_delta          – Difference in classification accuracy (corrupted − clean).
    answer_kl               – KL divergence of softmax output distributions.
    iteration_delta         – Difference in mean effective reasoning steps.
    resolution_time_delta   – Difference in mean halt time (ACT-only; else 0.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.external_workspace import ExternalWorkspaceState


# ---------------------------------------------------------------------------
# Corruption types
# ---------------------------------------------------------------------------

class CorruptionType(Enum):
    SWAP_VALUE       = auto()   # Replace payload with a perturbed vector
    DELETE_RECORD    = auto()   # Mark slot as empty (mask=False)
    RANDOMISE_RECORD = auto()   # Overwrite payload with Gaussian noise
    IRRELEVANT_SWAP  = auto()   # SWAP_VALUE on an irrelevant slot (negative control)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InterventionConfig:
    """Parameters governing a single causal intervention experiment.

    Attributes
    ----------
    intervention_step:
        The reasoning step t₁ at which to snapshot and branch.
        Must be < the model's max_reasoning_steps.
    target_slot:
        Which external workspace slot (0-indexed) to corrupt.
    corruption_type:
        The corruption strategy to apply.
    swap_noise_std:
        Standard deviation of Gaussian noise added for SWAP_VALUE corruptions.
        new_value = original + N(0, swap_noise_std) when new_value_override is None.
    new_value_override:
        If provided, use this tensor directly as the corrupted payload.
        Shape must match [record_dim].
    randomise_std:
        Standard deviation for RANDOMISE_RECORD noise.
    """
    intervention_step: int = 2
    target_slot: int = 0
    corruption_type: CorruptionType = CorruptionType.SWAP_VALUE
    swap_noise_std: float = 2.0
    new_value_override: Tensor | None = None
    randomise_std: float = 1.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class InterventionResult:
    """Collected metrics for a single causal intervention run.

    Attributes
    ----------
    corruption_type:
        Which corruption was applied.
    target_slot:
        The external workspace slot index that was corrupted.
    l2_trajectory_distance:
        Mean L2 distance between clean and corrupted latent slot trajectories
        (averaged across batch, slots, and reasoning steps after t₁).
    accuracy_delta:
        Corrupted accuracy − clean accuracy (negative means corruption hurts).
    answer_kl:
        Mean KL(clean_softmax ‖ corrupted_softmax) across the batch.
    iteration_delta:
        Corrupted mean_steps − clean mean_steps.
    resolution_time_delta:
        Corrupted mean halt step − clean mean halt step (ACT only; else 0.0).
    clean_logits:
        Tensor[B, num_classes] clean output logits.
    corrupted_logits:
        Tensor[B, num_classes] corrupted output logits.
    clean_info:
        Full info dict from the clean continuation forward pass.
    corrupted_info:
        Full info dict from the corrupted continuation forward pass.
    """
    corruption_type: CorruptionType
    target_slot: int
    l2_trajectory_distance: float
    accuracy_delta: float
    answer_kl: float
    iteration_delta: float
    resolution_time_delta: float
    clean_logits: Tensor
    corrupted_logits: Tensor
    clean_info: dict[str, Any]
    corrupted_info: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (no tensors)."""
        return {
            "corruption_type": self.corruption_type.name,
            "target_slot": self.target_slot,
            "l2_trajectory_distance": round(self.l2_trajectory_distance, 6),
            "accuracy_delta": round(self.accuracy_delta, 6),
            "answer_kl": round(self.answer_kl, 6),
            "iteration_delta": round(self.iteration_delta, 6),
            "resolution_time_delta": round(self.resolution_time_delta, 6),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_corruption(
    ws: ExternalWorkspaceState,
    cfg: InterventionConfig,
    batch_idx: int = 0,
) -> ExternalWorkspaceState:
    """Apply the specified corruption to the workspace snapshot.

    All ExternalWorkspaceState corruption helpers return a new object and
    never mutate the original, so the clean branch is preserved.
    """
    slot = cfg.target_slot
    ct = cfg.corruption_type

    if ct in (CorruptionType.SWAP_VALUE, CorruptionType.IRRELEVANT_SWAP):
        if cfg.new_value_override is not None:
            new_val = cfg.new_value_override.to(ws.records.device)
        else:
            original = ws.records[batch_idx, slot].clone()
            noise = torch.randn_like(original) * cfg.swap_noise_std
            new_val = original + noise
        return ws.corrupt_swap_value(batch_idx, slot, new_val)

    elif ct == CorruptionType.DELETE_RECORD:
        return ws.corrupt_delete_record(batch_idx, slot)

    elif ct == CorruptionType.RANDOMISE_RECORD:
        return ws.corrupt_noise_record(batch_idx, slot, std=cfg.randomise_std)

    else:
        raise ValueError(f"Unknown CorruptionType: {ct}")


def _compute_trajectory_distance(
    clean_traj: list[Tensor],
    corrupt_traj: list[Tensor],
) -> float:
    """Mean L2 distance between two latent trajectory lists.

    Each trajectory entry is Tensor[B, N, d].  Only the overlapping suffix
    steps (after the intervention point) are compared.
    """
    n = min(len(clean_traj), len(corrupt_traj))
    if n == 0:
        return 0.0

    distances = []
    for c, k in zip(clean_traj[-n:], corrupt_traj[-n:]):
        dist = (c - k).norm(dim=-1).mean().item()
        distances.append(dist)
    return float(sum(distances) / len(distances))


def _compute_answer_kl(
    clean_logits: Tensor,
    corrupt_logits: Tensor,
) -> float:
    """Mean KL(clean ‖ corrupted) over the batch.

    Uses log-space softmax for numerical stability.
    """
    log_p = F.log_softmax(clean_logits.detach().float(), dim=-1)
    log_q = F.log_softmax(corrupt_logits.detach().float(), dim=-1)
    p = log_p.exp()
    kl_per_example = (p * (log_p - log_q)).sum(dim=-1)  # [B]
    return float(kl_per_example.mean().item())


def _accuracy_from_logits(logits: Tensor, labels: Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


# ---------------------------------------------------------------------------
# Main intervention runner
# ---------------------------------------------------------------------------

def run_causal_intervention(
    *,
    model_forward_fn: Callable[..., tuple[Tensor, dict[str, Any]]],
    decoder_fn: Callable[[Tensor], Tensor],
    V_snapshot: Tensor,
    E_snapshot: Tensor | None,
    workspace_snapshot: ExternalWorkspaceState,
    labels: Tensor,
    intervention_cfg: InterventionConfig,
) -> InterventionResult:
    """Execute the Phase 6 causal intervention protocol for one corruption type.

    Parameters
    ----------
    model_forward_fn:
        Callable(V_0, E_0, workspace) -> (h_final, info).
        Should wrap LatentWorkspace.forward so the continuation starts from
        the provided snapshot state rather than re-running the full episode.
    decoder_fn:
        Callable(h_final [B, d]) -> logits [B, num_classes].
    V_snapshot:
        Tensor[B, N, d] — latent slot state at step t₁.
    E_snapshot:
        Tensor[B, N, N, d_e] — edge tensor at t₁, or None.
    workspace_snapshot:
        ExternalWorkspaceState at t₁ (detached, already cloned).
    labels:
        Tensor[B] — ground-truth class labels.
    intervention_cfg:
        Configuration controlling which slot/corruption to apply.

    Returns
    -------
    InterventionResult with all measured divergence metrics.
    """
    # ── Clean continuation ───────────────────────────────────────────────────
    with torch.no_grad():
        h_clean, info_clean = model_forward_fn(
            V_snapshot.clone(),
            E_snapshot.clone() if E_snapshot is not None else None,
            workspace_snapshot.clone(detach=True),
        )
        logits_clean = decoder_fn(h_clean)

    # ── Corrupted continuation ───────────────────────────────────────────────
    ws_corrupted = _apply_corruption(
        workspace_snapshot.clone(detach=True),
        intervention_cfg,
        batch_idx=0,
    )

    with torch.no_grad():
        h_corrupt, info_corrupt = model_forward_fn(
            V_snapshot.clone(),
            E_snapshot.clone() if E_snapshot is not None else None,
            ws_corrupted,
        )
        logits_corrupt = decoder_fn(h_corrupt)

    # ── Metrics ──────────────────────────────────────────────────────────────
    traj_clean   = info_clean.get("trajectory", [])
    traj_corrupt = info_corrupt.get("trajectory", [])
    l2_dist = _compute_trajectory_distance(traj_clean, traj_corrupt)

    acc_clean   = _accuracy_from_logits(logits_clean, labels)
    acc_corrupt = _accuracy_from_logits(logits_corrupt, labels)
    acc_delta   = acc_corrupt - acc_clean

    kl = _compute_answer_kl(logits_clean, logits_corrupt)

    steps_clean   = float(info_clean.get("effective_steps", 0.0))
    steps_corrupt = float(info_corrupt.get("effective_steps", 0.0))
    iteration_delta = steps_corrupt - steps_clean

    # Resolution time delta (ACT ponder-weight-based halt step)
    resolution_delta = 0.0
    pw_clean   = info_clean.get("ponder_weights")
    pw_corrupt = info_corrupt.get("ponder_weights")
    if pw_clean is not None and pw_corrupt is not None:
        T_c = pw_clean.shape[1]
        T_k = pw_corrupt.shape[1]
        device = pw_clean.device
        idx_c = torch.arange(1, T_c + 1, device=device, dtype=torch.float)
        idx_k = torch.arange(1, T_k + 1, device=device, dtype=torch.float)
        halt_c = (pw_clean.float() * idx_c.unsqueeze(0)).sum(dim=1).mean().item()
        halt_k = (pw_corrupt.float() * idx_k.unsqueeze(0)).sum(dim=1).mean().item()
        resolution_delta = halt_k - halt_c

    return InterventionResult(
        corruption_type=intervention_cfg.corruption_type,
        target_slot=intervention_cfg.target_slot,
        l2_trajectory_distance=l2_dist,
        accuracy_delta=acc_delta,
        answer_kl=kl,
        iteration_delta=iteration_delta,
        resolution_time_delta=resolution_delta,
        clean_logits=logits_clean,
        corrupted_logits=logits_corrupt,
        clean_info=info_clean,
        corrupted_info=info_corrupt,
    )


# ---------------------------------------------------------------------------
# Batch sweep — all four corruption types
# ---------------------------------------------------------------------------

def run_all_corruption_types(
    *,
    model_forward_fn: Callable[..., tuple[Tensor, dict[str, Any]]],
    decoder_fn: Callable[[Tensor], Tensor],
    V_snapshot: Tensor,
    E_snapshot: Tensor | None,
    workspace_snapshot: ExternalWorkspaceState,
    labels: Tensor,
    relevant_slot: int,
    irrelevant_slot: int,
    base_intervention_step: int = 2,
    swap_noise_std: float = 2.0,
    randomise_std: float = 1.0,
) -> list[InterventionResult]:
    """Sweep all four corruption types for a single snapshot.

    Parameters
    ----------
    relevant_slot:
        Slot index containing task-critical information (positive conditions).
    irrelevant_slot:
        Slot index that does not affect the task answer (negative control).

    Returns
    -------
    List of four InterventionResult objects in the order:
    [SWAP_VALUE, DELETE_RECORD, RANDOMISE_RECORD, IRRELEVANT_SWAP].
    """
    corruption_configs = [
        InterventionConfig(
            intervention_step=base_intervention_step,
            target_slot=relevant_slot,
            corruption_type=CorruptionType.SWAP_VALUE,
            swap_noise_std=swap_noise_std,
        ),
        InterventionConfig(
            intervention_step=base_intervention_step,
            target_slot=relevant_slot,
            corruption_type=CorruptionType.DELETE_RECORD,
        ),
        InterventionConfig(
            intervention_step=base_intervention_step,
            target_slot=relevant_slot,
            corruption_type=CorruptionType.RANDOMISE_RECORD,
            randomise_std=randomise_std,
        ),
        InterventionConfig(
            intervention_step=base_intervention_step,
            target_slot=irrelevant_slot,
            corruption_type=CorruptionType.IRRELEVANT_SWAP,
            swap_noise_std=swap_noise_std,
        ),
    ]

    results: list[InterventionResult] = []
    for cfg in corruption_configs:
        result = run_causal_intervention(
            model_forward_fn=model_forward_fn,
            decoder_fn=decoder_fn,
            V_snapshot=V_snapshot,
            E_snapshot=E_snapshot,
            workspace_snapshot=workspace_snapshot,
            labels=labels,
            intervention_cfg=cfg,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_intervention_summary(results: list[InterventionResult]) -> None:
    """Print a formatted table of intervention results with an exit-criterion check."""
    header = (
        f"{'Corruption':<22} {'Slot':>4} "
        f"{'L2 dist':>10} {'Acc Δ':>8} {'KL':>8} "
        f"{'Step Δ':>8} {'Halt Δ':>8}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r.corruption_type.name:<22} {r.target_slot:>4} "
            f"{r.l2_trajectory_distance:>10.4f} {r.accuracy_delta:>+8.4f} "
            f"{r.answer_kl:>8.4f} {r.iteration_delta:>+8.3f} "
            f"{r.resolution_time_delta:>+8.3f}"
        )
    print(sep)

    relevant   = [r for r in results if r.corruption_type != CorruptionType.IRRELEVANT_SWAP]
    irrelevant = [r for r in results if r.corruption_type == CorruptionType.IRRELEVANT_SWAP]

    mean_rel_l2 = sum(r.l2_trajectory_distance for r in relevant) / len(relevant) if relevant else 0.0
    mean_irr_l2 = sum(r.l2_trajectory_distance for r in irrelevant) / len(irrelevant) if irrelevant else 0.0

    print(f"\nMean L2 (relevant corruptions) : {mean_rel_l2:.4f}")
    print(f"Mean L2 (irrelevant control)   : {mean_irr_l2:.4f}")
    if mean_irr_l2 > 0:
        ratio = mean_rel_l2 / mean_irr_l2
        print(f"Relevant / Irrelevant L2 ratio : {ratio:.2f}x")
        if ratio > 2.0:
            print("✅  Phase 6 exit criterion MET — relevant corruptions have "
                  "significantly larger downstream effect than the irrelevant control.")
        else:
            print("⚠️  Ratio < 2× — effect weak; verify slot selection or increase "
                  "swap_noise_std.")
    else:
        print("(Irrelevant L2 = 0 — negative control produced no change, which is ideal.)")
    print()
