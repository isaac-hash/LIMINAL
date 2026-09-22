"""Phase 5 — Workspace Analysis: diagnostic inspection of external workspace dynamics.

All functions accept an info dict returned by LatentWorkspace.forward() and
extract meaningful summaries of how the external scratchpad was used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.models.external_workspace import ExternalWorkspaceState, ExternalRecordType


def compute_workspace_occupancy(info: dict[str, Any]) -> float:
    """Mean fraction of external workspace slots occupied across all steps.

    Args:
        info: Info dict from LatentWorkspace forward pass.

    Returns:
        occupancy: float in [0, 1]. 0 if no workspace trajectory exists.
    """
    ws_traj: list[ExternalWorkspaceState] = info.get("workspace_trajectory", [])
    if not ws_traj:
        return 0.0

    occupancies = []
    for ws in ws_traj:
        # mask: [B, M], mean over batch and slots
        occupancies.append(ws.mask.float().mean().item())

    return float(sum(occupancies) / len(occupancies))


def compute_type_distribution(info: dict[str, Any]) -> dict[str, float]:
    """Fraction of occupied slots assigned to each record type, averaged over all steps.

    Args:
        info: Info dict from LatentWorkspace forward pass.

    Returns:
        dist: dict mapping type name → fraction (sums to 1.0 for occupied slots).
    """
    ws_traj: list[ExternalWorkspaceState] = info.get("workspace_trajectory", [])
    if not ws_traj:
        return {}

    type_counts: dict[str, float] = {t.name: 0.0 for t in ExternalRecordType if t != ExternalRecordType.EMPTY}
    total_occupied = 0.0

    for ws in ws_traj:
        occupied = ws.mask  # [B, M]
        types = ws.types    # [B, M]
        for t in ExternalRecordType:
            if t == ExternalRecordType.EMPTY:
                continue
            count = float(((types == t.value) & occupied).float().sum().item())
            type_counts[t.name] += count
            total_occupied += count

    if total_occupied == 0.0:
        return type_counts

    return {k: v / total_occupied for k, v in type_counts.items()}


def extract_read_write_heatmap(info: dict[str, Any]) -> dict[str, Tensor]:
    """Build heatmaps of read attention weights and write patterns over steps.

    Returns:
        dict with:
            "read_weights":  Tensor[T, num_heads, M] — attention per step (batch item 0)
            "write_mask":    Tensor[T, M] — occupied mask per step (batch item 0)
    """
    ws_traj: list[ExternalWorkspaceState] = info.get("workspace_trajectory", [])
    rw_traj: list[Tensor] = info.get("read_weights_trajectory", [])

    result: dict[str, Tensor] = {}

    if rw_traj:
        # rw_traj: list of [B, num_heads, M] → stack to [T, num_heads, M] for batch item 0
        result["read_weights"] = torch.stack([rw[0] for rw in rw_traj], dim=0)

    if ws_traj:
        # masks: [T, M] for batch item 0
        result["write_mask"] = torch.stack([ws.mask[0].float() for ws in ws_traj], dim=0)

    return result


def export_workspace_trajectory_json(
    info: dict[str, Any],
    output_path: str | Path,
    batch_idx: int = 0,
) -> None:
    """Export step-by-step workspace trajectory to a human-readable JSON file.

    Args:
        info:        Info dict from LatentWorkspace forward pass.
        output_path: Where to write the JSON file.
        batch_idx:   Which batch item to export (default 0).
    """
    ws_traj: list[ExternalWorkspaceState] = info.get("workspace_trajectory", [])
    rw_traj: list[Tensor] = info.get("read_weights_trajectory", [])

    steps = []
    for t, ws in enumerate(ws_traj):
        step_data = ws.to_dict(step=t)

        # Add read attention weights if available
        if t < len(rw_traj):
            rw = rw_traj[t][batch_idx]  # [num_heads, M]
            step_data["read_attention"] = [
                {
                    "head": h,
                    "weights": [round(float(w), 6) for w in rw[h].tolist()],
                }
                for h in range(rw.shape[0])
            ]

        steps.append(step_data)

    output = {
        "num_steps": len(steps),
        "trajectory": steps,
    }

    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
