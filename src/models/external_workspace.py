"""Phase 5 — External Workspace: structured, inspectable scratchpad.

ExternalWorkspaceState is a pure data container (not an nn.Module) holding
M typed record slots that accompany each forward pass through the latent graph.

Anatomy of a record slot:
    records[b, m]      : Tensor[d_ext]   — continuous payload vector
    types[b, m]        : int              — ExternalRecordType ID
    mask[b, m]         : bool             — True = occupied, False = empty
    step_written[b, m] : int              — reasoning step at which slot was last written (LRU key)
    wrote_this_pass[b, m] : bool         — deduplication flag; reset at start of each forward pass
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------

class ExternalRecordType(IntEnum):
    EMPTY      = 0
    ENTITY     = 1
    ATTRIBUTE  = 2
    RELATION   = 3
    OPERATION  = 4
    CONSTRAINT = 5
    STATUS     = 6


_TYPE_NAMES = {t.value: t.name for t in ExternalRecordType}


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

@dataclass
class ExternalWorkspaceState:
    """Mutable data container for a batch of external workspace scratchpads.

    All tensors share the same batch dimension B and slot dimension M.

    Attributes:
        records:         Tensor[B, M, d_ext]  — continuous payload representations
        types:           Tensor[B, M]  (Long) — ExternalRecordType IDs
        mask:            Tensor[B, M]  (Bool) — True = occupied slot
        step_written:    Tensor[B, M]  (Long) — reasoning step of last write (LRU key)
        wrote_this_pass: Tensor[B, M]  (Bool) — dedup flag, reset per forward pass
    """

    records: Tensor
    types: Tensor
    mask: Tensor
    step_written: Tensor
    wrote_this_pass: Tensor

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def init_empty(
        cls,
        batch_size: int,
        num_slots: int,
        record_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "ExternalWorkspaceState":
        """Create an all-empty scratchpad for a fresh forward pass."""
        return cls(
            records=torch.zeros(batch_size, num_slots, record_dim, device=device, dtype=dtype),
            types=torch.zeros(batch_size, num_slots, device=device, dtype=torch.long),
            mask=torch.zeros(batch_size, num_slots, device=device, dtype=torch.bool),
            step_written=torch.full((batch_size, num_slots), -1, device=device, dtype=torch.long),
            wrote_this_pass=torch.zeros(batch_size, num_slots, device=device, dtype=torch.bool),
        )

    # ------------------------------------------------------------------
    # Cloning & detachment
    # ------------------------------------------------------------------

    def clone(self, detach: bool = False) -> "ExternalWorkspaceState":
        """Return a copy.  detach=True stops gradients (use between sequence turns)."""
        def _c(t: Tensor) -> Tensor:
            c = t.clone()
            return c.detach() if detach else c

        return ExternalWorkspaceState(
            records=_c(self.records),
            types=self.types.clone(),
            mask=self.mask.clone(),
            step_written=self.step_written.clone(),
            wrote_this_pass=self.wrote_this_pass.clone(),
        )

    def reset_pass_flags(self) -> None:
        """Clear the per-pass deduplication flags in-place."""
        self.wrote_this_pass.zero_()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, step: int = -1, max_preview_dims: int = 8) -> dict:
        """Serialise batch item 0 to a human-readable dict (for inspection / testing).

        Float values rounded to 6 decimal places for stable JSON roundtrip comparison.
        """
        B, M, d = self.records.shape
        records_list = []
        for m in range(M):
            payload = self.records[0, m].detach().cpu()
            preview = [round(float(v), 6) for v in payload[:max_preview_dims].tolist()]
            records_list.append({
                "slot": m,
                "type": _TYPE_NAMES.get(int(self.types[0, m].item()), "UNKNOWN"),
                "type_id": int(self.types[0, m].item()),
                "occupied": bool(self.mask[0, m].item()),
                "step_written": int(self.step_written[0, m].item()),
                "norm": round(float(payload.norm().item()), 6),
                "vector_preview": preview,
                "vector_full": [round(float(v), 6) for v in payload.tolist()],
            })
        return {"step": step, "num_slots": M, "record_dim": d, "records": records_list}

    def to_json(self, step: int = -1, path: Path | str | None = None) -> str:
        """Serialise to JSON string (and optionally write to file)."""
        d = self.to_dict(step=step)
        s = json.dumps(d, indent=2)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    @classmethod
    def from_dict(
        cls,
        d: dict,
        device: torch.device,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> "ExternalWorkspaceState":
        """Reconstruct from a dict produced by to_dict().  Reconstructs batch item 0 only."""
        M = d["num_slots"]
        dim = d["record_dim"]
        ws = cls.init_empty(batch_size, M, dim, device, dtype)
        for rec in d["records"]:
            m = rec["slot"]
            ws.records[0, m] = torch.tensor(rec["vector_full"], dtype=dtype, device=device)
            ws.types[0, m] = rec["type_id"]
            ws.mask[0, m] = rec["occupied"]
            ws.step_written[0, m] = rec["step_written"]
        return ws

    @classmethod
    def from_json(cls, path: Path | str, device: torch.device, batch_size: int = 1) -> "ExternalWorkspaceState":
        """Load from JSON file produced by to_json()."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d, device=device, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Phase 6 corruption helpers  (pre-planted; return new state, never mutate)
    # ------------------------------------------------------------------

    def _copy_with(self, **kwargs) -> "ExternalWorkspaceState":
        """Return a clone with specified fields replaced."""
        c = self.clone(detach=False)
        for k, v in kwargs.items():
            object.__setattr__(c, k, v)
        return c

    def corrupt_swap_value(self, batch_idx: int, slot_idx: int, new_value: Tensor) -> "ExternalWorkspaceState":
        """Replace the payload of slot (batch_idx, slot_idx) with new_value."""
        new_records = self.records.clone()
        new_records[batch_idx, slot_idx] = new_value.to(new_records)
        return self._copy_with(records=new_records)

    def corrupt_zero_record(self, batch_idx: int, slot_idx: int) -> "ExternalWorkspaceState":
        """Zero out the payload of a slot while preserving type and occupied mask."""
        new_records = self.records.clone()
        new_records[batch_idx, slot_idx] = 0.0
        return self._copy_with(records=new_records)

    def corrupt_noise_record(self, batch_idx: int, slot_idx: int, std: float = 1.0) -> "ExternalWorkspaceState":
        """Add Gaussian noise to a slot's payload."""
        new_records = self.records.clone()
        noise = torch.randn_like(new_records[batch_idx, slot_idx]) * std
        new_records[batch_idx, slot_idx] = new_records[batch_idx, slot_idx] + noise
        return self._copy_with(records=new_records)

    def corrupt_delete_record(self, batch_idx: int, slot_idx: int) -> "ExternalWorkspaceState":
        """Mark a slot as empty (mask=False) without changing payload content."""
        new_mask = self.mask.clone()
        new_mask[batch_idx, slot_idx] = False
        return self._copy_with(mask=new_mask)
