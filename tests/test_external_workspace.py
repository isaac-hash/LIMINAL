"""Phase 5 — External Workspace Tests.

Tests cover:
  1. State init and tensor shapes
  2. JSON roundtrip fidelity
  3. WriteController top-K and dedup
  4. Write gradient flows through payload (not through slot selection)
  5. ReadController attention masking
  6. EdgeAdapter shape and equivariance
  7. Full coupled latent-external loop (forward + backward)
  8. Corruption isolation (snapshot branching)
  9. Workspace detach between turns
"""

import pytest
import torch
import torch.nn as nn

from src.utils.config import (
    ModelConfig, ActivityConfig, ResolutionConfig, ExternalConfig, PersistenceConfig
)
from src.models.external_workspace import ExternalWorkspaceState, ExternalRecordType
from src.models.externaliser import WriteController
from src.models.reader import ReadController
from src.models.relationships import EdgeAdapter
from src.models.latent_workspace import LatentWorkspace


# ─── Shared fixtures ────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
B, N, D = 2, 8, 32
M = 16
NUM_TYPES = 7
EXT_CFG = ExternalConfig(
    enabled=True,
    num_slots=M,
    record_dim=D,
    num_types=NUM_TYPES,
    write_top_k=3,
    write_dedup=True,
    read_heads=2,
    read_gate=True,
    edge_adaptation=True,
    edge_hidden_dim=32,
    detach_workspace_between_turns=True,
)


def make_empty_ws(batch_size: int = B) -> ExternalWorkspaceState:
    return ExternalWorkspaceState.init_empty(batch_size, M, D, DEVICE)


def make_V_A() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    V = torch.randn(B, N, D)
    A = torch.sigmoid(torch.randn(B, N))
    return V, A


# ─── Test 1: State init and shapes ──────────────────────────────────────────

def test_external_workspace_state_init_and_shapes():
    ws = make_empty_ws()

    assert ws.records.shape == (B, M, D)
    assert ws.types.shape == (B, M)
    assert ws.mask.shape == (B, M)
    assert ws.step_written.shape == (B, M)
    assert ws.wrote_this_pass.shape == (B, M)

    # All empty on init
    assert not ws.mask.any(), "All slots should start empty"
    assert (ws.types == ExternalRecordType.EMPTY).all(), "All types should be EMPTY"
    assert (ws.step_written == -1).all(), "step_written should initialise to -1"
    assert ws.records.abs().sum() == 0.0, "records should be zero on init"


# ─── Test 2: JSON roundtrip ──────────────────────────────────────────────────

def test_workspace_json_roundtrip():
    ws = make_empty_ws(batch_size=1)
    # Manually write some payloads
    ws.records[0, 0] = torch.randn(D)
    ws.types[0, 0] = ExternalRecordType.ENTITY
    ws.mask[0, 0] = True
    ws.step_written[0, 0] = 2

    ws.records[0, 1] = torch.randn(D)
    ws.types[0, 1] = ExternalRecordType.ATTRIBUTE
    ws.mask[0, 1] = True
    ws.step_written[0, 1] = 3

    d = ws.to_dict(step=3)
    ws2 = ExternalWorkspaceState.from_dict(d, device=DEVICE, batch_size=1)

    # Records reconstructed exactly (rounded to 6dp in JSON)
    assert ws2.records.shape == ws.records.shape
    assert ws2.mask[0, 0].item() is True
    assert ws2.mask[0, 1].item() is True
    assert not ws2.mask[0, 2].item()
    assert ws2.types[0, 0].item() == ExternalRecordType.ENTITY
    assert ws2.types[0, 1].item() == ExternalRecordType.ATTRIBUTE

    # Payload values within rounding tolerance
    for m in range(2):
        diff = (ws2.records[0, m] - ws.records[0, m]).abs().max().item()
        assert diff < 1e-4, f"Slot {m} payload deviated by {diff} after JSON roundtrip"


# ─── Test 3: WriteController top-K and dedup ────────────────────────────────

def test_write_controller_top_k_and_dedup():
    torch.manual_seed(42)
    V, A = make_V_A()
    ws = make_empty_ws()
    write_ctrl = WriteController(D, EXT_CFG)

    # First write: top-3 slots should fill 3 empty slots
    ws, _ = write_ctrl(ws, V, A, step=0)
    occupied = ws.mask[0].sum().item()
    assert occupied == 3, f"Expected 3 occupied slots after first write, got {occupied}"

    # Dedup: second call should skip slots already written this pass
    ws, _ = write_ctrl(ws, V, A, step=0)
    occupied_after = ws.mask[0].sum().item()
    # No new slots written because wrote_this_pass is set for top-3
    assert occupied_after == occupied, "Dedup should prevent re-writing same slots in same pass"

    # After reset_pass_flags, writing again should work
    ws.reset_pass_flags()
    ws, _ = write_ctrl(ws, V, A, step=1)
    occupied_final = ws.mask[0].sum().item()
    assert occupied_final >= occupied, "After flag reset, new writes should proceed"


def test_write_controller_lru_eviction():
    torch.manual_seed(7)
    V, A = make_V_A()
    # Use small workspace (3 slots) so it fills up on first write
    cfg = ExternalConfig(enabled=True, num_slots=3, record_dim=D, num_types=NUM_TYPES,
                         write_top_k=3, write_dedup=True, read_heads=2, read_gate=True,
                         edge_adaptation=False, edge_hidden_dim=32, detach_workspace_between_turns=True)
    ws = ExternalWorkspaceState.init_empty(B, 3, D, DEVICE)
    write_ctrl = WriteController(D, cfg)

    # Step 0: fill all 3 slots
    ws, _ = write_ctrl(ws, V, A, step=0)
    assert ws.mask[0].all(), "All 3 slots should be occupied after first write"
    assert (ws.step_written[0] == 0).all(), "All slots written at step 0"

    # Reset dedup flags and write at step 1 — must evict since workspace is full
    ws.reset_pass_flags()
    ws, _ = write_ctrl(ws, V, A, step=1)

    # At least one slot should now have step_written = 1 (it was evicted and re-written)
    assert (ws.step_written[0] == 1).any(), "LRU eviction should update step_written to step=1"



# ─── Test 4: Write gradient flows through payload ────────────────────────────

def test_write_gradient_flows_through_payload():
    torch.manual_seed(1)
    V = torch.randn(B, N, D, requires_grad=True)
    A = torch.sigmoid(torch.randn(B, N)).detach()  # scores detached in WriteController

    ws = make_empty_ws()
    write_ctrl = WriteController(D, EXT_CFG)

    ws, _ = write_ctrl(ws, V, A, step=0)

    # Downstream loss on the payload
    loss = ws.records.sum()
    loss.backward()

    assert V.grad is not None, "Gradient should flow from workspace records back to V"
    assert V.grad.abs().sum().item() > 0, "Gradient on V should be non-zero"


# ─── Test 5: ReadController attention masking ────────────────────────────────

def test_read_controller_attention_masking():
    torch.manual_seed(2)
    V, A = make_V_A()
    V.requires_grad_(True)

    ws = make_empty_ws()
    # Only occupy slot 0
    ws.records[0, 0] = torch.randn(D)
    ws.mask[0, 0] = True

    read_ctrl = ReadController(D, EXT_CFG)
    V_out, read_weights = read_ctrl(V, A, ws)

    # Empty slots (1..M-1) should receive near-zero attention weight for batch 0
    empty_weights = read_weights[0, :, 1:]  # [heads, M-1]
    assert empty_weights.abs().max().item() < 1e-4, "Empty slots should get ~0 attention"

    # Gradient should propagate back to V
    loss = V_out.sum()
    loss.backward()
    assert V.grad is not None and V.grad.abs().sum().item() > 0


def test_read_controller_all_empty_no_nan():
    """When workspace is completely empty, attention should not produce NaN."""
    torch.manual_seed(3)
    V = torch.randn(B, N, D)
    A = torch.ones(B, N)
    ws = make_empty_ws()  # all empty

    read_ctrl = ReadController(D, EXT_CFG)
    V_out, read_weights = read_ctrl(V, A, ws)

    assert not torch.isnan(V_out).any(), "All-empty workspace should not produce NaN"
    assert not torch.isnan(read_weights).any()


# ─── Test 6: EdgeAdapter shape and equivariance ──────────────────────────────

def test_edge_adapter_shapes():
    torch.manual_seed(4)
    edge_dim = 1
    V = torch.randn(B, N, D)
    E = torch.randn(B, N, N, edge_dim)
    adapter = EdgeAdapter(latent_dim=D, edge_dim=edge_dim, hidden_dim=32)

    E_new = adapter(V, E)
    assert E_new.shape == (B, N, N, edge_dim), f"Expected {(B, N, N, edge_dim)}, got {E_new.shape}"


def test_edge_adapter_equivariance():
    """Permuting nodes should permute edge output consistently."""
    torch.manual_seed(5)
    edge_dim = 1
    adapter = EdgeAdapter(latent_dim=D, edge_dim=edge_dim, hidden_dim=32)
    adapter.eval()

    V = torch.randn(1, N, D)
    E = torch.randn(1, N, N, edge_dim)

    perm = torch.randperm(N)
    V_perm = V[:, perm]
    E_perm = E[:, perm][:, :, perm]

    with torch.no_grad():
        E_new = adapter(V, E)
        E_new_perm = adapter(V_perm, E_perm)

    # The permuted output should equal permuting the original output
    E_expected = E_new[:, perm][:, :, perm]
    diff = (E_new_perm - E_expected).abs().max().item()
    assert diff < 1e-5, f"EdgeAdapter not equivariant: max diff = {diff}"


# ─── Test 7: Full coupled loop forward + backward ────────────────────────────

def test_coupled_latent_external_loop_forward_backward():
    torch.manual_seed(6)
    model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=D, reasoning_steps=4)
    act_cfg = ActivityConfig(enabled=True)
    ws_model = LatentWorkspace(model_cfg, activity_config=act_cfg, external_config=EXT_CFG)

    V_0 = torch.randn(B, N, D, requires_grad=True)
    h_final, info = ws_model(V_0)

    assert h_final.shape == (B, D), f"Expected [{B}, {D}], got {h_final.shape}"
    assert not torch.isnan(h_final).any(), "NaN in h_final"

    # Workspace trajectory should be populated
    assert len(info["workspace_trajectory"]) > 0, "workspace_trajectory should not be empty"
    assert len(info["read_weights_trajectory"]) > 0, "read_weights_trajectory should not be empty"

    # Backward should not raise and produce non-None gradients
    loss = h_final.sum()
    loss.backward()
    assert V_0.grad is not None and not torch.isnan(V_0.grad).any(), "Gradient on V_0 should be clean"


# ─── Test 8: Corruption isolation ────────────────────────────────────────────

def test_corruption_isolation():
    """Corrupting a cloned snapshot must not alter the original."""
    ws = make_empty_ws(batch_size=1)
    ws.records[0, 0] = torch.ones(D)
    ws.mask[0, 0] = True

    original_val = ws.records[0, 0].clone()

    # Clone and corrupt
    ws_corrupt = ws.corrupt_swap_value(0, 0, torch.zeros(D))

    # Original unchanged
    assert torch.allclose(ws.records[0, 0], original_val), \
        "corrupt_swap_value should not mutate the original"

    # Corrupted copy is different
    assert not torch.allclose(ws_corrupt.records[0, 0], original_val), \
        "Corrupted copy should have different payload"


def test_corrupt_delete_isolation():
    ws = make_empty_ws(batch_size=1)
    ws.mask[0, 3] = True
    ws.types[0, 3] = ExternalRecordType.RELATION

    ws_deleted = ws.corrupt_delete_record(0, 3)

    # Original still occupied
    assert ws.mask[0, 3].item() is True
    # Deleted copy shows empty
    assert ws_deleted.mask[0, 3].item() is False


# ─── Test 9: Workspace detach between turns ──────────────────────────────────

def test_workspace_detach_between_turns():
    ws = make_empty_ws()
    ws.records.requires_grad_(True)

    # Detached clone should have no grad_fn on records
    ws_detached = ws.clone(detach=True)
    assert ws_detached.records.requires_grad is False, \
        "Detached workspace records should not require grad"

    # Non-detached clone should preserve grad requirement
    ws_attached = ws.clone(detach=False)
    # Note: clone() without detach keeps tensor leaf status,
    # so we just verify it's a separate tensor (not an alias)
    assert ws_attached.records.data_ptr() != ws.records.data_ptr(), \
        "Clone should be a separate tensor, not an alias"
