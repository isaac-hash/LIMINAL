"""Phase 6 — Causal Intervention Tests.

Tests cover:
  1.  WorkspaceSnapshot: construction, detachment, repr.
  2.  LatentWorkspace.snapshot() / restore() / clear_snapshot() API.
  3.  V_at_step / E_at_step / W_at_step lists emitted in the info dict
      (both fixed-step and adaptive forward paths).
  4.  Corruption helpers via causal_intervention._apply_corruption():
        - SWAP_VALUE        (relevant slot)
        - DELETE_RECORD     (mask cleared)
        - RANDOMISE_RECORD  (noise added)
        - IRRELEVANT_SWAP   (different slot, positive control)
  5.  run_causal_intervention():
        - Returns InterventionResult with correct field types.
        - Relevant corruption produces non-zero L2 distance.
        - InterventionResult.as_dict() is fully JSON-serialisable.
  6.  run_all_corruption_types():
        - Returns exactly four results.
        - Corruption order matches [SWAP, DELETE, RANDOMISE, IRRELEVANT].
  7.  Irrelevant-slot corruption (negative control):
        - L2 distance is smaller than the relevant corruption for the same
          snapshot (the key causal check).
  8.  No gradient leakage from intervention branches (torch.no_grad() safe).
  9.  Snapshot-restore round-trip: V, E, W values preserved exactly.
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.utils.config import (
    ModelConfig,
    ActivityConfig,
    ResolutionConfig,
    ExternalConfig,
    PersistenceConfig,
)
from src.models.external_workspace import ExternalWorkspaceState, ExternalRecordType
from src.models.latent_workspace import LatentWorkspace, WorkspaceSnapshot
from src.evaluation.causal_intervention import (
    CorruptionType,
    InterventionConfig,
    InterventionResult,
    _apply_corruption,
    run_causal_intervention,
    run_all_corruption_types,
    print_intervention_summary,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
B, N, D = 2, 8, 32
M = 16

EXT_CFG = ExternalConfig(
    enabled=True,
    num_slots=M,
    record_dim=D,
    num_types=7,
    write_top_k=3,
    write_dedup=True,
    read_heads=2,
    read_gate=True,
    edge_adaptation=True,
    edge_hidden_dim=32,
    detach_workspace_between_turns=True,
)

MODEL_CFG = ModelConfig(
    type="graph",
    latent_slots=N,
    latent_dim=D,
    edge_dim=1,
    msg_hidden_dim=32,
    update_hidden_dim=32,
    num_msg_layers=1,
    reasoning_steps=4,
    max_reasoning_steps=8,
)

ACT_CFG = ActivityConfig(enabled=True, gate_hidden_dim=16)

RES_CFG = ResolutionConfig(
    enabled=True,
    halt_hidden_dim=16,
    halt_threshold=0.8,
    ponder_lambda=0.01,
    max_reasoning_steps=8,
)


def make_ws_model(with_resolution: bool = False) -> LatentWorkspace:
    rc = RES_CFG if with_resolution else None
    return LatentWorkspace(
        MODEL_CFG,
        activity_config=ACT_CFG,
        resolution_config=rc,
        external_config=EXT_CFG,
    )


def make_empty_ws() -> ExternalWorkspaceState:
    return ExternalWorkspaceState.init_empty(B, M, D, DEVICE)


def make_occupied_ws(relevant_slot: int = 0) -> ExternalWorkspaceState:
    ws = make_empty_ws()
    torch.manual_seed(99)
    ws.records[0, relevant_slot] = torch.randn(D)
    ws.mask[0, relevant_slot] = True
    ws.types[0, relevant_slot] = ExternalRecordType.ENTITY
    ws.step_written[0, relevant_slot] = 1
    return ws


# ---------------------------------------------------------------------------
# 1. WorkspaceSnapshot
# ---------------------------------------------------------------------------

class TestWorkspaceSnapshot:
    def test_detachment(self):
        """Snapshot tensors must be detached from the autograd graph."""
        V = torch.randn(B, N, D, requires_grad=True)
        E = torch.randn(B, N, N, 1, requires_grad=True)
        W = make_empty_ws()

        snap = WorkspaceSnapshot(V=V, E=E, W=W, step=2)

        assert not snap.V.requires_grad, "Snapshot V should be detached"
        assert snap.E is not None and not snap.E.requires_grad, "Snapshot E should be detached"

    def test_clone_isolation(self):
        """Modifying V after snapshot should not affect snap.V."""
        V = torch.ones(B, N, D)
        snap = WorkspaceSnapshot(V=V, E=None, W=None, step=0)
        V.fill_(99.0)
        assert snap.V.max().item() == pytest.approx(1.0), \
            "Snapshot should be isolated from mutations to the source tensor"

    def test_repr(self):
        snap = WorkspaceSnapshot(V=torch.zeros(B, N, D), E=None, W=None, step=3)
        r = repr(snap)
        assert "step=3" in r
        assert "V=" in r

    def test_workspace_none(self):
        snap = WorkspaceSnapshot(V=torch.zeros(B, N, D), E=None, W=None, step=0)
        assert snap.W is None
        assert snap.E is None


# ---------------------------------------------------------------------------
# 2. LatentWorkspace snapshot / restore / clear
# ---------------------------------------------------------------------------

class TestSnapshotRestoreAPI:
    def test_snapshot_stores_and_returns(self):
        ws_model = make_ws_model()
        V = torch.randn(B, N, D)
        E = torch.randn(B, N, N, 1)
        W = make_empty_ws()

        snap = ws_model.snapshot(V, E, W, step=1)
        assert isinstance(snap, WorkspaceSnapshot)
        assert snap.step == 1

    def test_restore_returns_cached(self):
        ws_model = make_ws_model()
        V = torch.randn(B, N, D)
        ws_model.snapshot(V, None, None, step=2)
        retrieved = ws_model.restore()
        assert retrieved.step == 2

    def test_restore_with_explicit_snap(self):
        ws_model = make_ws_model()
        snap = WorkspaceSnapshot(V=torch.zeros(B, N, D), E=None, W=None, step=5)
        retrieved = ws_model.restore(snap=snap)
        assert retrieved.step == 5

    def test_restore_raises_without_snapshot(self):
        ws_model = make_ws_model()
        with pytest.raises(RuntimeError, match="No snapshot available"):
            ws_model.restore()

    def test_clear_snapshot(self):
        ws_model = make_ws_model()
        ws_model.snapshot(torch.zeros(B, N, D), None, None, step=0)
        ws_model.clear_snapshot()
        with pytest.raises(RuntimeError):
            ws_model.restore()

    def test_snapshot_values_preserved(self):
        """Round-trip: values in snap.V match the original V to float precision."""
        ws_model = make_ws_model()
        torch.manual_seed(7)
        V = torch.randn(B, N, D)
        snap = ws_model.snapshot(V, None, None, step=3)
        assert torch.allclose(snap.V, V), "Snapshot V should match original V"


# ---------------------------------------------------------------------------
# 3. V_at_step / E_at_step / W_at_step in info dict
# ---------------------------------------------------------------------------

class TestPerStepSnapshots:
    def test_fixed_steps_emit_v_at_step(self):
        ws_model = make_ws_model(with_resolution=False)
        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)

        assert "V_at_step" in info
        assert len(info["V_at_step"]) == MODEL_CFG.reasoning_steps, \
            "V_at_step should have one entry per reasoning step"

    def test_adaptive_steps_emit_v_at_step(self):
        ws_model = make_ws_model(with_resolution=True)
        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)

        assert "V_at_step" in info
        assert len(info["V_at_step"]) > 0
        assert len(info["V_at_step"]) <= RES_CFG.max_reasoning_steps

    def test_w_at_step_is_none_when_external_disabled(self):
        """When external workspace is disabled, W_at_step entries should be None."""
        model_cfg = ModelConfig(type="graph", latent_slots=N, latent_dim=D, reasoning_steps=2)
        ws_model = LatentWorkspace(model_cfg, activity_config=ACT_CFG)
        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)

        assert "W_at_step" in info
        assert all(w is None for w in info["W_at_step"]), \
            "W_at_step should be None when external workspace is disabled"

    def test_w_at_step_populated_with_external(self):
        ws_model = make_ws_model(with_resolution=False)
        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)

        # After at least one write, some W_at_step entries should be non-None
        non_none = [w for w in info["W_at_step"] if w is not None]
        assert len(non_none) > 0, "W_at_step should contain workspace snapshots"

    def test_shapes_consistent(self):
        ws_model = make_ws_model(with_resolution=False)
        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)

        for i, V_step in enumerate(info["V_at_step"]):
            assert V_step.shape == (B, N, D), \
                f"V_at_step[{i}] has wrong shape: {V_step.shape}"


# ---------------------------------------------------------------------------
# 4. Corruption helpers
# ---------------------------------------------------------------------------

class TestCorruptionHelpers:
    def _base_cfg(self, ct: CorruptionType, slot: int = 0) -> InterventionConfig:
        return InterventionConfig(
            intervention_step=1,
            target_slot=slot,
            corruption_type=ct,
            swap_noise_std=5.0,
            randomise_std=1.0,
        )

    def test_swap_value_changes_payload(self):
        ws = make_occupied_ws(relevant_slot=0)
        original = ws.records[0, 0].clone()
        cfg = self._base_cfg(CorruptionType.SWAP_VALUE, slot=0)
        ws_c = _apply_corruption(ws, cfg, batch_idx=0)
        assert not torch.allclose(ws_c.records[0, 0], original), \
            "SWAP_VALUE should change the payload"
        # Original unchanged
        assert torch.allclose(ws.records[0, 0], original), \
            "Original workspace must not be mutated"

    def test_delete_record_clears_mask(self):
        ws = make_occupied_ws(relevant_slot=0)
        assert ws.mask[0, 0].item() is True
        cfg = self._base_cfg(CorruptionType.DELETE_RECORD, slot=0)
        ws_c = _apply_corruption(ws, cfg, batch_idx=0)
        assert ws_c.mask[0, 0].item() is False, "DELETE_RECORD should clear the mask"
        assert ws.mask[0, 0].item() is True, "Original mask must not change"

    def test_randomise_record_noises_payload(self):
        ws = make_occupied_ws(relevant_slot=0)
        original = ws.records[0, 0].clone()
        cfg = InterventionConfig(
            intervention_step=1, target_slot=0,
            corruption_type=CorruptionType.RANDOMISE_RECORD, randomise_std=2.0,
        )
        ws_c = _apply_corruption(ws, cfg, batch_idx=0)
        diff = (ws_c.records[0, 0] - original).abs().max().item()
        assert diff > 0.01, "RANDOMISE_RECORD should substantially perturb the payload"

    def test_irrelevant_swap_targets_different_slot(self):
        ws = make_occupied_ws(relevant_slot=0)
        # Make slot 2 occupied too
        ws.records[0, 2] = torch.ones(D)
        ws.mask[0, 2] = True
        original_slot2 = ws.records[0, 2].clone()
        original_slot0 = ws.records[0, 0].clone()

        cfg = InterventionConfig(
            intervention_step=1, target_slot=2,
            corruption_type=CorruptionType.IRRELEVANT_SWAP, swap_noise_std=5.0,
        )
        ws_c = _apply_corruption(ws, cfg, batch_idx=0)

        # Slot 2 changed
        assert not torch.allclose(ws_c.records[0, 2], original_slot2)
        # Slot 0 unchanged
        assert torch.allclose(ws_c.records[0, 0], original_slot0)

    def test_new_value_override(self):
        ws = make_occupied_ws(relevant_slot=0)
        override = torch.zeros(D)
        cfg = InterventionConfig(
            intervention_step=1, target_slot=0,
            corruption_type=CorruptionType.SWAP_VALUE,
            new_value_override=override,
        )
        ws_c = _apply_corruption(ws, cfg, batch_idx=0)
        assert torch.allclose(ws_c.records[0, 0], override), \
            "new_value_override should be used directly"


# ---------------------------------------------------------------------------
# 5. run_causal_intervention()
# ---------------------------------------------------------------------------

class TestRunCausalIntervention:
    def _setup(self):
        torch.manual_seed(42)
        ws_model = make_ws_model(with_resolution=False)
        ws_model.eval()

        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)

        # Use step 1 snapshot
        V_snap = info["V_at_step"][1]
        E_snap = info["E_at_step"][1]
        W_snap = info["W_at_step"][1]
        labels = torch.randint(0, 2, (B,))

        decoder = nn.Linear(D, 2)

        def _fwd(V_0, E_0, workspace):
            return ws_model(V_0, E_0=E_0, workspace=workspace)

        return _fwd, decoder, V_snap, E_snap, W_snap, labels

    def test_returns_intervention_result(self):
        fwd, dec, V, E, W, labels = self._setup()
        if W is None:
            pytest.skip("W_at_step is None; external workspace inactive in this run")

        cfg = InterventionConfig(
            target_slot=0,
            corruption_type=CorruptionType.SWAP_VALUE,
            swap_noise_std=5.0,
        )
        result = run_causal_intervention(
            model_forward_fn=fwd,
            decoder_fn=dec,
            V_snapshot=V,
            E_snapshot=E,
            workspace_snapshot=W,
            labels=labels,
            intervention_cfg=cfg,
        )
        assert isinstance(result, InterventionResult)
        assert isinstance(result.l2_trajectory_distance, float)
        assert isinstance(result.accuracy_delta, float)
        assert isinstance(result.answer_kl, float)

    def test_as_dict_is_json_serialisable(self):
        fwd, dec, V, E, W, labels = self._setup()
        if W is None:
            pytest.skip("W_at_step is None")

        cfg = InterventionConfig(target_slot=0, corruption_type=CorruptionType.DELETE_RECORD)
        result = run_causal_intervention(
            model_forward_fn=fwd, decoder_fn=dec,
            V_snapshot=V, E_snapshot=E,
            workspace_snapshot=W, labels=labels, intervention_cfg=cfg,
        )
        d = result.as_dict()
        serialised = json.dumps(d)  # should not raise
        reloaded = json.loads(serialised)
        assert reloaded["corruption_type"] == "DELETE_RECORD"

    def test_relevant_corruption_changes_trajectory(self):
        """A large SWAP on an occupied slot should change the trajectory."""
        fwd, dec, V, E, W, labels = self._setup()
        if W is None:
            pytest.skip("W_at_step is None")

        # Find an occupied slot
        occupied = W.mask[0].nonzero(as_tuple=False)
        if occupied.numel() == 0:
            pytest.skip("No occupied slots in snapshot")
        slot = int(occupied[0].item())

        cfg = InterventionConfig(
            target_slot=slot,
            corruption_type=CorruptionType.SWAP_VALUE,
            swap_noise_std=10.0,  # large noise for definite change
        )
        result = run_causal_intervention(
            model_forward_fn=fwd, decoder_fn=dec,
            V_snapshot=V, E_snapshot=E,
            workspace_snapshot=W, labels=labels, intervention_cfg=cfg,
        )
        # With large noise, L2 distance should be strictly positive
        assert result.l2_trajectory_distance > 0.0, \
            "Large SWAP_VALUE should produce non-zero trajectory divergence"

    def test_no_gradient_leakage(self):
        """Intervention runs must not accumulate gradients."""
        fwd, dec, V, E, W, labels = self._setup()
        if W is None:
            pytest.skip("W_at_step is None")

        V_req = V.clone().requires_grad_(True)
        cfg = InterventionConfig(target_slot=0, corruption_type=CorruptionType.SWAP_VALUE)
        run_causal_intervention(
            model_forward_fn=fwd, decoder_fn=dec,
            V_snapshot=V_req, E_snapshot=E,
            workspace_snapshot=W, labels=labels, intervention_cfg=cfg,
        )
        # No backward call; gradient should not have been accumulated
        assert V_req.grad is None, \
            "Intervention should not accumulate gradients without an explicit backward()"


# ---------------------------------------------------------------------------
# 6. run_all_corruption_types()
# ---------------------------------------------------------------------------

class TestRunAllCorruptionTypes:
    def test_returns_four_results(self):
        torch.manual_seed(0)
        ws_model = make_ws_model(with_resolution=False)
        ws_model.eval()

        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)

        V_snap = info["V_at_step"][0]
        E_snap = info["E_at_step"][0]
        W_snap = info["W_at_step"][0]

        if W_snap is None:
            pytest.skip("No workspace at step 0")

        decoder = nn.Linear(D, 2)
        labels = torch.randint(0, 2, (B,))

        def _fwd(V_0, E_0, workspace):
            return ws_model(V_0, E_0=E_0, workspace=workspace)

        results = run_all_corruption_types(
            model_forward_fn=_fwd, decoder_fn=decoder,
            V_snapshot=V_snap, E_snapshot=E_snap, workspace_snapshot=W_snap,
            labels=labels, relevant_slot=0, irrelevant_slot=M - 1,
        )
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"

    def test_corruption_types_in_order(self):
        torch.manual_seed(1)
        ws_model = make_ws_model(with_resolution=False)
        ws_model.eval()

        V_0 = torch.randn(B, N, D)
        _, info = ws_model(V_0)
        W_snap = info["W_at_step"][0]
        if W_snap is None:
            pytest.skip("No workspace at step 0")

        decoder = nn.Linear(D, 2)
        labels = torch.randint(0, 2, (B,))

        def _fwd(V_0, E_0, workspace):
            return ws_model(V_0, E_0=E_0, workspace=workspace)

        results = run_all_corruption_types(
            model_forward_fn=_fwd, decoder_fn=decoder,
            V_snapshot=info["V_at_step"][0], E_snapshot=info["E_at_step"][0],
            workspace_snapshot=W_snap, labels=labels,
            relevant_slot=0, irrelevant_slot=M - 1,
        )
        expected_order = [
            CorruptionType.SWAP_VALUE,
            CorruptionType.DELETE_RECORD,
            CorruptionType.RANDOMISE_RECORD,
            CorruptionType.IRRELEVANT_SWAP,
        ]
        for result, expected_ct in zip(results, expected_order):
            assert result.corruption_type == expected_ct, \
                f"Expected {expected_ct}, got {result.corruption_type}"


# ---------------------------------------------------------------------------
# 7. Irrelevant-slot negative control
# ---------------------------------------------------------------------------

class TestNegativeControl:
    def test_irrelevant_smaller_than_relevant(self):
        """Corrupting an empty/irrelevant slot should cause less divergence
        than corrupting an occupied relevant slot.

        Note: This is a stochastic test. It checks directional consistency
        across the mean of multiple trials rather than a single run.
        """
        torch.manual_seed(123)
        ws_model = make_ws_model(with_resolution=False)
        ws_model.eval()

        decoder = nn.Linear(D, 2)
        labels = torch.randint(0, 2, (B,))

        l2_relevant_list: list[float] = []
        l2_irrelevant_list: list[float] = []

        for trial_seed in range(5):
            torch.manual_seed(trial_seed * 17)
            V_0 = torch.randn(B, N, D)
            _, info = ws_model(V_0)
            W_snap = info["W_at_step"][1] if len(info["W_at_step"]) > 1 else info["W_at_step"][0]
            if W_snap is None:
                continue

            # Force relevant slot 0 to be occupied with a large vector
            W_snap.records[0, 0] = torch.randn(D) * 10
            W_snap.mask[0, 0] = True

            def _fwd(V_0, E_0, workspace):
                return ws_model(V_0, E_0=E_0, workspace=workspace)

            cfg_rel = InterventionConfig(
                target_slot=0,
                corruption_type=CorruptionType.SWAP_VALUE,
                swap_noise_std=5.0,
            )
            cfg_irr = InterventionConfig(
                target_slot=M - 1,           # last slot, likely empty
                corruption_type=CorruptionType.IRRELEVANT_SWAP,
                swap_noise_std=5.0,
            )
            res_rel = run_causal_intervention(
                model_forward_fn=_fwd, decoder_fn=decoder,
                V_snapshot=info["V_at_step"][0],
                E_snapshot=info["E_at_step"][0],
                workspace_snapshot=W_snap, labels=labels,
                intervention_cfg=cfg_rel,
            )
            res_irr = run_causal_intervention(
                model_forward_fn=_fwd, decoder_fn=decoder,
                V_snapshot=info["V_at_step"][0],
                E_snapshot=info["E_at_step"][0],
                workspace_snapshot=W_snap, labels=labels,
                intervention_cfg=cfg_irr,
            )
            l2_relevant_list.append(res_rel.l2_trajectory_distance)
            l2_irrelevant_list.append(res_irr.l2_trajectory_distance)

        if not l2_relevant_list:
            pytest.skip("No trials completed — workspace not active")

        mean_rel = sum(l2_relevant_list) / len(l2_relevant_list)
        mean_irr = sum(l2_irrelevant_list) / len(l2_irrelevant_list)

        # Directional assertion: relevant should cause more disruption
        assert mean_rel >= mean_irr, (
            f"Expected mean_rel ({mean_rel:.4f}) >= mean_irr ({mean_irr:.4f}). "
            "Negative control failed the directional check."
        )


# ---------------------------------------------------------------------------
# 8. print_intervention_summary (smoke test — no crash)
# ---------------------------------------------------------------------------

def test_print_intervention_summary_smoke(capsys):
    fake = [
        InterventionResult(
            corruption_type=ct,
            target_slot=0,
            l2_trajectory_distance=0.5,
            accuracy_delta=-0.1,
            answer_kl=0.2,
            iteration_delta=0.3,
            resolution_time_delta=0.1,
            clean_logits=torch.zeros(B, 2),
            corrupted_logits=torch.zeros(B, 2),
            clean_info={},
            corrupted_info={},
        )
        for ct in CorruptionType
    ]
    print_intervention_summary(fake)
    captured = capsys.readouterr()
    assert "SWAP_VALUE" in captured.out
    assert "IRRELEVANT_SWAP" in captured.out
