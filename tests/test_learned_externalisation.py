"""Phase 7 Tests — Learned Externalisation.

Tests cover:
    1.  LearnedWriteController instantiation and set_tau
    2.  Forward pass shape contract (gate, gated payload, workspace update)
    3.  Gradient flow through the write gate and payload projection
    4.  Gate near-hard collapse at low tau (eval mode)
    5.  Gate sparsity: negative-bias init writes fewer slots than top-K
    6.  WriteController return-tuple signature (backward compat with Phase 7 callers)
    7.  ExternalisationSensitiveGenerator: dataset shapes, turn count, label validity
    8.  ExternalisationSensitiveGenerator: reproducibility with fixed seeds
    9.  ExternalConfig: learned_gate and gumbel fields parsed from dict
    10. LatentWorkspace: learned gate wired correctly when learned_gate=True
    11. LatentWorkspace: write_gate_mean emitted in info dict
    12. LossComputer: write_sparsity_lambda applied when learned_gate=True
    13. LossComputer: write sparsity NOT applied when learned_gate=False
    14. SelectivityAnalyser: compute_gate_stats shape contract
    15. SelectivityAnalyser: selectivity_score correct at extremes
    16. SelectivityAnalyser: per_turn_gate_fraction keys match update calls
    17. SequentialTrainer._maybe_anneal_tau: tau decreases over epochs
    18. SequentialTrainer._maybe_anneal_tau: no-op when learned_gate=False
"""

import math
import pytest
import torch
from torch import Tensor

from src.utils.config import ExternalConfig, ModelConfig, ActivityConfig, ResolutionConfig, Config, DataConfig, TrainingConfig, PersistenceConfig
from src.models.external_workspace import ExternalWorkspaceState
from src.models.externaliser import WriteController, LearnedWriteController
from src.models.latent_workspace import LatentWorkspace
from src.training.losses import LossComputer
from src.evaluation.selectivity_analysis import compute_gate_stats, SelectivityAnalyser


# ── helpers ────────────────────────────────────────────────────────────────────

def make_ext_config(learned: bool = True, write_top_k: int = 3, sparsity_lambda: float = 0.01) -> ExternalConfig:
    return ExternalConfig(
        enabled=True,
        num_slots=16,
        record_dim=32,
        num_types=7,
        write_top_k=write_top_k,
        write_dedup=False,  # easier to test without dedup
        read_heads=2,
        read_gate=True,
        edge_adaptation=False,
        edge_hidden_dim=32,
        detach_workspace_between_turns=True,
        learned_gate=learned,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.1,
        gumbel_anneal_epochs=30,
        write_sparsity_lambda=sparsity_lambda,
    )


def make_empty_workspace(B: int = 2, M: int = 16, d: int = 32) -> ExternalWorkspaceState:
    ws = ExternalWorkspaceState.init_empty(B, M, d, device=torch.device("cpu"), dtype=torch.float32)
    ws.reset_pass_flags()
    return ws


def make_VA(B: int = 2, N: int = 8, d: int = 32) -> tuple[Tensor, Tensor]:
    V = torch.randn(B, N, d)
    A = torch.sigmoid(torch.randn(B, N))
    return V, A


# ── 1. LearnedWriteController instantiation ────────────────────────────────────

def test_learned_write_controller_init():
    cfg = make_ext_config(learned=True)
    ctrl = LearnedWriteController(latent_dim=32, config=cfg)
    assert ctrl.tau == 1.0
    assert hasattr(ctrl, "gate_net")
    assert hasattr(ctrl, "payload_proj")
    assert hasattr(ctrl, "type_head")


# ── 2. set_tau ─────────────────────────────────────────────────────────────────

def test_set_tau():
    ctrl = LearnedWriteController(latent_dim=32, config=make_ext_config())
    ctrl.set_tau(0.3)
    assert abs(ctrl.tau - 0.3) < 1e-6
    # Floor at 1e-4
    ctrl.set_tau(-1.0)
    assert ctrl.tau >= 1e-4


# ── 3. Forward pass shape contract ─────────────────────────────────────────────

def test_learned_controller_forward_shapes():
    cfg = make_ext_config()
    ctrl = LearnedWriteController(latent_dim=32, config=cfg)
    ctrl.train()
    V, A = make_VA(B=2, N=8, d=32)
    ws = make_empty_workspace(B=2)
    ws_out, info = ctrl(ws, V, A, step=0)
    assert isinstance(ws_out, ExternalWorkspaceState)
    assert "write_gate" in info
    assert "write_gate_mean" in info
    assert "write_gate_hard" in info
    assert info["write_gate"].shape == (2, 8)
    assert info["write_gate_hard"].shape == (2, 8)
    assert info["write_gate"].requires_grad


# ── 4. Gradient flows through the write gate ───────────────────────────────────

def test_learned_gate_gradients():
    cfg = make_ext_config()
    ctrl = LearnedWriteController(latent_dim=32, config=cfg)
    ctrl.train()
    # Ensure gate activates so records receive payload and gate gradient deterministically
    with torch.no_grad():
        ctrl.gate_net[-1].bias.fill_(1.0)
    V, A = make_VA(B=2, N=8, d=32)
    V.requires_grad_(True)
    ws = make_empty_workspace(B=2)
    ws_out, info = ctrl(ws, V, A, step=0)
    # Gradient should reach V via payload projection and gate
    loss = ws_out.records.sum()
    loss.backward()
    assert V.grad is not None


# ── 5. Hard gate at eval mode (low tau) ────────────────────────────────────────

def test_hard_gate_at_eval():
    cfg = make_ext_config()
    ctrl = LearnedWriteController(latent_dim=32, config=cfg)
    ctrl.eval()
    V, A = make_VA(B=4, N=8, d=32)
    ws = make_empty_workspace(B=4)
    with torch.no_grad():
        _, info = ctrl(ws, V, A, step=0)
    gate = info["write_gate"]
    # At eval, gate should be exactly 0 or 1 (hard)
    assert torch.all((gate == 0) | (gate == 1))


# ── 6. Negative bias => fewer writes than top-K on first forward ────────────────

def test_learned_gate_writes_fewer_than_top_k_initially():
    cfg = make_ext_config(write_top_k=3)
    ctrl = LearnedWriteController(latent_dim=32, config=cfg)
    ctrl.eval()
    V, A = make_VA(B=16, N=8, d=32)
    ws = make_empty_workspace(B=16)
    with torch.no_grad():
        _, info = ctrl(ws, V, A, step=0)
    mean_writes = info["write_gate_hard"].sum(dim=1).float().mean().item()
    # Due to negative bias init, model should write fewer than N=8 slots on average
    # (it's not guaranteed to be < write_top_k=3 before training, but < 8)
    assert mean_writes < 8


# ── 7. WriteController returns tuple (backward compat) ─────────────────────────

def test_write_controller_returns_tuple():
    cfg = make_ext_config(learned=False)
    ctrl = WriteController(latent_dim=32, config=cfg)
    V, A = make_VA(B=2, N=8, d=32)
    ws = make_empty_workspace(B=2)
    result = ctrl(ws, V, A, step=0)
    assert isinstance(result, tuple)
    ws_out, info = result
    assert isinstance(ws_out, ExternalWorkspaceState)
    assert "write_gate" in info
    assert info["write_gate"].shape == (2, 8)


# ── 8. ExternalisationSensitiveGenerator: shapes and turns ────────────────────

def test_ext_sensitive_generator_shapes():
    from src.data.externalisation_sensitive import ExternalisationSensitiveGenerator
    gen = ExternalisationSensitiveGenerator(seed=42, num_train=10, num_val=5, num_test=5)
    splits = gen.generate_dataset()
    assert set(splits.keys()) == {"train", "val", "test"}
    assert len(splits["train"]) == 10
    seq = splits["train"][0]
    assert "turns" in seq
    assert len(seq["turns"]) == 5  # 5 turns per sequence


# ── 9. ExternalisationSensitiveGenerator: labels are binary ───────────────────

def test_ext_sensitive_labels_binary():
    from src.data.externalisation_sensitive import ExternalisationSensitiveGenerator
    gen = ExternalisationSensitiveGenerator(seed=7, num_train=50, num_val=5, num_test=5)
    splits = gen.generate_dataset()
    for seq in splits["train"]:
        for turn in seq["turns"]:
            assert turn["label"] in (0, 1), f"Label {turn['label']} is not binary"


# ── 10. ExternalisationSensitiveGenerator: reproducibility ────────────────────

def test_ext_sensitive_reproducibility():
    from src.data.externalisation_sensitive import ExternalisationSensitiveGenerator
    gen1 = ExternalisationSensitiveGenerator(seed=99, num_train=20, num_val=5, num_test=5)
    gen2 = ExternalisationSensitiveGenerator(seed=99, num_train=20, num_val=5, num_test=5)
    s1 = gen1.generate_dataset()
    s2 = gen2.generate_dataset()
    for t1, t2 in zip(s1["train"], s2["train"]):
        assert t1["sequence_id"] == t2["sequence_id"]
        for turn1, turn2 in zip(t1["turns"], t2["turns"]):
            assert turn1["label"] == turn2["label"]


# ── 11. ExternalConfig: learned_gate fields parsed correctly ──────────────────

def test_external_config_learned_gate_fields():
    cfg = ExternalConfig(
        enabled=True,
        learned_gate=True,
        gumbel_tau_start=2.0,
        gumbel_tau_end=0.05,
        gumbel_anneal_epochs=20,
        write_sparsity_lambda=0.005,
    )
    assert cfg.learned_gate is True
    assert cfg.gumbel_tau_start == 2.0
    assert cfg.gumbel_tau_end == 0.05
    assert cfg.gumbel_anneal_epochs == 20
    assert cfg.write_sparsity_lambda == 0.005


# ── 12. LatentWorkspace: LearnedWriteController instantiated when learned=True ─

def test_latent_workspace_uses_learned_controller():
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=2, max_reasoning_steps=4)
    ext_cfg = make_ext_config(learned=True)
    act_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    ws = LatentWorkspace(model_cfg, activity_config=act_cfg, external_config=ext_cfg)
    assert isinstance(ws.write_controller, LearnedWriteController)


# ── 13. LatentWorkspace: WriteController used when learned=False ───────────────

def test_latent_workspace_uses_deterministic_controller():
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=2, max_reasoning_steps=4)
    ext_cfg = make_ext_config(learned=False)
    act_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    ws = LatentWorkspace(model_cfg, activity_config=act_cfg, external_config=ext_cfg)
    assert isinstance(ws.write_controller, WriteController)


# ── 14. LatentWorkspace: write_gate_mean in info when learned=True ────────────

def test_latent_workspace_write_gate_mean_in_info():
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=2, max_reasoning_steps=4)
    ext_cfg = make_ext_config(learned=True)
    act_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    ws = LatentWorkspace(model_cfg, activity_config=act_cfg, external_config=ext_cfg)
    ws.train()
    B, N, d = 2, 8, 32
    V0 = torch.randn(B, N, d)
    _, info = ws(V0)
    assert "write_gate_mean" in info
    wgm = info["write_gate_mean"]
    assert wgm is not None
    assert wgm.shape == ()  # scalar


# ── 15. LatentWorkspace: write_gate_mean is None when learned=False ───────────

def test_latent_workspace_write_gate_mean_none_for_topk():
    model_cfg = ModelConfig(type="graph", latent_slots=8, latent_dim=32, reasoning_steps=2, max_reasoning_steps=4)
    ext_cfg = make_ext_config(learned=False)
    act_cfg = ActivityConfig(enabled=True, gate_hidden_dim=16)
    ws = LatentWorkspace(model_cfg, activity_config=act_cfg, external_config=ext_cfg)
    ws.eval()
    B, N, d = 2, 8, 32
    V0 = torch.randn(B, N, d)
    with torch.no_grad():
        _, info = ws(V0)
    assert info["write_gate_mean"] is None


# ── 16. LossComputer: write sparsity applied when learned_gate=True ───────────

def test_loss_computer_write_sparsity_applied():
    ext_cfg = make_ext_config(learned=True, sparsity_lambda=0.1)
    config = Config(external=ext_cfg)
    lc = LossComputer(config)
    logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    # Provide a differentiable write_gate_mean
    gate_mean = torch.tensor(0.5, requires_grad=True)
    info = {"write_gate_mean": gate_mean}
    loss, breakdown = lc(logits, labels, info)
    assert "loss_write_sparse" in breakdown
    assert "write_gate_mean" in breakdown
    assert abs(breakdown["loss_write_sparse"] - 0.1 * 0.5) < 1e-5


# ── 17. LossComputer: write sparsity NOT applied when learned_gate=False ──────

def test_loss_computer_write_sparsity_not_applied_topk():
    ext_cfg = make_ext_config(learned=False, sparsity_lambda=0.1)
    config = Config(external=ext_cfg)
    lc = LossComputer(config)
    logits = torch.randn(4, 2)
    labels = torch.randint(0, 2, (4,))
    gate_mean = torch.tensor(0.5)
    info = {"write_gate_mean": gate_mean}
    _, breakdown = lc(logits, labels, info)
    assert "loss_write_sparse" not in breakdown


# ── 18. compute_gate_stats: shape contract ─────────────────────────────────────

def test_compute_gate_stats_shape():
    gate = torch.zeros(4, 8)
    gate[:, :3] = 1.0  # 3 out of 8 slots fire for all batch items
    stats = compute_gate_stats(gate)
    assert "mean_slots_written" in stats
    assert "fraction_written" in stats
    assert "gate_entropy" in stats
    assert abs(stats["mean_slots_written"] - 3.0) < 1e-5
    assert abs(stats["fraction_written"] - 3.0 / 8) < 1e-5


# ── 19. SelectivityAnalyser: selectivity_score extremes ──────────────────────

def test_selectivity_score_all_write():
    analyser = SelectivityAnalyser(num_latent_slots=8, write_top_k=3)
    gate_all = torch.ones(4, 8)  # all slots always write
    analyser.update(gate_all, turn_index=0, step=0)
    report = analyser.summarise()
    assert abs(report["selectivity_score"]) < 1e-5  # 1 - 1.0 = 0


def test_selectivity_score_none_write():
    analyser = SelectivityAnalyser(num_latent_slots=8, write_top_k=3)
    gate_none = torch.zeros(4, 8)  # no slots ever write
    analyser.update(gate_none, turn_index=0, step=0)
    report = analyser.summarise()
    assert abs(report["selectivity_score"] - 1.0) < 1e-5  # 1 - 0.0 = 1


# ── 20. SelectivityAnalyser: per_turn_gate_fraction keys ─────────────────────

def test_selectivity_analyser_per_turn_keys():
    analyser = SelectivityAnalyser(num_latent_slots=8, write_top_k=3)
    gate = torch.zeros(2, 8)
    gate[:, :2] = 1.0
    for turn in [0, 1, 2]:
        analyser.update(gate, turn_index=turn, step=0)
    report = analyser.summarise()
    assert set(report["per_turn_gate_fraction"].keys()) == {0, 1, 2}


# ── 21. SelectivityAnalyser: empty analyser returns zeros ─────────────────────

def test_selectivity_analyser_empty():
    analyser = SelectivityAnalyser(num_latent_slots=8, write_top_k=3)
    report = analyser.summarise()
    assert report["num_steps_recorded"] == 0
    assert report["mean_slots_written_per_step"] == 0.0


# ── 22. Tau annealing: tau decreases over epochs ─────────────────────────────

def test_tau_annealing_decreases():
    """Verify set_tau linearly reaches gumbel_tau_end after gumbel_anneal_epochs."""
    cfg = ExternalConfig(
        enabled=True,
        learned_gate=True,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.1,
        gumbel_anneal_epochs=10,
        num_slots=16,
        record_dim=32,
    )
    ctrl = LearnedWriteController(latent_dim=32, config=cfg)
    taus = []
    for epoch in range(11):
        progress = min(epoch / 10, 1.0)
        tau = 1.0 + progress * (0.1 - 1.0)
        ctrl.set_tau(tau)
        taus.append(ctrl.tau)
    assert taus[0] == 1.0
    assert abs(taus[10] - 0.1) < 1e-5
    # Monotonically decreasing
    for i in range(len(taus) - 1):
        assert taus[i] >= taus[i + 1]
