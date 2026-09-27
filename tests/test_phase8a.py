"""Unit tests for Phase 8A benchmark harness.

Tests cover:
  1. State dict key remapping (bare ReasoningModel → SequentialReasoningModel).
  2. Canonical vocabulary construction and entity vocab size.
  3. Shared evaluation dataset generation (shape, reproducibility).
  4. Batch metric extraction correctness (per-turn accuracy computation).
  5. Smoke tests: load_model + evaluate_model on each real checkpoint.
  6. Latency measurement returns a positive finite float.
  7. CLI exporter correctness (JSON, CSV, Markdown table).

Run with::

    pytest tests/test_phase8a.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Import harness modules (project must be on sys.path, e.g. via pytest.ini or
# running pytest from the project root).
# ---------------------------------------------------------------------------
from src.evaluation.phase8a_benchmark import (
    HOLDOUT_SEED,
    N_EVAL_SEQUENCES,
    N_EVAL_TURNS,
    Phase8Evaluator,
    _extract_batch_metrics,
    _remap_state_dict_to_sequential,
    build_canonical_vocab,
    build_eval_dataset,
    evaluate_model,
    load_model,
    measure_latency,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. State dict key remapping
# ──────────────────────────────────────────────────────────────────────────────

class TestStateDict:
    """Tests for _remap_state_dict_to_sequential."""

    def test_bare_keys_are_prefixed(self):
        sd = {
            "encoder.entity_embed.weight": torch.zeros(20, 16),
            "workspace.mlp.weight": torch.zeros(32, 32),
            "decoder.fc.weight": torch.zeros(2, 32),
        }
        out = _remap_state_dict_to_sequential(sd)
        assert all(k.startswith("base_model.") for k in out), (
            "All bare keys should be prefixed with 'base_model.'"
        )

    def test_already_prefixed_keys_unchanged(self):
        sd = {
            "base_model.encoder.entity_embed.weight": torch.zeros(20, 16),
            "persistence_gate.fc.weight": torch.zeros(32, 32),
        }
        out = _remap_state_dict_to_sequential(sd)
        assert "base_model.encoder.entity_embed.weight" in out
        assert "persistence_gate.fc.weight" in out
        assert "base_model.base_model.encoder.entity_embed.weight" not in out, (
            "Keys already prefixed must not be double-prefixed."
        )

    def test_mixed_keys(self):
        sd = {
            "encoder.x": torch.zeros(4),
            "base_model.decoder.y": torch.zeros(4),
            "persistence_gate.z": torch.zeros(4),
        }
        out = _remap_state_dict_to_sequential(sd)
        assert "base_model.encoder.x" in out
        assert "base_model.decoder.y" in out
        assert "persistence_gate.z" in out
        assert len(out) == 3

    def test_no_data_loss(self):
        sd = {f"layer{i}.weight": torch.randn(8, 8) for i in range(5)}
        out = _remap_state_dict_to_sequential(sd)
        assert len(out) == len(sd)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Canonical vocabulary
# ──────────────────────────────────────────────────────────────────────────────

class TestCanonicalVocab:
    """Tests for build_canonical_vocab."""

    def test_has_pad_and_unk(self):
        vocab = build_canonical_vocab()
        assert "<PAD>" in vocab.entities
        assert "<UNK>" in vocab.entities

    def test_entity_vocab_size_is_20(self):
        """len(vocab.entities) should be 18 so encoder build gives size 20."""
        vocab = build_canonical_vocab()
        # ReasoningModel builds: entity_vocab_size = len(vocab.entities) + 2
        # We need this to equal 20 for A/B/C compatibility.
        assert len(vocab.entities) + 2 == 20, (
            f"Expected 20, got {len(vocab.entities) + 2}. "
            "Canonical entity list may have been changed."
        )

    def test_key_vocab_size_is_6(self):
        """Only money+price as named keys → key_vocab_size = 4+2 = 6 (fits A/B/C)."""
        vocab = build_canonical_vocab()
        # 2 named keys + PAD + UNK = 4 entries; encoder build adds +2 → size 6.
        assert len(vocab.keys) + 2 == 6, (
            f"Expected key_vocab_size 6, got {len(vocab.keys) + 2}. "
            "Canonical key list must only contain money and price."
        )

    def test_reproducible(self):
        v1 = build_canonical_vocab()
        v2 = build_canonical_vocab()
        assert v1.entities == v2.entities
        assert v1.keys == v2.keys


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dataset generation
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildEvalDataset:
    """Tests for build_eval_dataset."""

    @pytest.fixture(scope="class")
    def dataset_and_vocab(self):
        return build_eval_dataset(n_sequences=20, n_turns=5, seed=HOLDOUT_SEED)

    def test_dataset_length(self, dataset_and_vocab):
        ds, _ = dataset_and_vocab
        assert len(ds) == 20

    def test_each_sample_has_n_turns(self, dataset_and_vocab):
        ds, _ = dataset_and_vocab
        for i in range(min(5, len(ds))):
            sample = ds[i]
            assert len(sample["turns"]) == 5, (
                f"Sample {i} has {len(sample['turns'])} turns, expected 5."
            )

    def test_reproducibility(self):
        ds1, _ = build_eval_dataset(n_sequences=10, n_turns=3, seed=42)
        ds2, _ = build_eval_dataset(n_sequences=10, n_turns=3, seed=42)
        s1 = ds1[0]["turns"][0]["facts"]
        s2 = ds2[0]["turns"][0]["facts"]
        assert torch.equal(s1, s2), "Dataset is not deterministic across calls."

    def test_different_seeds_differ(self):
        ds1, _ = build_eval_dataset(n_sequences=10, n_turns=3, seed=1)
        ds2, _ = build_eval_dataset(n_sequences=10, n_turns=3, seed=2)
        s1 = ds1[0]["turns"][0]["facts"]
        s2 = ds2[0]["turns"][0]["facts"]
        assert not torch.equal(s1, s2), "Different seeds produced identical data."


# ──────────────────────────────────────────────────────────────────────────────
# 4. Batch metric extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractBatchMetrics:
    """Tests for _extract_batch_metrics."""

    def _make_batch(self, B: int, T: int, C: int, correct_all: bool):
        labels = torch.zeros(B, T, dtype=torch.long)  # all class 0
        if correct_all:
            # logits heavily favour class 0
            logits = torch.zeros(B, T, C)
            logits[:, :, 0] = 10.0
        else:
            # logits heavily favour class 1 (wrong)
            logits = torch.zeros(B, T, C)
            logits[:, :, 1] = 10.0
        turn_mask = torch.ones(B, T)
        infos = [{} for _ in range(T)]
        return logits, labels, turn_mask, infos

    def test_perfect_accuracy(self):
        logits, labels, turn_mask, infos = self._make_batch(4, 5, 2, correct_all=True)
        m = _extract_batch_metrics(logits, labels, turn_mask, infos, has_external=False)
        for t in range(5):
            assert m["per_turn_correct"][t] == 4
            assert m["per_turn_total"][t] == 4

    def test_zero_accuracy(self):
        logits, labels, turn_mask, infos = self._make_batch(4, 5, 2, correct_all=False)
        m = _extract_batch_metrics(logits, labels, turn_mask, infos, has_external=False)
        for t in range(5):
            assert m["per_turn_correct"][t] == 0

    def test_masked_turns_excluded(self):
        B, T, C = 4, 3, 2
        labels = torch.zeros(B, T, dtype=torch.long)
        logits = torch.zeros(B, T, C)
        logits[:, :, 0] = 10.0
        turn_mask = torch.ones(B, T)
        turn_mask[:, 2] = 0.0  # last turn masked out for all examples
        infos = [{} for _ in range(T)]

        m = _extract_batch_metrics(logits, labels, turn_mask, infos, has_external=False)
        assert m["per_turn_total"][2] == 0, "Masked turn should contribute 0 samples."

    def test_output_keys_present(self):
        logits, labels, turn_mask, infos = self._make_batch(2, 3, 2, correct_all=True)
        m = _extract_batch_metrics(logits, labels, turn_mask, infos, has_external=False)
        required = {
            "per_turn_correct", "per_turn_total", "per_turn_loss",
            "mean_halt_steps", "mean_active_slots", "mean_ext_writes",
        }
        assert required.issubset(m.keys())


# ──────────────────────────────────────────────────────────────────────────────
# 5. Smoke tests — real checkpoint loading & evaluation
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_MAP = {
    "A": Path("results/vector_baseline/best.pt"),
    "B": Path("results/static_graph/best.pt"),
    "C": Path("LIMINAL_results/persistence_incremental/best.pt"),
    "D": Path("LIMINAL_results/externalisation_comparison/best.pt"),
}


@pytest.mark.parametrize("model_id,ckpt_path", list(CHECKPOINT_MAP.items()))
class TestSmokeLoadAndEvaluate:
    """Smoke tests: load each real checkpoint and run a tiny evaluation pass."""

    @pytest.fixture(scope="class")
    def tiny_dataset(self):
        """A tiny 10-sequence dataset for fast smoke tests."""
        ds, vocab = build_eval_dataset(n_sequences=10, n_turns=5, seed=HOLDOUT_SEED)
        return ds, vocab

    def test_load_model(self, model_id, ckpt_path, tiny_dataset):
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")

        model, meta = load_model(model_id, ckpt_path, device="cpu")
        assert meta["model_id"] == model_id
        assert "param_count" in meta
        assert meta["param_count"]["trainable"] > 0

    def test_evaluate_model_returns_correct_keys(self, model_id, ckpt_path, tiny_dataset):
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")

        ds, _ = tiny_dataset
        model, meta = load_model(model_id, ckpt_path, device="cpu")

        result = evaluate_model(model, ds, meta, batch_size=4, device="cpu")
        required_keys = {
            "overall_accuracy", "per_turn_accuracy", "mean_loss",
            "mean_halt_steps", "mean_active_slots", "mean_ext_writes",
            "param_count", "has_persistence", "has_external",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_accuracy_in_valid_range(self, model_id, ckpt_path, tiny_dataset):
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")

        ds, _ = tiny_dataset
        model, meta = load_model(model_id, ckpt_path, device="cpu")
        result = evaluate_model(model, ds, meta, batch_size=4, device="cpu")

        assert 0.0 <= result["overall_accuracy"] <= 1.0, (
            f"Accuracy out of range: {result['overall_accuracy']}"
        )
        for t, acc in enumerate(result["per_turn_accuracy"]):
            assert 0.0 <= acc <= 1.0, f"Turn {t+1} accuracy out of range: {acc}"

    def test_per_turn_accuracy_length(self, model_id, ckpt_path, tiny_dataset):
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")

        ds, _ = tiny_dataset
        model, meta = load_model(model_id, ckpt_path, device="cpu")
        result = evaluate_model(model, ds, meta, batch_size=4, device="cpu")

        assert len(result["per_turn_accuracy"]) == N_EVAL_TURNS, (
            f"Expected {N_EVAL_TURNS} turn entries, "
            f"got {len(result['per_turn_accuracy'])}."
        )

    def test_latency_is_positive(self, model_id, ckpt_path, tiny_dataset):
        if not ckpt_path.exists():
            pytest.skip(f"Checkpoint not found: {ckpt_path}")

        ds, _ = tiny_dataset
        model, meta = load_model(model_id, ckpt_path, device="cpu")
        lat = measure_latency(model, ds, n_samples=8, batch_size=4, device="cpu")

        assert lat > 0.0, f"Latency should be positive, got {lat}"
        assert lat < 60_000.0, f"Latency suspiciously large: {lat} ms/seq"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Phase8Evaluator orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class TestPhase8Evaluator:
    """Tests for the high-level Phase8Evaluator class."""

    def test_register_and_run_available_checkpoints(self):
        ev = Phase8Evaluator(
            device="cpu",
            batch_size=8,
            n_eval_sequences=10,
            n_turns=5,
            seed=HOLDOUT_SEED,
        )
        registered = 0
        for mid, ckpt in CHECKPOINT_MAP.items():
            if ckpt.exists():
                ev.register(mid, ckpt)
                registered += 1

        if registered == 0:
            pytest.skip("No checkpoints found — cannot run evaluator smoke test.")

        results = ev.run()
        assert len(results) == registered
        for r in results:
            assert "overall_accuracy" in r
            assert "per_turn_accuracy" in r


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI exporter functions
# ──────────────────────────────────────────────────────────────────────────────

class TestExporters:
    """Tests for the CLI exporter functions (JSON, CSV, Markdown)."""

    @pytest.fixture
    def fake_results(self):
        return [
            {
                "model_id": "A",
                "label": "Vector Baseline",
                "overall_accuracy": 0.75,
                "per_turn_accuracy": [0.9, 0.8, 0.7, 0.65, 0.60],
                "per_turn_loss": [0.3, 0.35, 0.4, 0.45, 0.5],
                "mean_loss": 0.40,
                "mean_halt_steps": 0.0,
                "mean_active_slots": 0.0,
                "mean_ext_writes": 0.0,
                "latency_ms_per_seq": 2.5,
                "param_count": {"total": 10000, "trainable": 10000},
                "has_persistence": False,
                "has_external": False,
                "epoch": 49,
                "ckpt_path": "results/vector_baseline/best.pt",
            },
            {
                "model_id": "D",
                "label": "Full LIMINAL (Phase 7)",
                "overall_accuracy": 0.90,
                "per_turn_accuracy": [0.95, 0.93, 0.92, 0.88, 0.82],
                "per_turn_loss": [0.15, 0.18, 0.2, 0.25, 0.3],
                "mean_loss": 0.22,
                "mean_halt_steps": 6.2,
                "mean_active_slots": 0.65,
                "mean_ext_writes": 0.05,
                "latency_ms_per_seq": 8.1,
                "param_count": {"total": 50000, "trainable": 50000},
                "has_persistence": True,
                "has_external": True,
                "epoch": 49,
                "ckpt_path": "LIMINAL_results/externalisation_comparison/best.pt",
            },
        ]

    def test_json_export_valid(self, fake_results, tmp_path):
        from scripts.run_phase8a_benchmark import _save_json
        p = _save_json(fake_results, tmp_path)
        assert p.exists()
        with open(p) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["model_id"] == "A"

    def test_csv_export_has_header_and_rows(self, fake_results, tmp_path):
        from scripts.run_phase8a_benchmark import _save_csv
        import csv as csv_mod
        p = _save_csv(fake_results, tmp_path)
        assert p.exists()
        with open(p, newline="") as f:
            reader = list(csv_mod.DictReader(f))
        assert len(reader) == 2
        assert "overall_accuracy" in reader[0]
        assert "turn_1_acc" in reader[0]
        assert "turn_5_acc" in reader[0]

    def test_markdown_table_export(self, fake_results, tmp_path):
        from scripts.run_phase8a_benchmark import _save_markdown_table
        p = _save_markdown_table(fake_results, tmp_path)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "Overall Accuracy" in content
        assert "Vector Baseline" in content
        assert "Full LIMINAL" in content
        assert "Turn 1" in content
        assert "Turn 5" in content
