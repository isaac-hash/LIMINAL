"""Phase 8A Benchmark Engine.

Provides a unified evaluation harness for comparing four LIMINAL checkpoints
on an identical held-out 5-turn sequence test set:

    A — Vector Baseline       (ReasoningModel, stateless per-turn)
    B — Static Graph          (ReasoningModel, stateless per-turn)
    C — Persistent Graph      (SequentialReasoningModel, Phase 4)
    D — Full LIMINAL          (SequentialReasoningModel, Phase 7)

The shared evaluation dataset is ``affordability_sequence`` at 5 turns,
generated with holdout seed=999 to guarantee zero overlap with any
training or validation split from previous phases.

Architecture reconciliation
---------------------------
* Models A/B were saved as bare ``ReasoningModel`` — state dict keys are
  direct (``encoder.*``, ``workspace.*``, ``decoder.*``).
* Models C/D were saved as ``SequentialReasoningModel`` — keys are prefixed
  ``base_model.*`` and ``persistence_gate.*``.

For A/B the harness instantiates ``SequentialReasoningModel`` (with
persistence disabled), remaps keys to ``base_model.*``, and runs the shared
sequential evaluation loop.  The per-turn accuracy of A/B therefore reflects
stateless, per-turn inference — cleanly exposing temporal-collapse behaviour.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import (
    ActivityConfig,
    DataConfig,
    ExternalConfig,
    ModelConfig,
    PersistenceConfig,
    ResolutionConfig,
    set_seed,
)
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary
from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
from src.models.sequential_model import SequentialReasoningModel
from src.evaluation.metrics import count_parameters

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Holdout seed — never used in any prior training / validation split.
HOLDOUT_SEED: int = 999

#: Number of evaluation sequences.
N_EVAL_SEQUENCES: int = 500

#: Number of turns per shared evaluation sequence.
N_EVAL_TURNS: int = 5

#: Latency profiling: warm-up batches before timing begins.
WARMUP_BATCHES: int = 5

#: Canonical vocab entities for the shared affordability_sequence task.
_CANONICAL_ENTITIES = [
    "John", "Alice", "Bob", "Emma",
    "David", "Sarah", "Michael", "Olivia",
]
_CANONICAL_ITEMS = [
    "shoe", "book", "laptop", "watch",
    "phone", "jacket", "bicycle", "camera",
]


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def build_canonical_vocab() -> Vocabulary:
    """Return the canonical vocabulary used during shared evaluation.

    Uses only the entities and keys that the ``affordability_sequence``
    generator actually produces, so that encoded token indices always fall
    within every model's embedding table:

    * entity: 18 named (8 people + 8 items) → table size 20 (all models ≥ 20)
    * key:    2 named (money, price)         → table size  6 (all models ≥ 6)
    * type:   3 named (entity, attr, op)     → table size  7 ✓
    * op:     3 named (none, add, sub)       → table size  7 ✓  (via Vocabulary)
    """
    vocab = Vocabulary()
    for name in _CANONICAL_ENTITIES + _CANONICAL_ITEMS:
        idx = len(vocab.entities)
        vocab.entities[name] = idx
        vocab.idx2entity[idx] = name
    # Only add the 2 keys that affordability tasks actually use.
    # Adding more keys would exceed the size-6 key embedding of A/B/C.
    for key in ["money", "price"]:
        if key not in vocab.keys:
            idx = len(vocab.keys)
            vocab.keys[key] = idx
            vocab.idx2key[idx] = key
    return vocab


def _build_model_vocab(sd: dict[str, torch.Tensor], has_base_model: bool) -> Vocabulary:
    """Build a Vocabulary whose sizes exactly match the checkpoint's embeddings.

    Reads the four embedding weight shapes from the state dict and pads the
    canonical vocabulary with synthetic dummy tokens so that instantiating
    ``ReasoningModel`` with this vocab produces embeddings of exactly the
    saved sizes.  The data generated from ``build_canonical_vocab`` only uses
    low indices, so all its token indices are within every padded table.

    Args:
        sd:             Model state dict (already with or without ``base_model.`` prefix).
        has_base_model: Whether the state dict uses the ``base_model.`` prefix.

    Returns:
        Vocabulary with sizes matching the checkpoint embeddings.
    """
    pfx = "base_model." if has_base_model else ""

    # Read the saved sizes.  At model build time the encoder received:
    #   entity_vocab_size = len(vocab.entities) + 2
    # So we need:  len(vocab.entities) = saved_size - 2
    saved_entity = sd[f"{pfx}encoder.entity_embed.weight"].shape[0]
    saved_key    = sd[f"{pfx}encoder.key_embed.weight"].shape[0]
    saved_type   = sd[f"{pfx}encoder.type_embed.weight"].shape[0]
    saved_op     = sd[f"{pfx}encoder.op_embed.weight"].shape[0]

    # Start from the canonical data vocab (guaranteed to have small indices)
    vocab = build_canonical_vocab()

    # --- Pad entities ---
    target_entities = saved_entity - 2  # = len(vocab.entities) we need
    while len(vocab.entities) < target_entities:
        dummy = f"__pad_entity_{len(vocab.entities)}__"
        idx = len(vocab.entities)
        vocab.entities[dummy] = idx
        vocab.idx2entity[idx] = dummy

    # --- Pad keys ---
    target_keys = saved_key - 2
    while len(vocab.keys) < target_keys:
        dummy = f"__pad_key_{len(vocab.keys)}__"
        idx = len(vocab.keys)
        vocab.keys[dummy] = idx
        vocab.idx2key[idx] = dummy

    # --- Pad fact_types (Vocabulary already has entity/attribute/operation) ---
    target_types = saved_type - 2
    while len(vocab.fact_types) < target_types:
        dummy = f"__pad_type_{len(vocab.fact_types)}__"
        idx = len(vocab.fact_types)
        vocab.fact_types[dummy] = idx
        vocab.idx2fact_type[idx] = dummy

    # --- Pad ops (Vocabulary already has none/add/sub) ---
    target_ops = saved_op - 2
    while len(vocab.ops) < target_ops:
        dummy = f"__pad_op_{len(vocab.ops)}__"
        idx = len(vocab.ops)
        vocab.ops[dummy] = idx
        vocab.idx2op[idx] = dummy

    return vocab


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def build_eval_dataset(
    n_sequences: int = N_EVAL_SEQUENCES,
    n_turns: int = N_EVAL_TURNS,
    seed: int = HOLDOUT_SEED,
) -> tuple[SequenceReasoningDataset, Vocabulary]:
    """Generate the shared held-out evaluation dataset and vocabulary.

    Returns:
        (dataset, vocab): Ready-to-use dataset and associated vocabulary.
    """
    set_seed(seed)
    data_cfg = DataConfig(
        task_family="affordability_sequence",
        sequence_turns=n_turns,
        ops_per_turn=1,
        incremental_turns=False,
        num_train=0,
        num_val=0,
        num_test=n_sequences,
        max_entities=4,
        max_operations=2,
        distractor_ratio=0.0,
        seed=seed,
    )
    gen = ArithmeticGenerator(config=data_cfg, seed=seed)
    splits = gen.generate_dataset()
    records = splits["test"]

    vocab = build_canonical_vocab()
    dataset = SequenceReasoningDataset(records, vocab)
    return dataset, vocab


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL_LABELS = {
    "A": "Vector Baseline",
    "B": "Static Graph",
    "C": "Persistent Graph (Phase 4)",
    "D": "Full LIMINAL (Phase 7)",
}


def _remap_state_dict_to_sequential(
    sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Prepend ``base_model.`` to state dict keys that lack it.

    Used when a bare ``ReasoningModel`` checkpoint (Models A/B) is loaded
    into a ``SequentialReasoningModel`` wrapper.
    """
    new_sd: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("base_model.") or k.startswith("persistence_gate."):
            new_sd[k] = v
        else:
            new_sd[f"base_model.{k}"] = v
    return new_sd


def load_model(
    model_id: str,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SequentialReasoningModel, dict[str, Any]]:
    """Load a model checkpoint and return the model + metadata dict.

    Args:
        model_id:        One of ``"A"``, ``"B"``, ``"C"``, ``"D"``.
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        device:          Target device for inference.

    Returns:
        (model, meta): Instantiated eval-mode model and metadata dict.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd: dict[str, torch.Tensor] = ckpt["model_state_dict"]
    saved_cfg = ckpt.get("config")
    epoch: int = ckpt.get("epoch", -1)

    if saved_cfg is not None:
        model_cfg = saved_cfg.model
        activity_cfg = saved_cfg.activity
        resolution_cfg = saved_cfg.resolution
        persistence_cfg = saved_cfg.persistence
        external_cfg = saved_cfg.external
    else:
        model_cfg = ModelConfig()
        activity_cfg = ActivityConfig()
        resolution_cfg = ResolutionConfig()
        persistence_cfg = PersistenceConfig()
        external_cfg = ExternalConfig()

    has_base_model = any(k.startswith("base_model.") for k in sd)
    has_persistence = any(k.startswith("persistence_gate.") for k in sd)
    has_external = (
        saved_cfg is not None
        and saved_cfg.external is not None
        and saved_cfg.external.enabled
    )

    # Build a per-model vocab whose sizes match the checkpoint embeddings exactly.
    # This avoids size mismatches when calling load_state_dict.
    # The data vocab (build_canonical_vocab) only uses low indices that are
    # always within any model's (larger) embedding table.
    model_vocab = _build_model_vocab(sd, has_base_model)

    seq_model = SequentialReasoningModel(
        config=model_cfg,
        vocab=model_vocab,
        activity_config=activity_cfg if activity_cfg.enabled else None,
        resolution_config=resolution_cfg if resolution_cfg.enabled else None,
        persistence_config=persistence_cfg if persistence_cfg.enabled else None,
        external_config=external_cfg if has_external else None,
    )

    if not has_base_model:
        sd = _remap_state_dict_to_sequential(sd)

    missing, unexpected = seq_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load_model] {model_id}: {len(missing)} missing keys "
              f"(e.g. {missing[:3]})")
    if unexpected:
        print(f"  [load_model] {model_id}: {len(unexpected)} unexpected keys "
              f"(e.g. {unexpected[:3]})")

    seq_model.to(device)
    seq_model.eval()

    meta: dict[str, Any] = {
        "model_id": model_id,
        "label": _MODEL_LABELS.get(model_id, model_id),
        "config": saved_cfg,
        "epoch": epoch,
        "param_count": count_parameters(seq_model),
        "ckpt_path": str(checkpoint_path),
        "has_persistence": has_persistence,
        "has_external": has_external,
    }
    return seq_model, meta


# ---------------------------------------------------------------------------
# Batch-level metric extraction
# ---------------------------------------------------------------------------

def _extract_batch_metrics(
    stacked_logits: torch.Tensor,
    labels: torch.Tensor,
    turn_mask: torch.Tensor,
    all_infos: list[dict[str, Any]],
    has_external: bool,
) -> dict[str, Any]:
    """Extract per-turn accuracy and resource usage from a single batch."""
    criterion = nn.CrossEntropyLoss(reduction="none")
    B, T, C = stacked_logits.shape

    per_turn_correct: list[int] = []
    per_turn_total: list[int] = []
    per_turn_loss: list[float] = []

    for t in range(T):
        logits_t = stacked_logits[:, t]
        labels_t = labels[:, t]
        mask_t = turn_mask[:, t]

        loss_t = criterion(logits_t, labels_t)
        preds_t = logits_t.argmax(dim=-1)
        correct_t = int(((preds_t == labels_t) & (mask_t > 0)).sum().item())
        valid_t = int((mask_t > 0).sum().item())
        mean_loss_t = float((loss_t * mask_t).sum().item()) / max(1, valid_t)

        per_turn_correct.append(correct_t)
        per_turn_total.append(valid_t)
        per_turn_loss.append(mean_loss_t)

    # Resource metrics averaged across turns
    halt_acc: float = 0.0
    active_acc: float = 0.0
    ext_acc: float = 0.0

    for t_info in all_infos:
        halt_steps = t_info.get("halt_steps")
        if halt_steps is not None:
            halt_acc += halt_steps.float().mean().item()

        act_gates = t_info.get("activity_gates")
        if act_gates is not None:
            active_acc += (act_gates > 0.5).float().mean().item()

        if has_external:
            ws_traj = t_info.get("workspace_trajectory", [])
            if ws_traj:
                last_ws = ws_traj[-1]
                if hasattr(last_ws, "mask"):
                    ext_acc += last_ws.mask.float().sum(dim=-1).mean().item()

    n_t = max(1, len(all_infos))
    return {
        "per_turn_correct": per_turn_correct,
        "per_turn_total": per_turn_total,
        "per_turn_loss": per_turn_loss,
        "mean_halt_steps": halt_acc / n_t,
        "mean_active_slots": active_acc / n_t,
        "mean_ext_writes": ext_acc / n_t,
    }


# ---------------------------------------------------------------------------
# Full dataset evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: SequentialReasoningModel,
    dataset: SequenceReasoningDataset,
    meta: dict[str, Any],
    batch_size: int = 32,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run full evaluation over the shared dataset for one model.

    Returns:
        Comprehensive metric dict for this model.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
        drop_last=False,
    )

    has_external = meta["has_external"]
    n_turns = N_EVAL_TURNS

    agg_correct: list[int] = [0] * n_turns
    agg_total: list[int] = [0] * n_turns
    agg_loss: list[float] = [0.0] * n_turns
    agg_halt: float = 0.0
    agg_active: float = 0.0
    agg_ext: float = 0.0
    n_batches: int = 0

    for batch in loader:
        facts = batch["facts"].to(device)
        fact_mask = batch["fact_mask"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        labels = batch["labels"].to(device)

        stacked_logits, all_infos = model(facts, fact_mask, turn_mask)
        bm = _extract_batch_metrics(
            stacked_logits, labels, turn_mask, all_infos, has_external
        )

        T_batch = min(n_turns, len(bm["per_turn_correct"]))
        for t in range(T_batch):
            agg_correct[t] += bm["per_turn_correct"][t]
            agg_total[t] += bm["per_turn_total"][t]
            agg_loss[t] += bm["per_turn_loss"][t] * bm["per_turn_total"][t]

        agg_halt += bm["mean_halt_steps"]
        agg_active += bm["mean_active_slots"]
        agg_ext += bm["mean_ext_writes"]
        n_batches += 1

    n_batches = max(1, n_batches)
    total_correct = sum(agg_correct)
    total_valid = sum(agg_total)

    per_turn_acc = [float(c) / max(1, t) for c, t in zip(agg_correct, agg_total)]
    per_turn_loss = [agg_loss[t] / max(1, agg_total[t]) for t in range(n_turns)]
    overall_accuracy = float(total_correct) / max(1, total_valid)
    mean_loss = sum(agg_loss) / max(1, total_valid)

    return {
        "model_id": meta["model_id"],
        "label": meta["label"],
        "overall_accuracy": overall_accuracy,
        "per_turn_accuracy": per_turn_acc,
        "per_turn_loss": per_turn_loss,
        "mean_loss": mean_loss,
        "mean_halt_steps": agg_halt / n_batches,
        "mean_active_slots": agg_active / n_batches,
        "mean_ext_writes": agg_ext / n_batches,
        "param_count": meta["param_count"],
        "has_persistence": meta["has_persistence"],
        "has_external": meta["has_external"],
        "epoch": meta["epoch"],
        "ckpt_path": meta["ckpt_path"],
    }


# ---------------------------------------------------------------------------
# Latency profiling
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_latency(
    model: SequentialReasoningModel,
    dataset: SequenceReasoningDataset,
    n_samples: int = 64,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
) -> float:
    """Return mean inference latency in milliseconds per sequence."""
    from torch.utils.data import Subset
    subset = Subset(dataset, list(range(min(n_samples, len(dataset)))))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )

    for i, batch in enumerate(loader):
        if i >= WARMUP_BATCHES:
            break
        facts = batch["facts"].to(device)
        fact_mask = batch["fact_mask"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        _ = model(facts, fact_mask, turn_mask)

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    total_seqs = 0
    start = time.perf_counter()
    for batch in loader:
        facts = batch["facts"].to(device)
        fact_mask = batch["fact_mask"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        _ = model(facts, fact_mask, turn_mask)
        total_seqs += facts.shape[0]

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return float(elapsed_ms) / max(1, total_seqs)


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

class Phase8Evaluator:
    """High-level orchestrator for the Phase 8A comparative benchmark.

    Example::

        ev = Phase8Evaluator(device="cpu")
        ev.register("A", "results/vector_baseline/best.pt")
        ev.register("B", "results/static_graph/best.pt")
        ev.register("C", "LIMINAL_results/persistence_incremental/best.pt")
        ev.register("D", "LIMINAL_results/externalisation_comparison/best.pt")
        results = ev.run()
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        n_eval_sequences: int = N_EVAL_SEQUENCES,
        n_turns: int = N_EVAL_TURNS,
        seed: int = HOLDOUT_SEED,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self._registrations: list[tuple[str, Path]] = []
        self._dataset: SequenceReasoningDataset | None = None
        self._vocab: Vocabulary | None = None
        self._n_sequences = n_eval_sequences
        self._n_turns = n_turns
        self._seed = seed

    def register(self, model_id: str, checkpoint_path: str | Path) -> None:
        """Register a model checkpoint for evaluation."""
        self._registrations.append((model_id, Path(checkpoint_path)))

    def _get_dataset(self) -> tuple[SequenceReasoningDataset, Vocabulary]:
        if self._dataset is None:
            print(
                f"  Generating shared eval dataset "
                f"({self._n_sequences} seqs × {self._n_turns} turns, "
                f"seed={self._seed}) …"
            )
            self._dataset, self._vocab = build_eval_dataset(
                n_sequences=self._n_sequences,
                n_turns=self._n_turns,
                seed=self._seed,
            )
            print(f"  Dataset ready: {len(self._dataset)} sequences.")
        return self._dataset, self._vocab

    def run(self) -> list[dict[str, Any]]:
        """Evaluate all registered checkpoints. Returns list of metric dicts."""
        dataset, _ = self._get_dataset()
        results: list[dict[str, Any]] = []

        for model_id, ckpt_path in self._registrations:
            print(f"\n  ── Model {model_id}: {_MODEL_LABELS.get(model_id, model_id)}")
            print(f"     Checkpoint : {ckpt_path}")

            model, meta = load_model(model_id, ckpt_path, device=self.device)

            metrics = evaluate_model(
                model, dataset, meta,
                batch_size=self.batch_size,
                device=self.device,
            )
            metrics["latency_ms_per_seq"] = measure_latency(
                model, dataset, device=self.device
            )

            _log_result(metrics)
            results.append(metrics)
            del model

        return results


def _log_result(m: dict[str, Any]) -> None:
    """Pretty-print a single model's evaluation result."""
    print(f"     Overall Acc : {m['overall_accuracy'] * 100:.2f}%")
    turn_str = "  |  ".join(
        f"T{i+1}: {a * 100:.1f}%"
        for i, a in enumerate(m["per_turn_accuracy"])
    )
    print(f"     Per-Turn    : {turn_str}")
    print(f"     Test Loss   : {m['mean_loss']:.4f}")
    print(f"     Params      : {m['param_count']['trainable']:,}")
    print(f"     Latency     : {m['latency_ms_per_seq']:.2f} ms/seq")
    if m["has_persistence"] or m["mean_halt_steps"] > 0:
        print(f"     Halt Steps  : {m['mean_halt_steps']:.2f}")
        print(f"     Active Slots: {m['mean_active_slots']:.3f}")
    if m["has_external"]:
        print(f"     Ext Writes  : {m['mean_ext_writes']:.3f} slots/step")
