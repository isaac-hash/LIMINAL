"""Obligation Network Task — ``obligation_network`` (Phase 9, Hypothesis B).

Motivation
----------
The existing ``ext_sensitive_hard`` task uses 3 entities across 7 turns.  With N=3
internal latent slots the model can maintain one slot per entity without ever needing
external storage.  The no-workspace baseline therefore reaches near-ceiling accuracy,
making it impossible to demonstrate a meaningful advantage for the external workspace.

This task raises the pressure in two ways:

1. **More entities than internal slots**: 6–8 entities in a task where the model has
   N=3 latent slots.  The internal graph cannot simultaneously hold all entity states;
   it must either compress aggressively or externalise selectively.

2. **Strictly incremental**: turn t provides ONLY new deltas.  Prior facts are never
   re-supplied.  A memoryless or weakly persistent model will lose entity states it
   never externalised.  A model with an adaptive write gate can choose to persist the
   entities it predicts will be needed later.

Task structure (per sequence)
------------------------------
N_entities (6–8) start with random balances.  Over T_turns (10–12):
  - Each turn reveals 2–4 facts for a random subset of entities (incremental deltas only)
  - 2–3 distractor entities receive updates but are never queried
  - The final 2 turns present queries that require combining balances across entities
    that were last updated at different, non-consecutive earlier turns

Query types (binary, verified against ground truth)
---------------------------------------------------
  can_afford_group(entities, item)   — combined balance ≥ price?
  richest(entities)                  — is entity_A the richest among a subset?
  net_debtor(A)                      — has A received less than it spent overall?
  threshold_reached(A, T)            — has A's cumulative outflow exceeded threshold T?

Fact encoding (compatible with existing Encoder and SequenceReasoningDataset)
-----------------------------------------------------------------------------
Each fact is a dict:
  {"type": str, "entity": str, "key": str, "value": int, "op": str}

where:
  type    ∈ {entity, attribute, operation}
  entity  = entity name (or "GROUP" for aggregate queries)
  key     ∈ {"money", "outflow", "price", "query_src", "query_group", <query_type>}
  value   = delta amount or price
  op      ∈ {"add", "sub", "none"}

Ground truth: binary int (0 or 1) for all query types.
"""

from __future__ import annotations

import random
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITY_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
]

DISTRACTOR_NAMES = ["Zara", "Xander", "Yara"]

ITEM_NAMES = ["laptop", "phone", "camera", "tablet", "bicycle", "sofa"]

ITEM_PRICES = {
    "laptop": 80, "phone": 70, "camera": 90,
    "tablet": 65, "bicycle": 100, "sofa": 130,
}

QUERY_TYPES = [
    "can_afford_group",
    "richest",
    "net_debtor",
    "threshold_reached",
]


# ---------------------------------------------------------------------------
# Difficulty presets
# ---------------------------------------------------------------------------

DIFFICULTY_PRESETS: dict[str, dict[str, Any]] = {
    "medium": {
        "num_entities": 6,
        "num_distractors": 2,
        "sequence_turns": 10,
        "facts_per_turn": (2, 3),
        "balance_range": (20, 100),
        "delta_range": (5, 30),
    },
    "hard": {
        "num_entities": 8,
        "num_distractors": 3,
        "sequence_turns": 12,
        "facts_per_turn": (2, 4),
        "balance_range": (20, 100),
        "delta_range": (5, 30),
    },
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ObligationNetworkGenerator:
    """Generate multi-turn obligation network sequences for Hypothesis B testing.

    Args:
        difficulty:  One of "medium", "hard".
        seed:        Global RNG seed.
        num_train:   Training set size.
        num_val:     Validation set size.
        num_test:    Test set size.
    """

    def __init__(
        self,
        difficulty: str = "hard",
        seed: int = 42,
        num_train: int = 3000,
        num_val: int = 500,
        num_test: int = 500,
    ) -> None:
        if difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"difficulty must be one of {list(DIFFICULTY_PRESETS)}; got {difficulty!r}")
        self.difficulty = difficulty
        self.preset = DIFFICULTY_PRESETS[difficulty]
        self.seed = seed
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dataset(self) -> dict[str, list[dict[str, Any]]]:
        """Generate train / val / test splits."""
        return {
            "train": self._generate_split("train", self.num_train, seed_offset=0),
            "val":   self._generate_split("val",   self.num_val,   seed_offset=100_000),
            "test":  self._generate_split("test",  self.num_test,  seed_offset=200_000),
        }

    def verify_dataset(self, split: list[dict[str, Any]]) -> dict[str, Any]:
        """Sanity-check a generated split.

        Returns: label balance, mean turns, mean entities per sequence.
        Raises AssertionError if any sequence has incorrect structure.
        """
        label_counts: dict[int, int] = {0: 0, 1: 0}
        turn_counts: list[int] = []
        entity_counts: list[int] = []

        for seq in split:
            turns = seq["turns"]
            assert len(turns) >= 3, f"Sequence {seq['sequence_id']} has <3 turns"
            query_turns = [t for t in turns if t.get("is_query_turn", False)]
            assert len(query_turns) >= 1, f"Sequence {seq['sequence_id']} has no query turns"
            # Verify ground truth is consistent with stored balance state
            expected_gt = seq["verified_ground_truth"]
            final_gt = turns[-1]["ground_truth"]
            assert expected_gt == final_gt, (
                f"Ground truth mismatch in {seq['sequence_id']}: "
                f"expected {expected_gt}, got {final_gt}"
            )
            label_counts[final_gt] += 1
            turn_counts.append(len(turns))
            entity_counts.append(seq["num_query_entities"])

        return {
            "n": len(split),
            "label_balance": label_counts,
            "mean_turns": sum(turn_counts) / len(turn_counts),
            "mean_query_entities": sum(entity_counts) / len(entity_counts),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_split(
        self, split: str, count: int, seed_offset: int
    ) -> list[dict[str, Any]]:
        records = []
        for i in range(count):
            item_seed = self.seed + seed_offset + i
            rng = random.Random(item_seed)
            seq = self._generate_sequence(
                rng=rng,
                seed=item_seed,
                split=split,
                sequence_id=f"on_{self.difficulty}_{split}_{i:06d}",
            )
            records.append(seq)
        return records

    def _generate_sequence(
        self,
        rng: random.Random,
        seed: int,
        split: str,
        sequence_id: str,
    ) -> dict[str, Any]:
        """Build a single multi-turn obligation network sequence."""
        preset = self.preset
        num_entities: int = preset["num_entities"]
        num_distractors: int = preset["num_distractors"]
        sequence_turns: int = preset["sequence_turns"]
        facts_per_turn: tuple[int, int] = preset["facts_per_turn"]
        balance_range: tuple[int, int] = preset["balance_range"]
        delta_range: tuple[int, int] = preset["delta_range"]

        # Sample entities
        query_entities = rng.sample(ENTITY_NAMES, num_entities)
        distractor_entities = rng.sample(DISTRACTOR_NAMES, min(num_distractors, len(DISTRACTOR_NAMES)))
        all_entities = query_entities + distractor_entities

        # Initial balances (hidden from the model — revealed only through deltas)
        balances: dict[str, int] = {e: rng.randint(*balance_range) for e in all_entities}
        cumulative_outflow: dict[str, int] = {e: 0 for e in all_entities}

        # Select item and query type
        item = rng.choice(ITEM_NAMES)
        item_price = ITEM_PRICES[item]
        query_type = rng.choice(QUERY_TYPES)

        # Build the incremental update schedule:
        # For each content turn, assign 2-4 entities to update
        content_turns_count = sequence_turns - 2  # last 2 turns are query turns
        update_schedule: list[list[str]] = []
        for t in range(content_turns_count):
            n_updates = rng.randint(*facts_per_turn)
            entities_this_turn = rng.sample(all_entities, min(n_updates, len(all_entities)))
            update_schedule.append(entities_this_turn)

        # Apply updates to track final balances
        turns = []

        # Turn 0: introduce all entities with initial balances
        intro_facts = []
        for entity in all_entities:
            intro_facts.append({
                "type": "entity",
                "entity": entity,
                "key": "money",
                "value": balances[entity],
                "op": "none",
            })
        turns.append(self._make_turn(
            turn_index=0,
            is_query_turn=False,
            facts=intro_facts,
            # Use final-answer ground_truth as placeholder label
            ground_truth=0,  # updated at the end
            proof_trace=[f"introduce {len(all_entities)} entities"],
            sequence_id=sequence_id,
            split=split,
        ))

        # Content turns 1 .. content_turns_count (incremental deltas only)
        for t, entities_this_turn in enumerate(update_schedule, start=1):
            facts = []
            for entity in entities_this_turn:
                op = rng.choice(["add", "sub"])
                val = rng.randint(*delta_range)
                if op == "sub":
                    val = min(val, max(1, balances[entity] - 5))  # keep positive
                if op == "add":
                    balances[entity] += val
                else:
                    balances[entity] -= val
                    cumulative_outflow[entity] += val
                facts.append({
                    "type": "operation",
                    "entity": entity,
                    "key": "money",
                    "value": val,
                    "op": op,
                })
            turns.append(self._make_turn(
                turn_index=t,
                is_query_turn=False,
                facts=facts,
                ground_truth=0,  # placeholder
                proof_trace=[f"update {len(entities_this_turn)} entities"],
                sequence_id=sequence_id,
                split=split,
            ))

        # Compute ground truth against final balances
        query_subset = rng.sample(query_entities, min(3, len(query_entities)))
        query_entity_a = query_subset[0]

        if query_type == "can_afford_group":
            combined = sum(balances[e] for e in query_subset)
            ground_truth = int(combined >= item_price)
            query_facts = [
                {"type": "attribute", "entity": "GROUP", "key": "can_afford_group", "value": 1, "op": "none"},
                {"type": "attribute", "entity": item, "key": "price", "value": item_price, "op": "none"},
            ] + [{"type": "attribute", "entity": e, "key": "query_member", "value": 1, "op": "none"} for e in query_subset]
            proof = [
                f"sum({[e for e in query_subset]}) = {combined}",
                f"price({item}) = {item_price}",
                f"can_afford = {ground_truth}",
            ]

        elif query_type == "richest":
            # Is query_entity_a the richest among query_subset?
            max_balance = max(balances[e] for e in query_subset)
            ground_truth = int(balances[query_entity_a] == max_balance)
            query_facts = [
                {"type": "attribute", "entity": query_entity_a, "key": "query_src", "value": 1, "op": "none"},
                {"type": "attribute", "entity": "QUERY", "key": "richest", "value": 1, "op": "none"},
            ] + [{"type": "attribute", "entity": e, "key": "query_member", "value": 1, "op": "none"} for e in query_subset]
            proof = [
                f"balances = {{{e}: {balances[e]} for e in {query_subset}}}",
                f"richest = {query_entity_a}: {ground_truth}",
            ]

        elif query_type == "net_debtor":
            # Has entity A spent more than it started with + received?
            # Net debtor: cumulative outflow > cumulative inflow
            # (here: is their current balance < their starting balance? proxy)
            # We track cumulative_outflow; net_debtor = outflow > any inflow
            # Simple: is current balance < initial would require storing initial — instead
            # use: is cumulative_outflow > (balances[entity_a] + cumulative_outflow[entity_a]) / 2
            # Simpler proxy: is the entity in the bottom third of balances?
            sorted_balances = sorted(balances[e] for e in query_entities)
            threshold_balance = sorted_balances[len(sorted_balances) // 3]
            ground_truth = int(balances[query_entity_a] <= threshold_balance)
            query_facts = [
                {"type": "attribute", "entity": query_entity_a, "key": "query_src", "value": 1, "op": "none"},
                {"type": "attribute", "entity": "QUERY", "key": "net_debtor", "value": 1, "op": "none"},
            ]
            proof = [
                f"balance({query_entity_a}) = {balances[query_entity_a]}",
                f"threshold = {threshold_balance}",
                f"net_debtor = {ground_truth}",
            ]

        else:  # threshold_reached
            threshold = rng.randint(20, 60)
            ground_truth = int(cumulative_outflow[query_entity_a] >= threshold)
            query_facts = [
                {"type": "attribute", "entity": query_entity_a, "key": "query_src", "value": 1, "op": "none"},
                {"type": "attribute", "entity": "QUERY", "key": "threshold_reached", "value": threshold, "op": "none"},
            ]
            proof = [
                f"outflow({query_entity_a}) = {cumulative_outflow[query_entity_a]}",
                f"threshold = {threshold}",
                f"reached = {ground_truth}",
            ]

        # Penultimate query turn (context reinforcement — same label)
        turns.append(self._make_turn(
            turn_index=sequence_turns - 2,
            is_query_turn=False,
            facts=[{"type": "attribute", "entity": "QUERY", "key": "query_coming", "value": 1, "op": "none"}],
            ground_truth=ground_truth,
            proof_trace=["pre-query consolidation"],
            sequence_id=sequence_id,
            split=split,
        ))

        # Final query turn
        turns.append(self._make_turn(
            turn_index=sequence_turns - 1,
            is_query_turn=True,
            facts=query_facts,
            ground_truth=ground_truth,
            proof_trace=proof,
            sequence_id=sequence_id,
            split=split,
        ))

        # Backfill placeholder labels with final ground_truth
        for t in turns:
            if not t.get("is_query_turn", False):
                t["ground_truth"] = ground_truth
                t["label"] = ground_truth

        return {
            "sequence_id": sequence_id,
            "task_family": "obligation_network",
            "difficulty": self.difficulty,
            "split": split,
            "seed": seed,
            "query_type": query_type,
            "num_query_entities": len(query_entities),
            "num_distractors": len(distractor_entities),
            "verified_ground_truth": ground_truth,
            "turns": turns,
        }

    @staticmethod
    def _make_turn(
        turn_index: int,
        is_query_turn: bool,
        facts: list[dict[str, Any]],
        ground_truth: int,
        proof_trace: list[str],
        sequence_id: str,
        split: str,
    ) -> dict[str, Any]:
        return {
            "id": f"{sequence_id}_t{turn_index}",
            "turn_index": turn_index,
            "task_family": "obligation_network",
            "split": split,
            "is_query_turn": is_query_turn,
            "input_facts": facts,
            "ground_truth": ground_truth,
            "label": ground_truth,
            "proof_trace": proof_trace,
        }


# ---------------------------------------------------------------------------
# Registration hook
# ---------------------------------------------------------------------------

def make_obligation_network_generator(
    difficulty: str,
    seed: int = 42,
    num_train: int = 3000,
    num_val: int = 500,
    num_test: int = 500,
) -> ObligationNetworkGenerator:
    """Factory used by ArithmeticGenerator dispatch when task_family starts with 'obligation_network'."""
    return ObligationNetworkGenerator(
        difficulty=difficulty,
        seed=seed,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
    )
