"""Stage 2 Capacity-Starvation Task — ``ext_sensitive_hard``.

Motivation
----------
The standard ``ext_sensitive`` task uses 1–2 entities per turn, so a
3-slot internal latent graph can track full state without ever touching
the external workspace.  This task raises the entity load so a model
with only N=3 internal slots **cannot** hold all live entities
simultaneously and **must** externalise to maintain accuracy.

Task structure (per sequence — 7 turns)
---------------------------------------
Three people (P1, P2, P3) maintain individual wallets across 7 turns.
Each turn introduces facts about 1–2 of the 3 people.

  T0  Introduce P1 balance.            Query: balance >= 50?
  T1  Introduce P2, P2 transacts.      Query: can P2 afford item_A?
  T2  Introduce P3.                    Query: combined P1+P2 >= 100?
  T3  P1 and P3 both transact.         Query: is P1 NOT the richest? (binary)
  T4  P2 transacts.                    Query: can P3 afford item_B?
  T5  No new entity.                   Query: group total (P1+P2+P3) >= 150?
  T6  (item_C injected as fact only)   Query: can group afford item_C?

Entity pressure
---------------
At T5 and T6 the model needs the current balances of all three people.
Those balances were written across T0–T4 in different turns.  With N=3
internal slots the model can only hold 3 simultaneous entity vectors;
after each turn the graph updates in-place and the previous turn's
"memory trace" must either persist via the gated blend OR be offloaded
to the external workspace.  The externalisation-sensitive pressure is
highest at T5/T6 where recall across all 5 prior turns is required.

Label scheme: binary 0/1 throughout (T3 ternary collapsed to 0/1).
"""

from __future__ import annotations

import random
from typing import Any


class ExtSensitiveHardGenerator:
    """Generate 7-turn high-entity-load sequences for capacity-starvation tests.

    Args:
        seed:      Global RNG seed (per-example seeds are derived from this).
        num_train: Training set size.
        num_val:   Validation set size.
        num_test:  Test set size.
    """

    PEOPLE = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
    ITEMS_A = ["shoe", "book", "watch", "hat"]
    ITEMS_B = ["laptop", "phone", "camera", "tablet"]
    ITEMS_C = ["bicycle", "jacket", "headphones", "sofa"]
    PRICES: dict[str, int] = {
        # cheap
        "shoe": 30, "book": 15, "watch": 50, "hat": 20,
        # medium
        "laptop": 80, "phone": 70, "camera": 90, "tablet": 65,
        # expensive
        "bicycle": 100, "jacket": 60, "headphones": 45, "sofa": 130,
    }

    def __init__(
        self,
        seed: int = 42,
        num_train: int = 3000,
        num_val: int = 500,
        num_test: int = 500,
    ) -> None:
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
                sequence_id=f"esh_{split}_{i:06d}",
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
        """Build a single 7-turn sequence with 3 people and 3 items."""
        # Sample three distinct people and one item per tier
        p1, p2, p3 = rng.sample(self.PEOPLE, 3)
        item_a = rng.choice(self.ITEMS_A)
        item_b = rng.choice(self.ITEMS_B)
        item_c = rng.choice(self.ITEMS_C)
        price_a = self.PRICES[item_a]
        price_b = self.PRICES[item_b]
        price_c = self.PRICES[item_c]

        # Starting balances: each in [20, 100]
        m1 = rng.randint(20, 100)
        m2 = rng.randint(20, 100)
        m3 = rng.randint(20, 100)

        turns = []

        # ── Turn 0: introduce P1 ─────────────────────────────────────
        turns.append(self._make_turn(
            turn_index=0,
            facts=[
                {"type": "entity",    "name": p1},
                {"type": "attribute", "entity": p1, "key": "money", "value": m1},
            ],
            label=int(m1 >= 50),
            proof_trace=[f"money({p1})={m1}", f"label={int(m1 >= 50)}"],
            sequence_id=sequence_id, split=split,
        ))

        # ── Turn 1: introduce P2, P2 transacts; can P2 afford item_A? ─
        delta1 = rng.choice([-1, 1]) * rng.randint(5, 30)
        m2 += delta1
        turns.append(self._make_turn(
            turn_index=1,
            facts=[
                {"type": "entity",    "name": p2},
                {"type": "attribute", "entity": p2, "key": "money",  "value": m2 - delta1},
                {"type": "attribute", "entity": p2, "key": "delta",  "value": delta1},
                {"type": "attribute", "entity": item_a, "key": "price", "value": price_a},
            ],
            label=int(m2 >= price_a),
            proof_trace=[
                f"money({p2})_before={m2 - delta1}",
                f"delta={delta1}",
                f"money({p2})={m2}",
                f"price({item_a})={price_a}",
                f"can_afford={m2 >= price_a}",
            ],
            sequence_id=sequence_id, split=split,
        ))

        # ── Turn 2: introduce P3; combined P1+P2 >= 100? ─────────────
        turns.append(self._make_turn(
            turn_index=2,
            facts=[
                {"type": "entity",    "name": p3},
                {"type": "attribute", "entity": p3, "key": "money", "value": m3},
            ],
            label=int((m1 + m2) >= 100),
            proof_trace=[
                f"money({p1})={m1}",
                f"money({p2})={m2}",
                f"combined={m1 + m2}",
                f"label={int((m1 + m2) >= 100)}",
            ],
            sequence_id=sequence_id, split=split,
        ))

        # ── Turn 3: P1 and P3 both transact; is P1 NOT the richest? ──
        delta3_p1 = rng.choice([-1, 1]) * rng.randint(5, 25)
        delta3_p3 = rng.choice([-1, 1]) * rng.randint(5, 25)
        m1 += delta3_p1
        m3 += delta3_p3
        richest_idx = [m1, m2, m3].index(max(m1, m2, m3))
        label_t3 = int(richest_idx != 0)   # 0 = P1 is richest, 1 = someone else
        turns.append(self._make_turn(
            turn_index=3,
            facts=[
                {"type": "attribute", "entity": p1, "key": "delta", "value": delta3_p1},
                {"type": "attribute", "entity": p3, "key": "delta", "value": delta3_p3},
            ],
            label=label_t3,
            proof_trace=[
                f"money({p1})={m1}",
                f"money({p2})={m2}",
                f"money({p3})={m3}",
                f"richest_not_p1={label_t3}",
            ],
            sequence_id=sequence_id, split=split,
        ))

        # ── Turn 4: P2 transacts; can P3 afford item_B? ──────────────
        delta4 = rng.choice([-1, 1]) * rng.randint(5, 20)
        m2 += delta4
        turns.append(self._make_turn(
            turn_index=4,
            facts=[
                {"type": "attribute", "entity": p2, "key": "delta",  "value": delta4},
                {"type": "attribute", "entity": item_b, "key": "price", "value": price_b},
            ],
            label=int(m3 >= price_b),
            proof_trace=[
                f"money({p3})={m3}",
                f"price({item_b})={price_b}",
                f"can_afford={m3 >= price_b}",
            ],
            sequence_id=sequence_id, split=split,
        ))

        # ── Turn 5: group total (P1+P2+P3) >= 150? ───────────────────
        group_total = m1 + m2 + m3
        turns.append(self._make_turn(
            turn_index=5,
            facts=[
                {"type": "attribute", "entity": "group", "key": "query", "value": 1},
            ],
            label=int(group_total >= 150),
            proof_trace=[
                f"money({p1})={m1}",
                f"money({p2})={m2}",
                f"money({p3})={m3}",
                f"total={group_total}",
                f"label={int(group_total >= 150)}",
            ],
            sequence_id=sequence_id, split=split,
        ))

        # ── Turn 6: can group afford item_C? (demands full history) ──
        turns.append(self._make_turn(
            turn_index=6,
            facts=[
                {"type": "attribute", "entity": item_c, "key": "price", "value": price_c},
            ],
            label=int(group_total >= price_c),
            proof_trace=[
                f"group_total={group_total}",
                f"price({item_c})={price_c}",
                f"label={int(group_total >= price_c)}",
            ],
            sequence_id=sequence_id, split=split,
        ))

        return {
            "sequence_id": sequence_id,
            "task_family": "ext_sensitive_hard",
            "split": split,
            "seed": seed,
            "turns": turns,
            "final_balances": {p1: m1, p2: m2, p3: m3},
            "group_total": group_total,
        }

    @staticmethod
    def _make_turn(
        turn_index: int,
        facts: list[dict[str, Any]],
        label: int,
        proof_trace: list[str],
        sequence_id: str,
        split: str,
    ) -> dict[str, Any]:
        """Format a single turn dict compatible with SequenceReasoningDataset."""
        return {
            "id": f"{sequence_id}_t{turn_index}",
            "turn_index": turn_index,
            "task_family": "ext_sensitive_hard",
            "split": split,
            "input_facts": facts,
            "ground_truth": label,
            "label": label,
            "proof_trace": proof_trace,
        }
