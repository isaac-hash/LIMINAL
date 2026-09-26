"""Phase 7 — Externalisation-Sensitive Task Family.

Generates 5-turn multi-hop arithmetic sequences where correctly answering
the final question requires the model to have externalised intermediate
results from earlier turns.  Without an active external workspace, the
internal latent graph must carry all intermediate state across 5 turns,
creating a memory bottleneck that the learned write gate can relieve.

Task structure (per sequence)
-----------------------------
Turn 1: person gets initial money              → answer: current money  (regression → binned label)
Turn 2: person spends or earns at shop A       → answer: can buy item A?
Turn 3: person finds/loses extra money         → answer: current money (binned)
Turn 4: cumulative transaction at shop B       → answer: can buy item B?
Turn 5: final tally across all events         → answer: can buy item C? (requires all prior turns)

The key feature is that Turn 5 requires facts from Turns 1–3 that may no
longer be in the immediate-context window.  A model with an active, well-used
external workspace can look up slot contents written in earlier turns.

Label scheme: binary.
    0 = cannot afford  (money < item_price)
    1 = can afford     (money >= item_price)

This file exposes:
    ExternalisationSensitiveGenerator   — generates sequences
    It integrates with ArithmeticGenerator via the task_family='ext_sensitive'
    dispatch path.
"""

import random
from typing import Any


class ExternalisationSensitiveGenerator:
    """Generate 5-turn externalisation-sensitive reasoning sequences.

    Each sequence has turns that depend on all previous turns' numeric state.
    The final-turn answer can only be computed correctly if intermediate
    monetary state was retained across all turns.

    Args:
        seed: Global random seed (per-example seeds derived from this).
        num_train: Training set size.
        num_val:   Validation set size.
        num_test:  Test set size.
    """

    ENTITIES = ["John", "Alice", "Bob", "Emma", "David", "Sarah", "Michael", "Olivia"]
    ITEMS_A = ["shoe", "book", "watch"]          # cheap-medium items
    ITEMS_B = ["laptop", "phone", "camera"]       # medium-expensive items
    ITEMS_C = ["bicycle", "jacket", "headphones"] # final-turn test items
    PRICES = {
        "shoe": 30, "book": 15, "watch": 50,
        "laptop": 80, "phone": 70, "camera": 90,
        "bicycle": 100, "jacket": 60, "headphones": 45,
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

    def generate_dataset(self) -> dict[str, list[dict[str, Any]]]:
        """Generate train / val / test splits."""
        return {
            "train": self._generate_split("train", self.num_train, seed_offset=0),
            "val":   self._generate_split("val",   self.num_val,   seed_offset=100_000),
            "test":  self._generate_split("test",  self.num_test,  seed_offset=200_000),
        }

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
                sequence_id=f"ext_{split}_{i:06d}",
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
        """Generate a single 5-turn externalisation-sensitive sequence."""
        person = rng.choice(self.ENTITIES)
        item_a = rng.choice(self.ITEMS_A)
        item_b = rng.choice(self.ITEMS_B)
        item_c = rng.choice(self.ITEMS_C)
        price_a = self.PRICES[item_a]
        price_b = self.PRICES[item_b]
        price_c = self.PRICES[item_c]

        # Initial money: enough range so ~50% can afford each item
        money = rng.randint(20, 120)

        turns = []

        # ── Turn 1: starting balance ─────────────────────────────────────
        proof_1 = [f"money({person})={money}"]
        label_1 = int(money >= 50)  # binary: above/below 50 threshold
        facts_1 = [
            {"type": "entity", "name": person},
            {"type": "attribute", "entity": person, "key": "money", "value": money},
        ]
        turns.append(self._make_turn(
            turn_index=0,
            facts=facts_1,
            label=label_1,
            proof_trace=proof_1,
            sequence_id=sequence_id,
            split=split,
            task_family="ext_sensitive",
        ))

        # ── Turn 2: shop A transaction ───────────────────────────────────
        delta_2 = rng.choice([-1, 1]) * rng.randint(5, 25)
        money += delta_2
        op_str = f"+{delta_2}" if delta_2 > 0 else str(delta_2)
        proof_2 = [f"money({person}){op_str}", f"item={item_a},price={price_a}", f"money={money}"]
        label_2 = int(money >= price_a)
        facts_2 = [
            {"type": "entity", "name": person},
            {"type": "attribute", "entity": person, "key": "delta", "value": delta_2},
            {"type": "attribute", "entity": item_a, "key": "price", "value": price_a},
        ]
        turns.append(self._make_turn(
            turn_index=1,
            facts=facts_2,
            label=label_2,
            proof_trace=proof_2,
            sequence_id=sequence_id,
            split=split,
            task_family="ext_sensitive",
        ))

        # ── Turn 3: found/lost extra cash ────────────────────────────────
        delta_3 = rng.choice([-1, 1]) * rng.randint(5, 30)
        money += delta_3
        op_str = f"+{delta_3}" if delta_3 > 0 else str(delta_3)
        proof_3 = [f"money({person}){op_str}", f"money={money}"]
        label_3 = int(money >= 50)
        facts_3 = [
            {"type": "attribute", "entity": person, "key": "extra", "value": delta_3},
        ]
        turns.append(self._make_turn(
            turn_index=2,
            facts=facts_3,
            label=label_3,
            proof_trace=proof_3,
            sequence_id=sequence_id,
            split=split,
            task_family="ext_sensitive",
        ))

        # ── Turn 4: shop B transaction ───────────────────────────────────
        delta_4 = rng.choice([-1, 1]) * rng.randint(5, 20)
        money += delta_4
        op_str = f"+{delta_4}" if delta_4 > 0 else str(delta_4)
        proof_4 = [f"money({person}){op_str}", f"item={item_b},price={price_b}", f"money={money}"]
        label_4 = int(money >= price_b)
        facts_4 = [
            {"type": "attribute", "entity": person, "key": "delta", "value": delta_4},
            {"type": "attribute", "entity": item_b, "key": "price", "value": price_b},
        ]
        turns.append(self._make_turn(
            turn_index=3,
            facts=facts_4,
            label=label_4,
            proof_trace=proof_4,
            sequence_id=sequence_id,
            split=split,
            task_family="ext_sensitive",
        ))

        # ── Turn 5: final affordability check (requires full history) ────
        # This is the externalisation-demanding turn: money state is the
        # cumulative sum of all prior deltas, which must have been retained.
        proof_5 = [
            f"money({person})={money}",
            f"item={item_c},price={price_c}",
            f"can_afford={money >= price_c}",
        ]
        label_5 = int(money >= price_c)
        facts_5 = [
            {"type": "attribute", "entity": item_c, "key": "price", "value": price_c},
        ]
        turns.append(self._make_turn(
            turn_index=4,
            facts=facts_5,
            label=label_5,
            proof_trace=proof_5,
            sequence_id=sequence_id,
            split=split,
            task_family="ext_sensitive",
        ))

        return {
            "sequence_id": sequence_id,
            "task_family": "ext_sensitive",
            "split": split,
            "seed": seed,
            "turns": turns,
            "final_money": money,
        }

    @staticmethod
    def _make_turn(
        turn_index: int,
        facts: list[dict[str, Any]],
        label: int,
        proof_trace: list[str],
        sequence_id: str,
        split: str,
        task_family: str,
    ) -> dict[str, Any]:
        """Format a single turn dict compatible with SequenceReasoningDataset."""
        return {
            "id": f"{sequence_id}_t{turn_index}",
            "turn_index": turn_index,
            "task_family": task_family,
            "split": split,
            "input_facts": facts,
            "ground_truth": label,
            "label": label,
            "proof_trace": proof_trace,
        }
