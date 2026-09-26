import json
from pathlib import Path
import random
from typing import Any
from src.utils.config import DataConfig


class ArithmeticGenerator:
    """Synthetic dataset generator for relational and multi-step arithmetic tasks."""

    ENTITIES = ["John", "Alice", "Bob", "Emma", "David", "Sarah", "Michael", "Olivia"]
    ITEMS = ["shoe", "book", "laptop", "watch", "phone", "jacket", "bicycle", "camera"]
    ATTRIBUTES = ["money", "price", "count", "weight", "distance"]
    OPERATIONS = ["add", "sub"]

    def __init__(self, config: DataConfig | None = None, seed: int = 42):
        self.config = config or DataConfig()
        self.seed = seed

    def generate_dataset(self) -> dict[str, list[dict[str, Any]]]:
        """Generate full train, val, and test splits with deterministic seeding."""
        if self.config.task_family == "affordability_sequence":
            return self.generate_sequence_dataset()
        if self.config.task_family == "ext_sensitive":
            from src.data.externalisation_sensitive import ExternalisationSensitiveGenerator
            gen = ExternalisationSensitiveGenerator(
                seed=self.seed,
                num_train=self.config.num_train,
                num_val=self.config.num_val,
                num_test=self.config.num_test,
            )
            return gen.generate_dataset()
        splits = {
            "train": self._generate_split("train", self.config.num_train, seed_offset=0),
            "val": self._generate_split("val", self.config.num_val, seed_offset=100000),
            "test": self._generate_split("test", self.config.num_test, seed_offset=200000),
        }
        return splits

    def _generate_split(self, split_name: str, count: int, seed_offset: int) -> list[dict[str, Any]]:
        records = []

        if self.config.task_family == "mixed":
            # Interleave families in round-robin order so each family is equally
            # represented.  E.g. with families=("affordability", "multi_step"),
            # even indices get affordability and odd indices get multi_step.
            families = list(self.config.mixed_families)
            if not families:
                raise ValueError("mixed_families must be non-empty when task_family='mixed'")
        else:
            families = None  # single-family mode

        for i in range(count):
            item_seed = self.seed + seed_offset + i
            rng = random.Random(item_seed)
            difficulty = {
                "num_ops": self.config.max_operations,
                "num_entities": self.config.max_entities,
                "num_distractors": int(self.config.distractor_ratio * self.config.max_entities),
            }

            # Resolve which task family to use for this example
            if families is not None:
                family = families[i % len(families)]
            else:
                family = self.config.task_family

            record = self._dispatch_generate(family, rng, difficulty, item_seed, split_name, i)
            records.append(record)
        return records

    def generate_sequence_dataset(self) -> dict[str, list[dict[str, Any]]]:
        """Generate multi-turn sequence splits (used when task_family='affordability_sequence')."""
        return {
            "train": self._generate_sequence_split("train", self.config.num_train, seed_offset=0),
            "val": self._generate_sequence_split("val", self.config.num_val, seed_offset=100000),
            "test": self._generate_sequence_split("test", self.config.num_test, seed_offset=200000),
        }

    def _generate_sequence_split(self, split_name: str, count: int, seed_offset: int) -> list[dict[str, Any]]:
        """Generate `count` multi-turn sequences for the given split."""
        records = []
        for i in range(count):
            item_seed = self.seed + seed_offset + i
            rng = random.Random(item_seed)
            seq = self._generate_affordability_sequence(
                rng=rng,
                seed=item_seed,
                split=split_name,
                sequence_id=f"seq_{split_name}_{i:06d}",
            )
            records.append(seq)
        return records

    def _dispatch_generate(
        self,
        family: str,
        rng: random.Random,
        difficulty: dict[str, int],
        item_seed: int,
        split_name: str,
        i: int,
    ) -> dict[str, Any]:
        """Route generation to the correct task-family generator."""
        if family == "affordability":
            target_balance = (i % 2 == 0)
            return self._generate_affordability(rng, difficulty, item_seed, split_name, f"aff_{split_name}_{i:06d}", target_balance)
        elif family == "simple_arithmetic":
            return self._generate_simple_arithmetic(rng, difficulty, item_seed, split_name, f"arith_{split_name}_{i:06d}")
        elif family == "multi_step":
            return self._generate_multi_step(rng, difficulty, item_seed, split_name, f"mstep_{split_name}_{i:06d}")
        elif family == "comparison":
            target_balance = (i % 2 == 0)
            return self._generate_comparison(rng, difficulty, item_seed, split_name, f"comp_{split_name}_{i:06d}", target_balance)
        elif family == "affordability_sequence":
            # Single-example call: generate a sequence and return only the first turn
            # for compatibility with single-turn training. Use generate_sequence_dataset
            # for full multi-turn training.
            rng2 = random.Random(item_seed)
            seq = self._generate_affordability_sequence(rng2, item_seed, split_name, f"seq_{split_name}_{i:06d}")
            return seq["turns"][0]  # first turn only
        elif family == "ext_sensitive":
            # Delegate to ExternalisationSensitiveGenerator for a single turn.
            from src.data.externalisation_sensitive import ExternalisationSensitiveGenerator
            gen = ExternalisationSensitiveGenerator(seed=item_seed, num_train=1, num_val=1, num_test=1)
            seq = gen._generate_sequence(rng, item_seed, split_name, f"ext_{split_name}_{i:06d}")
            return seq["turns"][0]
        else:
            raise ValueError(f"Unknown task family: {family}")

    def _generate_affordability(
        self, rng: random.Random, difficulty: dict[str, int], seed: int, split: str, record_id: str, target_can_afford: bool
    ) -> dict[str, Any]:
        """Generate an affordability reasoning problem: Person has money, receives/spends amounts, can they buy item?"""
        person = rng.choice(self.ENTITIES)
        item = rng.choice(self.ITEMS)

        # Base starting money
        start_money = rng.randint(20, 60)
        current_money = start_money
        proof_trace = [f"money({person})={start_money}"]

        input_facts = [
            {"type": "entity", "name": person},
            {"type": "attribute", "entity": person, "key": "money", "value": start_money},
        ]

        # Operations
        num_ops = difficulty["num_ops"]
        for _ in range(num_ops):
            op = rng.choice(self.OPERATIONS)
            val = rng.randint(5, 30)
            if op == "add":
                current_money += val
                proof_trace.append(f"money({person})+={val}")
            else:
                # keep money positive
                val = min(val, max(1, current_money - 5))
                current_money -= val
                proof_trace.append(f"money({person})-={val}")

            input_facts.append(
                {"type": "operation", "op": op, "entity": person, "key": "money", "value": val}
            )

        proof_trace.append(f"money({person})={current_money}")

        # Set item price according to target_can_afford
        if target_can_afford:
            # price <= current_money
            delta = rng.randint(0, min(15, current_money - 1))
            price = current_money - delta
            ground_truth = 1
        else:
            # price > current_money
            delta = rng.randint(1, 20)
            price = current_money + delta
            ground_truth = 0

        input_facts.append({"type": "attribute", "entity": item, "key": "price", "value": price})
        proof_trace.append(f"price({item})={price}")
        proof_trace.append(f"{current_money}>={price}")
        proof_trace.append("true" if ground_truth == 1 else "false")

        # Distractor entities if configured
        distractor_count = difficulty.get("num_distractors", 0)
        available_distractors = [e for e in self.ENTITIES if e != person]
        rng.shuffle(available_distractors)
        for d_idx in range(min(distractor_count, len(available_distractors))):
            d_entity = available_distractors[d_idx]
            d_val = rng.randint(10, 50)
            input_facts.insert(rng.randint(0, len(input_facts)), {"type": "entity", "name": d_entity})
            input_facts.insert(rng.randint(0, len(input_facts)), {"type": "attribute", "entity": d_entity, "key": "money", "value": d_val})

        return {
            "id": record_id,
            "task_family": "affordability",
            "difficulty": difficulty,
            "input_facts": input_facts,
            "ground_truth": ground_truth,
            "proof_trace": proof_trace,
            "seed": seed,
            "split": split,
        }

    def _generate_simple_arithmetic(
        self, rng: random.Random, difficulty: dict[str, int], seed: int, split: str, record_id: str
    ) -> dict[str, Any]:
        """Simple arithmetic: A + B or A - B."""
        person = rng.choice(self.ENTITIES)
        a = rng.randint(10, 50)
        b = rng.randint(5, 30)
        op = rng.choice(self.OPERATIONS)
        result = a + b if op == "add" else a - b

        input_facts = [
            {"type": "entity", "name": person},
            {"type": "attribute", "entity": person, "key": "count", "value": a},
            {"type": "operation", "op": op, "entity": person, "key": "count", "value": b},
        ]
        proof_trace = [f"count({person})={a}", f"count({person}) {op}= {b}", f"result={result}"]

        return {
            "id": record_id,
            "task_family": "simple_arithmetic",
            "difficulty": difficulty,
            "input_facts": input_facts,
            "ground_truth": result,
            "proof_trace": proof_trace,
            "seed": seed,
            "split": split,
        }

    def _generate_multi_step(
        self, rng: random.Random, difficulty: dict[str, int], seed: int, split: str, record_id: str
    ) -> dict[str, Any]:
        """Multi-step arithmetic chain across 1 entity."""
        person = rng.choice(self.ENTITIES)
        current = rng.randint(20, 50)
        input_facts = [
            {"type": "entity", "name": person},
            {"type": "attribute", "entity": person, "key": "count", "value": current},
        ]
        proof_trace = [f"count({person})={current}"]

        for _ in range(difficulty["num_ops"]):
            op = rng.choice(self.OPERATIONS)
            val = rng.randint(5, 25)
            if op == "add":
                current += val
            else:
                val = min(val, current)
                current -= val
            input_facts.append({"type": "operation", "op": op, "entity": person, "key": "count", "value": val})
            proof_trace.append(f"count({person}) {op}= {val}")

        proof_trace.append(f"result={current}")

        return {
            "id": record_id,
            "task_family": "multi_step",
            "difficulty": difficulty,
            "input_facts": input_facts,
            "ground_truth": current,
            "proof_trace": proof_trace,
            "seed": seed,
            "split": split,
        }

    def _generate_comparison(
        self, rng: random.Random, difficulty: dict[str, int], seed: int, split: str, record_id: str, target_a_greater: bool
    ) -> dict[str, Any]:
        """Compare final attributes of entity A vs entity B."""
        p_a, p_b = rng.sample(self.ENTITIES, 2)
        val_a = rng.randint(20, 70)
        val_b = rng.randint(20, 70)
        if target_a_greater:
            val_a = max(val_a, val_b + rng.randint(1, 10))
            gt = 1
        else:
            val_b = max(val_b, val_a + rng.randint(1, 10))
            gt = 0

        input_facts = [
            {"type": "entity", "name": p_a},
            {"type": "attribute", "entity": p_a, "key": "score", "value": val_a},
            {"type": "entity", "name": p_b},
            {"type": "attribute", "entity": p_b, "key": "score", "value": val_b},
        ]
        proof_trace = [f"score({p_a})={val_a}", f"score({p_b})={val_b}", f"{val_a}>{val_b}", "true" if gt == 1 else "false"]

        return {
            "id": record_id,
            "task_family": "comparison",
            "difficulty": difficulty,
            "input_facts": input_facts,
            "ground_truth": gt,
            "proof_trace": proof_trace,
            "seed": seed,
            "split": split,
        }

    def _generate_affordability_sequence(
        self,
        rng: random.Random,
        seed: int,
        split: str,
        sequence_id: str,
    ) -> dict[str, Any]:
        """Generate a multi-turn affordability sequence (Option A: standalone facts per turn).

        Each turn is a STANDALONE affordability problem that includes ALL cumulative facts
        (so a reset model can also solve each turn independently).  The persistent model
        should gain an efficiency advantage by reusing its prior workspace state rather
        than re-computing from scratch.

        Structure:
          Turn 0: Alice has <start>. [op_1]. Can she afford <item> (price=<p>)?  → Y/N
          Turn 1: Alice has <start>. [op_1]. [op_2]. Same item. Can she afford?  → Y/N
          Turn 2: Alice has <start>. [op_1]. [op_2]. [op_3]. Same item.           → Y/N
          ...

        Args:
            rng:          seeded RNG for this sequence
            seed:         integer seed for record metadata
            split:        "train" | "val" | "test"
            sequence_id:  unique ID for this sequence

        Returns:
            {"sequence_id": str, "turns": [turn_dict, ...], "task_family": "affordability_sequence"}
        """
        num_turns = self.config.sequence_turns
        ops_per_turn = self.config.ops_per_turn

        person = rng.choice(self.ENTITIES)
        item = rng.choice(self.ITEMS)
        price = rng.randint(20, 80)
        start_money = rng.randint(20, 60)

        # Pre-generate ALL operations for the full sequence
        total_ops = num_turns * ops_per_turn
        all_ops: list[tuple[str, int]] = []
        running = start_money
        for _ in range(total_ops):
            op = rng.choice(self.OPERATIONS)
            val = rng.randint(5, 20)
            if op == "sub":
                val = min(val, max(1, running - 5))  # keep running positive
            running = running + val if op == "add" else running - val
            all_ops.append((op, val))

        # Build per-turn records
        turns: list[dict[str, Any]] = []
        incremental = getattr(self.config, "incremental_turns", False)

        for t in range(num_turns):
            ops_so_far = all_ops[: (t + 1) * ops_per_turn]
            new_ops = all_ops[t * ops_per_turn : (t + 1) * ops_per_turn]

            # Recompute running budget and proof trace for this turn
            budget = start_money
            proof_trace = [f"money({person})={start_money}"]
            for op, val in ops_so_far:
                budget = budget + val if op == "add" else budget - val
                proof_trace.append(f"money({person}) {op}= {val}")

            ground_truth = 1 if budget >= price else 0
            proof_trace.append(f"budget={budget}  price={price}  afford={'yes' if ground_truth else 'no'}")

            # Construct input facts according to incremental mode
            if incremental and t > 0:
                # Incremental regime: ONLY new operations and target price (requires memory)
                input_facts: list[dict[str, Any]] = []
                for op, val in new_ops:
                    input_facts.append({"type": "operation", "op": op, "entity": person, "key": "money", "value": val})
                input_facts.append({"type": "attribute", "entity": item, "key": "price", "value": price})
            else:
                # Standalone regime: full history of facts up to this turn
                input_facts = [
                    {"type": "entity", "name": person},
                    {"type": "attribute", "entity": person, "key": "money", "value": start_money},
                ]
                for op, val in ops_so_far:
                    input_facts.append({"type": "operation", "op": op, "entity": person, "key": "money", "value": val})
                input_facts.append({"type": "attribute", "entity": item, "key": "price", "value": price})

            turns.append({
                "id": f"{sequence_id}_t{t}",
                "task_family": "affordability_sequence",
                "turn_index": t,
                "sequence_id": sequence_id,
                "input_facts": input_facts,
                "ground_truth": ground_truth,
                "proof_trace": proof_trace,
                "seed": seed,
                "split": split,
            })

        return {
            "sequence_id": sequence_id,
            "task_family": "affordability_sequence",
            "turns": turns,
            "seed": seed,
            "split": split,
        }

    @staticmethod
    def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
        """Save records as JSONL file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    @staticmethod
    def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
        """Load records from JSONL file."""
        path = Path(path)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
