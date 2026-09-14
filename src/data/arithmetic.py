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
        splits = {
            "train": self._generate_split("train", self.config.num_train, seed_offset=0),
            "val": self._generate_split("val", self.config.num_val, seed_offset=100000),
            "test": self._generate_split("test", self.config.num_test, seed_offset=200000),
        }
        return splits

    def _generate_split(self, split_name: str, count: int, seed_offset: int) -> list[dict[str, Any]]:
        records = []
        for i in range(count):
            item_seed = self.seed + seed_offset + i
            rng = random.Random(item_seed)
            difficulty = {
                "num_ops": self.config.max_operations,
                "num_entities": self.config.max_entities,
                "num_distractors": int(self.config.distractor_ratio * self.config.max_entities),
            }

            if self.config.task_family == "affordability":
                # Ensure 50/50 balance across the split
                target_balance = (i % 2 == 0)
                record = self._generate_affordability(rng, difficulty, item_seed, split_name, f"aff_{split_name}_{i:06d}", target_balance)
            elif self.config.task_family == "simple_arithmetic":
                record = self._generate_simple_arithmetic(rng, difficulty, item_seed, split_name, f"arith_{split_name}_{i:06d}")
            elif self.config.task_family == "multi_step":
                record = self._generate_multi_step(rng, difficulty, item_seed, split_name, f"mstep_{split_name}_{i:06d}")
            elif self.config.task_family == "comparison":
                target_balance = (i % 2 == 0)
                record = self._generate_comparison(rng, difficulty, item_seed, split_name, f"comp_{split_name}_{i:06d}", target_balance)
            else:
                raise ValueError(f"Unknown task family: {self.config.task_family}")

            records.append(record)
        return records

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
