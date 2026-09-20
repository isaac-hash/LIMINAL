from typing import Any
import torch
from torch.utils.data import Dataset


class Vocabulary:
    """Vocabulary mapping string tokens to integer indices across all fact fields."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.fact_types = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1, "entity": 2, "attribute": 3, "operation": 4}
        self.entities = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.keys = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.ops = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1, "none": 2, "add": 3, "sub": 4}

        self.idx2fact_type = {v: k for k, v in self.fact_types.items()}
        self.idx2entity = {v: k for k, v in self.entities.items()}
        self.idx2key = {v: k for k, v in self.keys.items()}
        self.idx2op = {v: k for k, v in self.ops.items()}

    def build_from_records(self, records: list[dict[str, Any]]) -> None:
        """Scan all records and populate entities and keys."""
        flat_records = []
        for r in records:
            if "turns" in r:
                flat_records.extend(r["turns"])
            else:
                flat_records.append(r)

        for r in flat_records:
            for fact in r.get("input_facts", []):
                f_type = fact.get("type", "")
                if f_type and f_type not in self.fact_types:
                    idx = len(self.fact_types)
                    self.fact_types[f_type] = idx
                    self.idx2fact_type[idx] = f_type

                entity = fact.get("name") or fact.get("entity")
                if entity and entity not in self.entities:
                    idx = len(self.entities)
                    self.entities[entity] = idx
                    self.idx2entity[idx] = entity

                key = fact.get("key")
                if key and key not in self.keys:
                    idx = len(self.keys)
                    self.keys[key] = idx
                    self.idx2key[idx] = key

                op = fact.get("op")
                if op and op not in self.ops:
                    idx = len(self.ops)
                    self.ops[op] = idx
                    self.idx2op[idx] = op

    @property
    def total_vocab_size(self) -> int:
        """Sum of all categorical token spaces."""
        return len(self.fact_types) + len(self.entities) + len(self.keys) + len(self.ops)

    def encode_fact(self, fact: dict[str, Any]) -> list[int]:
        """Encode a single fact dict to [type_idx, entity_idx, key_idx, value_int, op_idx]."""
        f_type = fact.get("type", self.UNK_TOKEN)
        type_idx = self.fact_types.get(f_type, self.fact_types[self.UNK_TOKEN])

        entity = fact.get("name") or fact.get("entity", self.PAD_TOKEN)
        entity_idx = self.entities.get(entity, self.entities[self.UNK_TOKEN])

        key = fact.get("key", self.PAD_TOKEN)
        key_idx = self.keys.get(key, self.keys[self.UNK_TOKEN])

        value_int = int(fact.get("value", 0))

        op = fact.get("op", "none")
        op_idx = self.ops.get(op, self.ops[self.UNK_TOKEN])

        return [type_idx, entity_idx, key_idx, value_int, op_idx]

    def decode_fact(self, encoded: list[int]) -> dict[str, Any]:
        """Decode [type_idx, entity_idx, key_idx, value_int, op_idx] back to a dictionary."""
        type_idx, entity_idx, key_idx, value_int, op_idx = encoded
        f_type = self.idx2fact_type.get(type_idx, self.UNK_TOKEN)
        entity = self.idx2entity.get(entity_idx, self.PAD_TOKEN)
        key = self.idx2key.get(key_idx, self.PAD_TOKEN)
        op = self.idx2op.get(op_idx, "none")

        if f_type == "entity":
            return {"type": "entity", "name": entity}
        elif f_type == "attribute":
            return {"type": "attribute", "entity": entity, "key": key, "value": value_int}
        elif f_type == "operation":
            return {"type": "operation", "op": op, "entity": entity, "key": key, "value": value_int}
        else:
            return {"type": f_type, "entity": entity, "key": key, "value": value_int, "op": op}


class ReasoningDataset(Dataset):
    """PyTorch Dataset for synthetic multi-step reasoning tasks."""

    def __init__(self, records: list[dict[str, Any]], vocab: Vocabulary):
        self.records = records
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        raw_facts = record.get("input_facts", [])
        encoded_facts = [self.vocab.encode_fact(f) for f in raw_facts]

        facts_tensor = torch.tensor(encoded_facts, dtype=torch.long)  # [num_facts, 5]
        label = torch.tensor(record["ground_truth"], dtype=torch.long if isinstance(record["ground_truth"], int) else torch.float32)

        return {
            "facts": facts_tensor,
            "label": label,
            "id": record["id"],
            "task_family": record.get("task_family", ""),
            "difficulty": record.get("difficulty", {}),
            "proof_trace": record.get("proof_trace", []),
        }


def collate_reasoning_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length fact sequences into padded batch tensors.
    
    Returns:
        {
            'facts': Tensor[B, max_facts, 5],
            'fact_mask': Tensor[B, max_facts] (1.0 for valid, 0.0 for pad),
            'label': Tensor[B],
            'metadata': list of dicts
        }
    """
    batch_size = len(batch)
    max_facts = max(item["facts"].shape[0] for item in batch)
    max_facts = max(max_facts, 1)

    padded_facts = torch.zeros((batch_size, max_facts, 5), dtype=torch.long)
    fact_mask = torch.zeros((batch_size, max_facts), dtype=torch.float32)
    labels = torch.stack([item["label"] for item in batch])

    for i, item in enumerate(batch):
        n_facts = item["facts"].shape[0]
        if n_facts > 0:
            padded_facts[i, :n_facts] = item["facts"]
            fact_mask[i, :n_facts] = 1.0

    metadata = [
        {
            "id": item["id"],
            "task_family": item["task_family"],
            "difficulty": item["difficulty"],
            "proof_trace": item["proof_trace"],
        }
        for item in batch
    ]

    return {
        "facts": padded_facts,
        "fact_mask": fact_mask,
        "label": labels,
        "metadata": metadata,
    }
