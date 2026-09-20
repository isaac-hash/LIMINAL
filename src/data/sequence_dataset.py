from typing import Any
import torch
from torch.utils.data import Dataset
from src.data.dataset import Vocabulary


class SequenceReasoningDataset(Dataset):
    """PyTorch Dataset for multi-turn sequence reasoning tasks (Phase 4).

    Each sample is a full multi-turn sequence consisting of an ordered list of turns,
    where each turn has its own input facts, question/task, and ground truth.
    """

    def __init__(self, records: list[dict[str, Any]], vocab: Vocabulary):
        self.records = records
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        turns_data = []

        for t_idx, turn in enumerate(record.get("turns", [])):
            raw_facts = turn.get("input_facts", [])
            encoded_facts = [self.vocab.encode_fact(f) for f in raw_facts]
            facts_tensor = torch.tensor(encoded_facts, dtype=torch.long)  # [num_facts, 5]
            gt = turn["ground_truth"]
            label = torch.tensor(gt, dtype=torch.long if isinstance(gt, int) else torch.float32)

            turns_data.append({
                "facts": facts_tensor,
                "label": label,
                "id": turn.get("id", f"{record.get('sequence_id', index)}_t{t_idx}"),
                "turn_index": turn.get("turn_index", t_idx),
                "proof_trace": turn.get("proof_trace", []),
            })

        return {
            "sequence_id": record.get("sequence_id", f"seq_{index}"),
            "task_family": record.get("task_family", "affordability_sequence"),
            "turns": turns_data,
        }


def collate_sequence_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-turn and variable-fact sequences into padded batch tensors.

    Returns:
        {
            'facts':     Tensor[B, max_turns, max_facts, 5],
            'fact_mask': Tensor[B, max_turns, max_facts] (1.0 for real facts, 0.0 for pad),
            'turn_mask': Tensor[B, max_turns] (1.0 for real turns, 0.0 for pad),
            'labels':    Tensor[B, max_turns],
            'label':     Tensor[B, max_turns] (alias for compatibility),
            'metadata':  list of sequence metadata dicts
        }
    """
    batch_size = len(batch)
    max_turns = max(len(item["turns"]) for item in batch) if batch else 1
    max_turns = max(max_turns, 1)

    max_facts = 1
    for item in batch:
        for turn in item["turns"]:
            max_facts = max(max_facts, turn["facts"].shape[0])

    facts = torch.zeros((batch_size, max_turns, max_facts, 5), dtype=torch.long)
    fact_mask = torch.zeros((batch_size, max_turns, max_facts), dtype=torch.float32)
    turn_mask = torch.zeros((batch_size, max_turns), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_turns), dtype=torch.long)

    for i, item in enumerate(batch):
        for t, turn in enumerate(item["turns"]):
            n_facts = turn["facts"].shape[0]
            if n_facts > 0:
                facts[i, t, :n_facts] = turn["facts"]
                fact_mask[i, t, :n_facts] = 1.0
            turn_mask[i, t] = 1.0
            labels[i, t] = turn["label"]

    metadata = [
        {
            "sequence_id": item["sequence_id"],
            "task_family": item["task_family"],
            "turns": [
                {
                    "id": turn["id"],
                    "turn_index": turn["turn_index"],
                    "proof_trace": turn["proof_trace"],
                }
                for turn in item["turns"]
            ],
        }
        for item in batch
    ]

    return {
        "facts": facts,
        "fact_mask": fact_mask,
        "turn_mask": turn_mask,
        "labels": labels,
        "label": labels,
        "metadata": metadata,
    }
