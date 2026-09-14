from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 classification accuracy."""
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return float(correct) / max(1, labels.size(0))


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def evaluate_model_detailed(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run comprehensive evaluation returning per-family and overall accuracy and loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total_samples = 0
    family_stats: dict[str, dict[str, Any]] = {}

    with torch.no_grad():
        for batch in loader:
            facts = batch["facts"].to(device)
            fact_mask = batch["fact_mask"].to(device)
            labels = batch["label"].to(device)
            metadata = batch["metadata"]

            logits, _ = model(facts, fact_mask)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=-1)
            is_correct = (preds == labels).cpu().numpy()
            correct += int(is_correct.sum())
            total_samples += bs

            for i, meta in enumerate(metadata):
                fam = meta.get("task_family", "default")
                if fam not in family_stats:
                    family_stats[fam] = {"correct": 0, "total": 0}
                family_stats[fam]["total"] += 1
                if is_correct[i]:
                    family_stats[fam]["correct"] += 1

    overall_acc = float(correct) / max(1, total_samples)
    mean_loss = float(total_loss) / max(1, total_samples)

    breakdown = {}
    for fam, stats in family_stats.items():
        breakdown[fam] = {
            "accuracy": float(stats["correct"]) / max(1, stats["total"]),
            "total_samples": stats["total"],
        }

    return {
        "loss": mean_loss,
        "accuracy": overall_acc,
        "total_samples": total_samples,
        "breakdown": breakdown,
    }
