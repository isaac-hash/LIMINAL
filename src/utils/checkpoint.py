from pathlib import Path
import shutil
from typing import Any
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    path: str | Path,
    config: Any = None,
    drive_path: str | Path | None = None,
) -> None:
    """Save model checkpoint. Optionally copy to Google Drive / external storage path.
    
    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer (optional).
        epoch: Current epoch index.
        metrics: Dictionary of metric values (loss, accuracy, etc.).
        path: Target file path to write checkpoint.
        config: Configuration object or dict (optional).
        drive_path: Optional secondary path (e.g. Google Drive) to mirror checkpoint.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "config": config,
    }

    torch.save(state, target_path)

    if drive_path is not None:
        drive_target = Path(drive_path)
        drive_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target_path, drive_target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> tuple[int, dict[str, Any], Any]:
    """Load checkpoint into model and optimizer.
    
    Args:
        path: Path to checkpoint file.
        model: PyTorch model instance to populate.
        optimizer: Optional PyTorch optimizer instance to populate.
        device: Device to map storage locations.
        
    Returns:
        tuple: (epoch, metrics, config)
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    config = checkpoint.get("config", None)

    return epoch, metrics, config
