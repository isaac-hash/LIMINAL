"""Phase 2B: Activity Analysis utilities.

Provides functions to:
  1. Collect slot activity trajectories over a dataset.
  2. Compute per-slot mean activity and entropy statistics.
  3. Plot slot activation heatmaps (slot x step, coloured by mean activity).
  4. Plot per-class activity distributions (do different answer types use different slots?).

All functions are side-effect free (return data) except the plot helpers,
which accept an ax or output_path argument.
"""
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_activity_data(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run model on loader and collect activity gate trajectories.

    Args:
        model:       Trained ReasoningModel (must be in activity mode).
        loader:      DataLoader for evaluation split.
        device:      Computation device.
        max_batches: Optional cap on number of batches (for speed).

    Returns:
        dict with keys:
            "A_finals"         : np.ndarray [n_samples, N]   — final step activities
            "A_trajectories"   : np.ndarray [n_samples, T, N]— per-step activities
            "labels"           : np.ndarray [n_samples]       — ground truth labels
            "preds"            : np.ndarray [n_samples]       — model predictions
            "correct"          : np.ndarray [n_samples] bool  — correctness mask
            "n_slots"          : int
            "n_steps"          : int
    """
    model.eval()

    all_A_finals: list[np.ndarray] = []
    all_A_traj: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            facts = batch["facts"].to(device)
            fact_mask = batch["fact_mask"].to(device)
            labels = batch["label"].to(device)

            logits, info = model(facts, fact_mask)
            preds = logits.argmax(dim=-1)

            # A_final: [B, N]
            A_final = info.get("A_final")
            if A_final is None:
                raise RuntimeError(
                    "Model info does not contain 'A_final'. "
                    "Ensure the model was built with activity_config.enabled=True."
                )

            # activity_trajectory: list of T tensors [B, N]
            A_traj_list = info.get("activity_trajectory", [])
            # Stack to [B, T, N]
            if A_traj_list:
                A_traj = torch.stack(A_traj_list, dim=1).cpu().numpy()
            else:
                # Fallback: expand final activity
                A_traj = A_final.unsqueeze(1).cpu().numpy()

            all_A_finals.append(A_final.cpu().numpy())
            all_A_traj.append(A_traj)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    A_finals = np.concatenate(all_A_finals, axis=0)        # [n, N]
    A_traj = np.concatenate(all_A_traj, axis=0)            # [n, T, N]
    labels_arr = np.concatenate(all_labels, axis=0)        # [n]
    preds_arr = np.concatenate(all_preds, axis=0)          # [n]

    return {
        "A_finals": A_finals,
        "A_trajectories": A_traj,
        "labels": labels_arr,
        "preds": preds_arr,
        "correct": (labels_arr == preds_arr),
        "n_slots": A_finals.shape[1],
        "n_steps": A_traj.shape[1],
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_activity_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Summarise collected activity data.

    Returns:
        dict with:
            "mean_per_slot"   : [N]     mean activity across samples & steps
            "std_per_slot"    : [N]     std of activity per slot
            "mean_per_step"   : [T]     mean activity across samples & slots
            "entropy_per_slot": [N]     binary entropy H(a) per slot
            "utilization"     : float   fraction of slots with mean activity > 0.5
            "effective_slots" : float   sum of mean activities (soft count of active slots)
    """
    A = data["A_finals"]          # [n, N]
    A_traj = data["A_trajectories"]  # [n, T, N]

    mean_slot = A.mean(axis=0)    # [N]
    std_slot = A.std(axis=0)      # [N]
    mean_step = A_traj.mean(axis=(0, 2))  # [T]

    eps = 1e-8
    p = np.clip(mean_slot, eps, 1.0 - eps)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))  # [N]

    return {
        "mean_per_slot": mean_slot,
        "std_per_slot": std_slot,
        "mean_per_step": mean_step,
        "entropy_per_slot": entropy,
        "utilization": float((mean_slot > 0.5).mean()),
        "effective_slots": float(mean_slot.sum()),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_activity_heatmap(
    data: dict[str, Any],
    output_path: str | None = None,
    title: str = "Slot Activity Heatmap (mean over dataset)",
) -> None:
    """Plot a [T x N] heatmap of mean slot activities per reasoning step.

    Args:
        data:        Output of collect_activity_data().
        output_path: If given, save figure here. Otherwise show interactively.
        title:       Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A_traj = data["A_trajectories"]  # [n, T, N]
    mean_heatmap = A_traj.mean(axis=0)  # [T, N]

    T, N = mean_heatmap.shape
    fig, ax = plt.subplots(figsize=(max(6, N * 0.8), max(3, T * 0.7)))
    im = ax.imshow(mean_heatmap, aspect="auto", cmap="plasma", vmin=0.0, vmax=1.0)

    ax.set_xlabel("Slot index")
    ax.set_ylabel("Reasoning step")
    ax.set_xticks(range(N))
    ax.set_yticks(range(T))
    ax.set_xticklabels([f"S{i}" for i in range(N)])
    ax.set_yticklabels([f"T{t}" for t in range(T)])
    ax.set_title(title, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean activity gate value")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved -> {output_path}")
    else:
        plt.show()
    plt.close()


def plot_per_class_activity(
    data: dict[str, Any],
    class_names: list[str] | None = None,
    output_path: str | None = None,
    title: str = "Mean Slot Activity by Predicted Class",
) -> None:
    """Bar chart: mean final slot activity split by predicted class.

    Shows whether different answer classes recruit different slot subsets.

    Args:
        data:        Output of collect_activity_data().
        class_names: Optional list of class label strings.
        output_path: If given, save figure here.
        title:       Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = data["A_finals"]      # [n, N]
    preds = data["preds"]     # [n]
    N = data["n_slots"]

    unique_classes = sorted(set(preds.tolist()))
    if class_names is None:
        class_names = [f"Class {c}" for c in unique_classes]

    x = np.arange(N)
    width = 0.8 / max(1, len(unique_classes))
    palette = ["#4C9BE8", "#E8834C", "#6CC66C", "#C66CC6"]

    fig, ax = plt.subplots(figsize=(max(8, N), 4))

    for idx, cls in enumerate(unique_classes):
        mask = preds == cls
        if mask.sum() == 0:
            continue
        mean_act = A[mask].mean(axis=0)  # [N]
        offset = (idx - len(unique_classes) / 2 + 0.5) * width
        color = palette[idx % len(palette)]
        label = class_names[idx] if idx < len(class_names) else f"Class {cls}"
        ax.bar(x + offset, mean_act, width=width * 0.9, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Slot index")
    ax.set_ylabel("Mean activity gate value")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in range(N)])
    ax.set_ylim([0.0, 1.05])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Per-class activity plot saved -> {output_path}")
    else:
        plt.show()
    plt.close()


def plot_activity_over_steps(
    data: dict[str, Any],
    output_path: str | None = None,
    title: str = "Mean Slot Activity Across Reasoning Steps",
) -> None:
    """Line plot: mean activity per slot across T reasoning steps.

    Helps identify slots that 'wake up' or 'shut down' progressively.

    Args:
        data:        Output of collect_activity_data().
        output_path: If given, save figure here.
        title:       Figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A_traj = data["A_trajectories"]  # [n, T, N]
    mean_over_samples = A_traj.mean(axis=0)  # [T, N]
    T, N = mean_over_samples.shape

    cmap = matplotlib.colormaps["tab10"].resampled(N)
    fig, ax = plt.subplots(figsize=(8, 4))

    for slot_idx in range(N):
        ax.plot(
            range(T),
            mean_over_samples[:, slot_idx],
            marker="o",
            markersize=4,
            label=f"S{slot_idx}",
            color=cmap(slot_idx),
            linewidth=1.5,
        )

    ax.set_xlabel("Reasoning step")
    ax.set_ylabel("Mean activity gate value")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"T{t}" for t in range(T)])
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Step activity plot saved -> {output_path}")
    else:
        plt.show()
    plt.close()
