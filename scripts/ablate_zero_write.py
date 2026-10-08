"""Zero-Write Ablation & Slot-Corruption Causal Test.

Three conditions evaluated on the same checkpoint:

  1. NORMAL     -- learned gate runs as trained (baseline eval)
  2. ZERO_WRITE -- gate is forced to 0 for every slot/step at eval time;
                   workspace remains empty throughout the sequence.
                   If accuracy == NORMAL: writes are unused by the reader.
                   If accuracy < NORMAL:  writes causally support predictions.
  3. CORRUPT    -- gate fires normally but written records are zeroed before
                   reads (Phase-6-style causal corruption).  Tests whether
                   the *content* of writes matters vs the mask alone.

Usage:
    python scripts/ablate_zero_write.py \
        --config  configs/experiments/fair_hard_n3_lam001.yaml \
        --checkpoint results/fair_hard_n3_lam001/best.pt \
        --output-dir results/ablations/fair_hard_n3_lam001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.utils.device import print_hardware_info
from src.utils.config import load_config, set_seed
from src.utils.checkpoint import load_checkpoint, resolve_checkpoint_path
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary
from src.data.sequence_dataset import SequenceReasoningDataset, collate_sequence_batch
from src.models.sequential_model import SequentialReasoningModel


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def _find_controller(model: SequentialReasoningModel):
    try:
        return model.base_model.workspace.write_controller
    except AttributeError:
        return None


class _ZeroWriteHook:
    """Replace gate_net with one that always returns -1e9, suppressing all writes."""

    def __init__(self, model):
        self._ctrl = _find_controller(model)

    def __enter__(self):
        if self._ctrl is None:
            raise RuntimeError("LearnedWriteController not found in model.")

        orig_gate_net = self._ctrl.gate_net
        self._orig_gate_net = orig_gate_net

        class _NegInfNet(nn.Module):
            def forward(self, x):
                B_N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
                return torch.full(
                    (x.shape[0], x.shape[1], 1) if x.dim() == 3 else (x.shape[0], 1),
                    fill_value=-1e9,
                    device=x.device,
                    dtype=x.dtype,
                )

        self._ctrl.gate_net = _NegInfNet()
        return self

    def __exit__(self, *_):
        self._ctrl.gate_net = self._orig_gate_net


class _CorruptWriteHook:
    """After each write step zero all written records before returning."""

    def __init__(self, model):
        self._ctrl = _find_controller(model)

    def __enter__(self):
        if self._ctrl is None:
            raise RuntimeError("LearnedWriteController not found in model.")

        orig_forward = self._ctrl.forward

        def _corrupt_forward(workspace, V, A, step):
            ws_out, info = orig_forward(workspace, V, A, step)
            # Zero records of all occupied slots (mask=True)
            mask_f = ws_out.mask.float().unsqueeze(-1)   # [B, M, 1]
            ws_out.records = ws_out.records * (1.0 - mask_f)
            return ws_out, info

        self._orig_forward = orig_forward
        self._ctrl.forward = _corrupt_forward
        return self

    def __exit__(self, *_):
        self._ctrl.forward = self._orig_forward


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, label, num_batches=None):
    model.eval()
    turn_correct: dict[int, int] = {}
    turn_counts:  dict[int, int] = {}
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if num_batches is not None and b_idx >= num_batches:
                break

            facts_seq    = batch["facts"].to(device)
            fact_mask    = batch["fact_mask"].to(device)
            turn_mask    = batch["turn_mask"].to(device)
            labels       = (batch["labels"] if "labels" in batch else batch["label"]).to(device)

            all_logits, _ = model(facts_seq, fact_mask, turn_mask)
            B, max_turns, _ = all_logits.shape

            for t in range(max_turns):
                valid = (turn_mask[:, t] > 0).nonzero(as_tuple=True)[0]
                if len(valid) == 0:
                    continue
                preds = all_logits[valid, t].argmax(dim=-1)
                labs  = labels[valid, t]
                c = (preds == labs).sum().item()
                n = len(valid)
                turn_correct[t] = turn_correct.get(t, 0) + c
                turn_counts[t]  = turn_counts.get(t, 0) + n
                total_correct += c
                total_samples += n

    acc = total_correct / max(1, total_samples)
    per_turn = {
        f"turn_{t+1}": turn_correct[t] / max(1, turn_counts[t])
        for t in sorted(turn_counts)
    }

    print(f"\n  [{label}]")
    print(f"    Overall : {acc*100:.2f}%")
    for k, v in per_turn.items():
        print(f"    {k:8s}: {v*100:.2f}%")

    return {"label": label, "overall_acc": acc, "per_turn_acc": per_turn}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--num-batches", type=int, default=None)
    p.add_argument("--seed",        type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = print_hardware_info()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        overrides = yaml.safe_load(f) or {}
    config = load_config(Path("configs/base.yaml"), overrides=overrides)
    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    generator = ArithmeticGenerator(config=config.data, seed=seed)
    splits = generator.generate_dataset()
    vocab  = Vocabulary()
    vocab.build_from_records(splits["train"] + splits["val"] + splits["test"])

    loader = DataLoader(
        SequenceReasoningDataset(splits["test"], vocab),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_batch,
    )
    print(f"Test set: {len(splits['test'])} sequences.")

    model = SequentialReasoningModel(
        config=config.model, vocab=vocab,
        activity_config=config.activity,
        resolution_config=config.resolution,
        persistence_config=config.persistence,
        external_config=config.external,
    ).to(device)

    ckpt = resolve_checkpoint_path(args.checkpoint, None)
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    load_checkpoint(ckpt, model=model, device=device)

    print("\n" + "=" * 60)
    print("  Zero-Write Ablation & Slot Corruption Test")
    print("=" * 60)

    normal     = evaluate(model, loader, device, "NORMAL (learned gate)",              args.num_batches)
    with _ZeroWriteHook(model):
        zero   = evaluate(model, loader, device, "ZERO_WRITE (gate forced closed)",    args.num_batches)
    with _CorruptWriteHook(model):
        corrupt = evaluate(model, loader, device, "CORRUPT (records zeroed post-write)", args.num_batches)

    d_zero    = zero["overall_acc"]   - normal["overall_acc"]
    d_corrupt = corrupt["overall_acc"] - normal["overall_acc"]

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  NORMAL      : {normal['overall_acc']*100:.2f}%")
    print(f"  ZERO_WRITE  : {zero['overall_acc']*100:.2f}%   delta = {d_zero*100:+.2f}pp")
    print(f"  CORRUPT     : {corrupt['overall_acc']*100:.2f}%   delta = {d_corrupt*100:+.2f}pp")

    thr = 0.005
    print()
    if abs(d_zero) < thr:
        verdict_zero = "NO causal effect of writes (delta < 0.5pp)"
    else:
        verdict_zero = f"Writes are causally necessary (delta = {d_zero*100:+.2f}pp)"

    if abs(d_corrupt) < thr:
        verdict_corrupt = "Record CONTENT not causally active (mask alone drives reads)"
    elif abs(d_corrupt - d_zero) < thr:
        verdict_corrupt = "Record CONTENT is the causally active component (matches zero-write delta)"
    else:
        verdict_corrupt = "Partial: mask effect and content effect both contribute"

    print(f"  ZERO_WRITE verdict : {verdict_zero}")
    print(f"  CORRUPT    verdict : {verdict_corrupt}")
    print("=" * 60)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": args.config, "checkpoint": args.checkpoint,
        "conditions": [normal, zero, corrupt],
        "delta_zero_write_pp": round(d_zero * 100, 3),
        "delta_corrupt_pp":    round(d_corrupt * 100, 3),
        "verdict_zero_write": verdict_zero,
        "verdict_corrupt":    verdict_corrupt,
    }
    out = out_dir / "zero_write_ablation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()