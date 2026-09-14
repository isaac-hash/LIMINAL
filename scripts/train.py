import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.device import print_hardware_info
from src.utils.config import load_config, set_seed
from src.data.arithmetic import ArithmeticGenerator
from src.data.dataset import Vocabulary, ReasoningDataset, collate_reasoning_batch
from src.models.model import ReasoningModel
from src.training.losses import LossComputer
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="LIMINAL Training Runner")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--drive-path", type=str, default=None, help="Google Drive backup directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    device = print_hardware_info()

    # Load base config, overlay specific config if different from base
    base_config_path = Path("configs/base.yaml")
    config_path = Path(args.config)

    if config_path != base_config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        config = load_config(base_config_path, overrides=overrides)
    else:
        config = load_config(base_config_path)

    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    print(f"\n--- Training Configuration ---")
    print(f"Model Type:      {config.model.type}")
    print(f"Latent Slots:    {config.model.latent_slots}")
    print(f"Latent Dim:      {config.model.latent_dim}")
    print(f"Reasoning Steps: {config.model.reasoning_steps}")
    print(f"Activity Gates:  {'ON (adaptive)' if config.activity.enabled else 'OFF (static)'}")
    print(f"Task Family:     {config.data.task_family}")
    print(f"Epochs:          {config.training.epochs}")
    print(f"Batch Size:      {config.training.batch_size}")
    print(f"Checkpoint Dir:  {config.training.checkpoint_dir}\n")

    # 1. Generate synthetic dataset splits
    print("Generating synthetic dataset...")
    generator = ArithmeticGenerator(config=config.data, seed=seed)
    dataset_splits = generator.generate_dataset()

    train_records = dataset_splits["train"]
    val_records = dataset_splits["val"]
    test_records = dataset_splits["test"]

    print(f"Dataset generated: {len(train_records)} train, {len(val_records)} val, {len(test_records)} test.")

    # 2. Build vocabulary
    vocab = Vocabulary()
    vocab.build_from_records(train_records + val_records + test_records)

    # 3. Create DataLoaders
    train_dataset = ReasoningDataset(train_records, vocab)
    val_dataset = ReasoningDataset(val_records, vocab)
    test_dataset = ReasoningDataset(test_records, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_reasoning_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_reasoning_batch,
    )

    # 4. Build Model & Training Infrastructure
    model = ReasoningModel(
        config=config.model,
        vocab=vocab,
        activity_config=config.activity,
    ).to(device)
    loss_computer = LossComputer(config)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_computer,
        config=config,
        device=device,
        drive_path=args.drive_path,
    )

    # 5. Optionally Resume
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        resumed_epoch = trainer.resume_from(args.resume)
        print(f"Resumed from epoch {resumed_epoch}")

    # 6. Train
    print("Starting training loop...")
    metrics = trainer.train()
    print("Training finished.")

    # 7. Final Test Evaluation
    test_metrics = trainer.evaluate(test_loader)
    print(f"\n--- Final Test Results ---")
    print(f"Test Loss:     {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['acc'] * 100:.2f}%\n")


if __name__ == "__main__":
    main()
