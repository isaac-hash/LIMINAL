import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("--runs", nargs="+", required=True, help="List of run directories to compare")
    parser.add_argument("--output", type=str, default="results/comparison.png", help="Output comparison plot path")
    return parser.parse_args()


def main():
    args = parse_args()
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for run_dir in args.runs:
        run_path = Path(run_dir)
        csv_path = run_path / "metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            plt.plot(df["epoch"], df["train_loss"], label=f"{run_path.name} (train)")
            if "val_loss" in df.columns:
                plt.plot(df["epoch"], df["val_loss"], linestyle="--", label=f"{run_path.name} (val)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for run_dir in args.runs:
        run_path = Path(run_dir)
        csv_path = run_path / "metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "val_acc" in df.columns:
                plt.plot(df["epoch"], df["val_acc"] * 100, label=f"{run_path.name}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    main()
