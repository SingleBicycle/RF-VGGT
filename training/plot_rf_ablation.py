import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RF ablation loss curves from a JSON summary.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to ablation summary JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--title", type=str, default="RF Encoder Ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_json.open("r") as f:
        summary = json.load(f)

    methods = summary["methods"]

    plt.figure(figsize=(8, 5))
    for item in methods:
        steps = list(range(len(item["loss_curve"])))
        plt.plot(steps, item["loss_curve"], marker="o", linewidth=2, markersize=3, label=item["method"])
    plt.xlabel("Step")
    plt.ylabel("Camera Loss")
    plt.title(args.title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    curve_path = output_dir / f"{input_json.stem}_curve.png"
    plt.savefig(curve_path, dpi=200)
    plt.close()

    method_names = [item["method"] for item in methods]
    initial_losses = [item["initial_loss"] for item in methods]
    final_losses = [item["final_loss"] for item in methods]
    best_losses = [item["best_loss"] for item in methods]

    x = range(len(method_names))
    width = 0.24
    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], initial_losses, width=width, label="initial")
    plt.bar(x, final_losses, width=width, label="final")
    plt.bar([i + width for i in x], best_losses, width=width, label="best")
    plt.xticks(list(x), method_names, rotation=15)
    plt.ylabel("Camera Loss")
    plt.title(f"{args.title} Summary")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    bar_path = output_dir / f"{input_json.stem}_summary.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()

    print(f"Saved curve plot to {curve_path}")
    print(f"Saved summary plot to {bar_path}")


if __name__ == "__main__":
    main()
