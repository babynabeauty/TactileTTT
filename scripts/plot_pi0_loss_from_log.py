#!/usr/bin/env python3
"""Plot the training loss curve from a vanilla pi0 training log.

env/.venv/bin/python scripts/plot_pi0_loss_from_log.py \
  --log logs/B_calc_dual_ae_history_future_pool_0627_grasp_cob_pool_20k_0628.log \
  --output outputs/loss_curves/B_calc_dual_ae_history_future_pool_0627_grasp_cob_pool_20k_0628.png \
  --smooth 3

"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(r"Step\s+(\d+):\s+(.*)")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+)=([-+0-9.eE]+)")


def parse_log(log_path: Path, loss_key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            match = STEP_RE.search(line)
            if match is None:
                continue
            step = int(match.group(1))
            values = {k: float(v) for k, v in KV_RE.findall(match.group(2))}
            if loss_key not in values:
                continue
            steps.append(step)
            losses.append(values[loss_key])
    if not steps:
        raise ValueError(f"No '{loss_key}' values found in {log_path}")
    return steps, losses


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    running = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Training log path.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV path.")
    parser.add_argument("--loss-key", default="loss", help="Metric key to plot. Default: loss.")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window over logged points.")
    parser.add_argument("--title", default=None, help="Plot title.")
    args = parser.parse_args()

    log_path = args.log
    output = args.output or Path("outputs/loss_curves") / f"{log_path.stem}_pi0_loss.png"
    csv_path = args.csv or output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    steps, losses = parse_log(log_path, args.loss_key)
    plot_losses = moving_average(losses, args.smooth)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", args.loss_key])
        writer.writerows(zip(steps, losses, strict=True))

    plt.figure(figsize=(9, 5))
    plt.plot(steps, plot_losses, label=args.loss_key, linewidth=2)
    if args.smooth > 1:
        plt.plot(steps, losses, label=f"{args.loss_key} raw", alpha=0.25, linewidth=1)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title or f"{log_path.stem}: {args.loss_key}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Saved plot: {output}")
    print(f"Saved csv:  {csv_path}")


if __name__ == "__main__":
    main()
