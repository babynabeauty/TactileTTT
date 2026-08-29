#!/usr/bin/env python3
"""Plot total and component losses from a dual-AE tactile training log.

env/.venv/bin/python scripts/plot_dual_ae_losses_from_log.py \
  --log logs/B_calc_dual_ae_history_future_pool_0627_grasp_cob_pool_20k_0628.log \
  --output logs/loss_curves/B_calc_dual_ae_history_future_pool_0627_grasp_cob_pool_20k_0628.png \
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

DEFAULT_KEYS = (
    "loss/total",
    "loss/student_action",
    "loss/teacher_action",
    "loss/distill_future_force",
    "loss/student_action_arm",
    "loss/student_action_hand",
    "loss/teacher_action_arm",
    "loss/teacher_action_hand",
)


def parse_log(log_path: Path) -> tuple[list[int], dict[str, list[float]]]:
    steps: list[int] = []
    metrics: dict[str, list[float]] = {}

    with log_path.open("r", errors="ignore") as f:
        for line in f:
            match = STEP_RE.search(line)
            if match is None:
                continue
            step = int(match.group(1))
            values = {k: float(v) for k, v in KV_RE.findall(match.group(2))}
            if not any(k.startswith("loss") for k in values):
                continue
            steps.append(step)
            for key in set(metrics) | set(values):
                if key.startswith("loss"):
                    metrics.setdefault(key, []).append(values.get(key, float("nan")))

    if not steps:
        raise ValueError(f"No loss metrics found in {log_path}")

    # Backfill any metric that appeared after the first logged step.
    for key, vals in metrics.items():
        if len(vals) < len(steps):
            metrics[key] = [float("nan")] * (len(steps) - len(vals)) + vals
    return steps, metrics


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    smoothed: list[float] = []
    queue: list[float] = []
    for value in values:
        queue.append(value)
        if len(queue) > window:
            queue.pop(0)
        valid = [v for v in queue if v == v]
        smoothed.append(sum(valid) / len(valid) if valid else float("nan"))
    return smoothed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="Training log path.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional output CSV path.")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=list(DEFAULT_KEYS),
        help="Loss keys to plot. Use --list-keys first to inspect available keys.",
    )
    parser.add_argument("--list-keys", action="store_true", help="Print available loss keys and exit.")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window over logged points.")
    parser.add_argument("--title", default=None, help="Plot title.")
    args = parser.parse_args()

    log_path = args.log
    steps, metrics = parse_log(log_path)
    available_keys = sorted(metrics)

    if args.list_keys:
        print("\n".join(available_keys))
        return

    selected_keys = [key for key in args.keys if key in metrics]
    if not selected_keys:
        raise ValueError(
            f"None of the requested keys were found in {log_path}. "
            f"Available keys: {', '.join(available_keys)}"
        )

    output = args.output or Path("outputs/loss_curves") / f"{log_path.stem}_dual_ae_losses.png"
    csv_path = args.csv or output.with_suffix(".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *available_keys])
        for i, step in enumerate(steps):
            writer.writerow([step, *[metrics[key][i] for key in available_keys]])

    plt.figure(figsize=(11, 6))
    for key in selected_keys:
        values = moving_average(metrics[key], args.smooth)
        plt.plot(steps, values, label=key, linewidth=2 if key in {"loss/total", "loss"} else 1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title or f"{log_path.stem}: dual-AE losses")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Saved plot: {output}")
    print(f"Saved csv:  {csv_path}")
    print("Plotted keys:")
    for key in selected_keys:
        print(f"  - {key}")


if __name__ == "__main__":
    main()
