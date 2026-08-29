#!/usr/bin/env python3
"""Combine per-checkpoint recursive-revision metrics and draw comparison plots."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _plot(frame: pd.DataFrame, output: pathlib.Path, metric: str, scope: str, phase: str) -> None:
    selected = frame[
        (frame["metric"] == metric)
        & (frame["scope"] == scope)
        & (frame["phase"] == phase)
        & (frame["samples"] > 0)
    ]
    if selected.empty:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=200)
    for model, group in selected.groupby("model", sort=False):
        group = group.sort_values("offset")
        ax.plot(group["offset"], group["value"], marker="o", linewidth=1.8, label=model)
    ax.set_xlabel("execution offset k")
    direction = "higher is better" if metric == "latent_cosine" else "lower is better"
    ax.set_ylabel(f"{metric.replace('_', ' ')} ({scope}, {phase}, {direction})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / f"{metric}_{scope}_{phase}_checkpoint_comparison.png")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    input_root = pathlib.Path(args.input_root).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() if args.output_dir else input_root
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = sorted(input_root.glob("*/metrics_long.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No */metrics_long.csv files found under {input_root}")
    frames = []
    for metric_file in metric_files:
        frame = pd.read_csv(metric_file)
        frame["result_dir"] = metric_file.parent.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "suite_metrics_long.csv", index=False)

    for scope in ("full", "suffix"):
        for phase in ("overall", "pre_contact", "post_contact", "no_contact"):
            _plot(combined, output_dir, "latent_cosine", scope, phase)
            for metric in (
                "action_mse",
                "action_mse_arm",
                "action_mse_hand",
                "action_mse_normalized",
                "action_mse_normalized_arm",
                "action_mse_normalized_hand",
            ):
                _plot(combined, output_dir, metric, scope, phase)

    summary_columns = [
        "model",
        "mode",
        "metric",
        "phase",
        "offset",
        "value",
        "samples",
        "result_dir",
    ]
    for scope in ("full", "suffix"):
        summary = combined[
            (combined["scope"] == scope)
            & (combined["phase"].isin(("overall", "pre_contact", "post_contact")))
        ].copy()
        summary = summary[summary_columns].sort_values(["metric", "phase", "offset", "model"])
        summary.to_csv(output_dir / f"suite_summary_{scope}.csv", index=False)
        if scope == "suffix":
            summary.to_csv(output_dir / "suite_summary.csv", index=False)
    print(f"Combined {len(metric_files)} runs into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
