#!/usr/bin/env python3
"""Plot phase-separated recursive-revision results from metrics_long.csv."""

from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OFFSETS = (0, 4, 8, 12)
PHASES = ("pre_contact", "post_contact")
PHASE_TITLES = ("Pre-contact", "Post-contact")
ONE_SHOT_COLOR = "#666666"
RETOUCH_COLOR = "#0072B2"
FRESH_COLOR = "#D55E00"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        default="outputs/task_test_recursive_revision_fair/metrics_long.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="AAAI27_tactile/Figures",
    )
    return parser.parse_args()


def _read_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _series(
    rows: list[dict[str, str]],
    *,
    metric: str,
    mode: str,
    phase: str,
) -> np.ndarray:
    values = []
    for offset in OFFSETS:
        matches = [
            row
            for row in rows
            if row["metric"] == metric
            and row["scope"] == "suffix"
            and row["mode"] == mode
            and row["phase"] == phase
            and int(row["offset"]) == offset
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one row for metric={metric}, mode={mode}, "
                f"phase={phase}, offset={offset}; found {len(matches)}."
            )
        values.append(float(matches[0]["value"]))
    return np.asarray(values, dtype=np.float64)


def _write_source_data(
    rows: list[dict[str, str]],
    output_dir: pathlib.Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["scope"] == "suffix"
        and row["phase"] in PHASES
        and (
            (row["metric"] == "latent_cosine" and row["mode"] in ("one_shot", "retouch"))
            or (row["metric"] == "action_mse" and row["mode"] in ("one_shot", "fresh_reinfer"))
        )
    ]
    selected.sort(
        key=lambda row: (
            row["metric"],
            PHASES.index(row["phase"]),
            row["mode"],
            int(row["offset"]),
        )
    )
    fieldnames = ["model", "metric", "scope", "mode", "offset", "phase", "value", "samples"]
    with (output_dir / "recursive_revision_phase_curves_source.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def _plot(
    rows: list[dict[str, str]],
    output_dir: pathlib.Path,
    *,
    metric: str,
    revision_mode: str,
    revision_label: str,
    revision_color: str,
    ylabel: str,
    stem: str,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.75),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    all_values = []
    for axis, phase, title in zip(axes, PHASES, PHASE_TITLES, strict=True):
        one_shot = _series(rows, metric=metric, mode="one_shot", phase=phase)
        revision = _series(rows, metric=metric, mode=revision_mode, phase=phase)
        all_values.extend(one_shot)
        all_values.extend(revision)

        axis.set_facecolor("#F7F7F7" if phase == "pre_contact" else "#FFF7F0")
        axis.plot(
            OFFSETS,
            one_shot,
            color=ONE_SHOT_COLOR,
            linestyle="--",
            marker="o",
            markersize=4.5,
            linewidth=1.7,
            label="One-shot",
            zorder=3,
        )
        axis.plot(
            OFFSETS,
            revision,
            color=revision_color,
            linestyle="-",
            marker="D",
            markersize=4.3,
            linewidth=2.0,
            label=revision_label,
            zorder=4,
        )
        axis.set_title(title, fontweight="bold", pad=6)
        axis.set_xticks(OFFSETS)
        axis.set_xlabel("Revision offset $k$")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    lower = min(all_values)
    upper = max(all_values)
    margin = max((upper - lower) * 0.10, 1e-5)
    axes[0].set_ylim(lower - margin, upper + margin)
    axes[0].set_ylabel(ylabel)

    handles = [
        Line2D(
            [0],
            [0],
            color=ONE_SHOT_COLOR,
            linestyle="--",
            marker="o",
            markersize=4.5,
            linewidth=1.7,
            label="One-shot",
        ),
        Line2D(
            [0],
            [0],
            color=revision_color,
            linestyle="-",
            marker="D",
            markersize=4.3,
            linewidth=2.0,
            label=revision_label,
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.19, top=0.80)
    divider_x = (axes[0].get_position().x1 + axes[1].get_position().x0) * 0.5
    divider = Line2D(
        [divider_x, divider_x],
        [0.17, 0.86],
        transform=fig.transFigure,
        color="#222222",
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.75,
    )
    fig.add_artist(divider)
    fig.text(
        divider_x,
        0.875,
        "Contact onset",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
    )

    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    metrics_path = pathlib.Path(args.metrics).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(metrics_path)
    _write_source_data(rows, output_dir)

    _plot(
        rows,
        output_dir,
        metric="latent_cosine",
        revision_mode="retouch",
        revision_label="ReTouch",
        revision_color=RETOUCH_COLOR,
        ylabel="Future tactile latent cosine $\\uparrow$",
        stem="recursive_revision_latent_curve",
    )
    _plot(
        rows,
        output_dir,
        metric="action_mse",
        revision_mode="fresh_reinfer",
        revision_label="Fresh",
        revision_color=FRESH_COLOR,
        ylabel="Physical suffix action MSE $\\downarrow$",
        stem="recursive_revision_action_curve",
    )
    print(f"Saved recursive-revision figures to {output_dir}")


if __name__ == "__main__":
    main()
