#!/usr/bin/env python3
"""Plot XHand index-finger calc-force resultant curves for LeRobot datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


AXES = ("fx", "fy", "fz")
INDEX_SENSOR_ID = 1


def _dataset_dirs(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").exists():
        return [root]
    datasets = sorted(path.parent.parent for path in root.rglob("meta/info.json"))
    if not datasets:
        raise FileNotFoundError(f"No LeRobot datasets found under {root}")
    return datasets


def _episode_files(dataset: Path) -> list[Path]:
    files = sorted(dataset.glob("data/**/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode_*.parquet files found under {dataset}")
    return files


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _load_state_names(dataset: Path) -> list[str]:
    info = json.loads((dataset / "meta" / "info.json").read_text())
    return info["features"]["observation.state"]["names"]


def _calc_force_indices(state_names: list[str], sensor_id: int = INDEX_SENSOR_ID) -> list[int]:
    indices = []
    for axis in AXES:
        name = f"hand_tactile_sensor_{sensor_id}.calc_force.{axis}"
        if name not in state_names:
            raise ValueError(f"Missing state field: {name}")
        indices.append(state_names.index(name))
    return indices


def _read_episode(path: Path, indices: list[int], unit_scale: float) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=["observation.state", "frame_index", "timestamp"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    force = states[:, indices] * unit_scale
    magnitude = np.linalg.norm(force, axis=-1)
    time = timestamps - timestamps[0] if timestamps.size else frame_indices.astype(np.float32)
    progress = np.linspace(0.0, 100.0, states.shape[0], dtype=np.float32)
    return {
        "force": force,
        "magnitude": magnitude,
        "time": time,
        "progress": progress,
        "frame_indices": frame_indices,
    }


def _plot_episode(
    *,
    dataset_name: str,
    episode: int,
    payload: dict[str, np.ndarray],
    output_dir: Path,
    unit_label: str,
) -> dict[str, object]:
    force = payload["force"]
    magnitude = payload["magnitude"]
    time = payload["time"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    fig.suptitle(f"{dataset_name} episode_{episode:06d} index calc_force resultant")

    axes[0].plot(time, magnitude, color="#f58518", linewidth=1.8)
    axes[0].set_ylabel(f"|F| ({unit_label})")
    axes[0].grid(True, alpha=0.25)

    colors = ("#4c78a8", "#54a24b", "#e45756")
    for axis_id, axis in enumerate(AXES):
        axes[1].plot(time, force[:, axis_id], color=colors[axis_id], linewidth=1.1, label=axis)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].set_ylabel(f"component ({unit_label})")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", ncols=3)
    axes[1].grid(True, alpha=0.25)

    output_path = output_dir / dataset_name / f"episode_{episode:06d}_index_calc_force_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "episode_index": episode,
        "frames": int(magnitude.size),
        "duration_sec": float(time[-1] - time[0]) if time.size else 0.0,
        "mean_force": float(np.mean(magnitude)),
        "p95_force": float(np.percentile(magnitude, 95)),
        "max_force": float(np.max(magnitude)),
        "max_frame_index": int(payload["frame_indices"][int(np.argmax(magnitude))]),
        "output_png": str(output_path),
    }


def _plot_overlay(
    *,
    curves: list[tuple[str, int, dict[str, np.ndarray]]],
    output_path: Path,
    unit_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for idx, (dataset_name, episode, payload) in enumerate(curves):
        ax.plot(
            payload["progress"],
            payload["magnitude"],
            linewidth=1.5,
            color=cmap(idx % 10),
            label=f"{dataset_name} ep{episode:02d}",
        )
    ax.set_title("Index calc_force resultant curves, episode-aligned")
    ax.set_xlabel("episode progress (%)")
    ax.set_ylabel(f"|F| ({unit_label})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_timeline(
    *,
    curves: list[tuple[str, int, dict[str, np.ndarray]]],
    output_path: Path,
    unit_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    offset = 0.0
    gap = 1.0
    cmap = plt.get_cmap("tab10")
    boundaries = []
    for idx, (dataset_name, episode, payload) in enumerate(curves):
        time = payload["time"]
        shifted = time - time[0] + offset
        ax.plot(shifted, payload["magnitude"], linewidth=1.4, color=cmap(idx % 10), label=f"{dataset_name} ep{episode:02d}")
        boundaries.append((offset, f"ep{episode}"))
        offset = float(shifted[-1]) + gap if shifted.size else offset + gap
    for x, label in boundaries:
        ax.axvline(x, color="black", alpha=0.12, linewidth=0.8)
        ax.text(x, ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8)
    ax.set_title("Index calc_force resultant timeline")
    ax.set_xlabel("concatenated time (s)")
    ax.set_ylabel(f"|F| ({unit_label})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncols=2, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/press_button_4_times"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/index_calc_force_curves"))
    parser.add_argument("--unit-scale", type=float, default=0.1, help="Default converts XHand force LSB to N.")
    parser.add_argument("--unit-label", default="N")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows: list[dict[str, object]] = []
    curves: list[tuple[str, int, dict[str, np.ndarray]]] = []

    for dataset in _dataset_dirs(dataset_root):
        dataset_name = dataset.name
        indices = _calc_force_indices(_load_state_names(dataset))
        for episode_path in _episode_files(dataset):
            episode = _episode_index(episode_path)
            payload = _read_episode(episode_path, indices, args.unit_scale)
            curves.append((dataset_name, episode, payload))
            rows.append(
                _plot_episode(
                    dataset_name=dataset_name,
                    episode=episode,
                    payload=payload,
                    output_dir=output_dir,
                    unit_label=args.unit_label,
                )
            )

    _plot_overlay(curves=curves, output_path=output_dir / "index_calc_force_overlay.png", unit_label=args.unit_label)
    _plot_timeline(curves=curves, output_path=output_dir / "index_calc_force_timeline.png", unit_label=args.unit_label)
    _write_summary(rows, output_dir / "index_calc_force_summary.csv")

    print(json.dumps({"output_dir": str(output_dir), "episodes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
