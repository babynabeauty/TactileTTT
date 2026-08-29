#!/usr/bin/env python3
"""Plot thumb/index XHand calc-force curves for LeRobot datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


TACTILE_SENSOR_COUNT = 5
TACTILE_BLOCK_SIZE = 384
TACTILE_BLOCK_START = 52
TACTILE_CALC_FORCE_OFFSET = 0
FINGERS = {"thumb": 0, "index": 1}
AXES = ("fx", "fy", "fz")


def _load_state_names(dataset: Path) -> list[str] | None:
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text())
    state_feature = info.get("features", {}).get("observation.state") or info.get("features", {}).get("observation/state")
    if not isinstance(state_feature, dict):
        return None
    names = state_feature.get("names")
    return list(names) if names else None


def _calc_force_indices(state_names: list[str] | None) -> dict[str, list[int]]:
    if state_names:
        name_to_idx = {name: idx for idx, name in enumerate(state_names)}
        indices: dict[str, list[int]] = {}
        for finger_name, sensor_id in FINGERS.items():
            finger_indices = []
            for axis in AXES:
                candidates = (
                    f"hand_tactile_sensor_{sensor_id}.calc_force.{axis}",
                    f"hand_tactile_sensor_{sensor_id}.calc_force.{axis[-1]}",
                )
                for candidate in candidates:
                    if candidate in name_to_idx:
                        finger_indices.append(name_to_idx[candidate])
                        break
                else:
                    finger_indices = []
                    break
            if finger_indices:
                indices[finger_name] = finger_indices
        if set(indices) == set(FINGERS):
            return indices

    return {
        finger_name: [
            TACTILE_BLOCK_START + sensor_id * TACTILE_BLOCK_SIZE + TACTILE_CALC_FORCE_OFFSET + axis_id
            for axis_id in range(3)
        ]
        for finger_name, sensor_id in FINGERS.items()
    }


def _episode_files(dataset: Path) -> list[Path]:
    files = sorted(dataset.glob("data/**/episode_*.parquet"))
    if files:
        return files
    files = sorted(dataset.glob("**/episode_*.parquet"))
    if files:
        return files
    raise FileNotFoundError(f"No episode_*.parquet files found under {dataset}")


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _read_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    parquet = pq.read_table(path, columns=["observation.state", "frame_index", "timestamp"])
    states = np.asarray(parquet["observation.state"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(parquet["frame_index"].to_pylist(), dtype=np.int64)
    timestamps = np.asarray(parquet["timestamp"].to_pylist(), dtype=np.float32)
    return states, frame_indices, timestamps


def _time_axis(frame_indices: np.ndarray, timestamps: np.ndarray | None) -> tuple[np.ndarray, str]:
    if timestamps is not None and timestamps.size and np.all(np.isfinite(timestamps)):
        return timestamps - timestamps[0], "time (s)"
    return frame_indices.astype(np.float32), "frame index"


def _plot_episode(
    *,
    dataset_name: str,
    episode_path: Path,
    states: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray | None,
    indices: dict[str, list[int]],
    output_dir: Path,
    unit_scale: float,
    unit_label: str,
) -> dict[str, object]:
    required_dim = max(max(v) for v in indices.values()) + 1
    if states.ndim != 2 or states.shape[1] < required_dim:
        raise ValueError(f"{episode_path}: expected observation.state [T,D] with D >= {required_dim}, got {states.shape}")

    thumb = states[:, indices["thumb"]] * unit_scale
    index = states[:, indices["index"]] * unit_scale
    thumb_mag = np.linalg.norm(thumb, axis=-1)
    index_mag = np.linalg.norm(index, axis=-1)
    t, xlabel = _time_axis(frame_indices, timestamps)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.8, 1.1, 1.1, 1.1]},
    )
    episode = _episode_index(episode_path)
    fig.suptitle(f"{dataset_name} episode_{episode:06d} thumb/index calc_force")

    axes[0].plot(t, thumb_mag, label="thumb |F|", color="#4c78a8", linewidth=1.6)
    axes[0].plot(t, index_mag, label="index |F|", color="#f58518", linewidth=1.6)
    axes[0].set_ylabel(f"|F| ({unit_label})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    for axis_id, axis in enumerate(AXES):
        ax = axes[axis_id + 1]
        ax.plot(t, thumb[:, axis_id], label=f"thumb {axis}", color="#4c78a8", linewidth=1.1)
        ax.plot(t, index[:, axis_id], label=f"index {axis}", color="#f58518", linewidth=1.1)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.set_ylabel(f"{axis} ({unit_label})")
        ax.grid(True, alpha=0.25)
        if axis_id == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel(xlabel)

    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_output_dir / f"episode_{episode:06d}_thumb_index_calc_force.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "episode_index": episode,
        "episode_file": str(episode_path),
        "output_png": str(output_path),
        "frames": int(states.shape[0]),
        "thumb_mean": float(np.mean(thumb_mag)),
        "thumb_p95": float(np.percentile(thumb_mag, 95)),
        "thumb_max": float(np.max(thumb_mag)),
        "index_mean": float(np.mean(index_mag)),
        "index_p95": float(np.percentile(index_mag, 95)),
        "index_max": float(np.max(index_mag)),
        "both_contact_frames_gt_1N": int(np.sum((thumb_mag > 1.0) & (index_mag > 1.0))),
    }


def _write_summary(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/thumb_index_tactile_curves"))
    parser.add_argument("--unit-scale", type=float, default=0.1, help="Default converts XHand force LSB to N.")
    parser.add_argument("--unit-label", default="N")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for dataset in args.datasets:
        dataset = dataset.expanduser().resolve()
        dataset_name = dataset.name
        state_names = _load_state_names(dataset)
        indices = _calc_force_indices(state_names)
        files = _episode_files(dataset)
        print(f"{dataset_name}: {len(files)} episodes, indices={indices}", flush=True)
        for count, episode_path in enumerate(files, start=1):
            print(f"[{dataset_name} {count}/{len(files)}] {episode_path}", flush=True)
            states, frame_indices, timestamps = _read_episode(episode_path)
            summaries.append(
                _plot_episode(
                    dataset_name=dataset_name,
                    episode_path=episode_path,
                    states=states,
                    frame_indices=frame_indices,
                    timestamps=timestamps,
                    indices=indices,
                    output_dir=args.output_dir,
                    unit_scale=args.unit_scale,
                    unit_label=args.unit_label,
                )
            )

    summary_path = args.output_dir / "thumb_index_calc_force_summary.csv"
    _write_summary(summaries, summary_path)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
