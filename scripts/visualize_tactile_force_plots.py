"""Plot XHand fingertip calc-force traces for LeRobot episodes.

This script creates static PNG figures only. It does not decode or render
camera videos, so it is faster and avoids OpenCV text/font artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


SENSOR_COUNT = 5
AXES = ("fx", "fy", "fz")
COLORS = ("tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:red")


def _load_state_names(repo_id: Path) -> list[str]:
    info = json.loads((repo_id / "meta" / "info.json").read_text())
    return info["features"]["observation.state"]["names"]


def _calc_force_indices(names: list[str]) -> np.ndarray:
    indices = []
    for sensor_id in range(SENSOR_COUNT):
        sensor_indices = []
        for axis in AXES:
            field = f"hand_tactile_sensor_{sensor_id}.calc_force.{axis}"
            if field not in names:
                raise ValueError(f"Missing state field: {field}")
            sensor_indices.append(names.index(field))
        indices.append(sensor_indices)
    return np.asarray(indices, dtype=np.int64)


def _episode_indices(repo_id: Path) -> list[int]:
    data_dir = repo_id / "data" / "chunk-000"
    return sorted(int(path.stem.split("_")[1]) for path in data_dir.glob("episode_*.parquet"))


def _read_episode_forces(repo_id: Path, episode_index: int, force_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parquet_path = repo_id / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    table = pq.read_table(parquet_path, columns=["observation.state", "timestamp"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    forces = states[:, force_indices.reshape(-1)].reshape(states.shape[0], SENSOR_COUNT, 3)
    return forces, timestamps


def _plot_episode(
    *,
    episode_index: int,
    timestamps: np.ndarray,
    forces: np.ndarray,
    output_dir: Path,
    magnitude_limit: float | None,
    component_limit: float | None,
) -> dict[str, object]:
    magnitudes = np.linalg.norm(forces, axis=-1)
    time_axis = timestamps - timestamps[0] if len(timestamps) else np.arange(forces.shape[0])

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4, 1.4, 1.4]},
        constrained_layout=True,
    )

    axes[0].set_title(f"episode_{episode_index:06d} XHand calc_force traces")
    for sensor_id in range(SENSOR_COUNT):
        axes[0].plot(
            time_axis,
            magnitudes[:, sensor_id],
            color=COLORS[sensor_id],
            linewidth=1.5,
            label=f"finger {sensor_id}",
        )
    axes[0].set_ylabel("|F|")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncols=SENSOR_COUNT)
    if magnitude_limit is not None:
        axes[0].set_ylim(0, magnitude_limit)

    for axis_id, axis_name in enumerate(AXES):
        ax = axes[axis_id + 1]
        for sensor_id in range(SENSOR_COUNT):
            ax.plot(
                time_axis,
                forces[:, sensor_id, axis_id],
                color=COLORS[sensor_id],
                linewidth=1.1,
                label=f"finger {sensor_id}" if axis_id == 0 else None,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_ylabel(axis_name)
        ax.grid(True, alpha=0.25)
        if component_limit is not None:
            ax.set_ylim(-component_limit, component_limit)

    axes[-1].set_xlabel("time (s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"episode_{episode_index:06d}_tactile_calc_force.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        "episode_index": episode_index,
        "output": str(out_path),
        "frames": int(forces.shape[0]),
        "mag_mean": magnitudes.mean(axis=0).round(4).tolist(),
        "mag_max": magnitudes.max(axis=0).round(4).tolist(),
        "zero_ratio": (magnitudes < 1e-6).mean(axis=0).round(4).tolist(),
        "all_zero_fingers": [int(i) for i in np.where(magnitudes.max(axis=0) < 1e-6)[0]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=Path, default=Path("grasp_pipette_and_press_button"))
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Episodes to plot. Default: all episodes.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tactile_force_plots"))
    parser.add_argument("--magnitude-limit", type=float, default=None)
    parser.add_argument("--component-limit", type=float, default=None)
    args = parser.parse_args()

    names = _load_state_names(args.repo_id)
    force_indices = _calc_force_indices(names)
    episodes = args.episodes if args.episodes else _episode_indices(args.repo_id)

    summaries = []
    for episode_index in episodes:
        forces, timestamps = _read_episode_forces(args.repo_id, episode_index, force_indices)
        summary = _plot_episode(
            episode_index=episode_index,
            timestamps=timestamps,
            forces=forces,
            output_dir=args.output_dir,
            magnitude_limit=args.magnitude_limit,
            component_limit=args.component_limit,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
