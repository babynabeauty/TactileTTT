#!/usr/bin/env python3
"""Compare thumb/index XHand calc-force curves grouped by object weight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


AXES = ("fx", "fy", "fz")
FINGER_IDS = {"thumb": 0, "index": 1}


def _load_state_names(dataset: Path) -> list[str]:
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    return info["features"]["observation.state"]["names"]


def _calc_force_indices(state_names: list[str], sensor_id: int) -> list[int]:
    return [state_names.index(f"hand_tactile_sensor_{sensor_id}.calc_force.{axis}") for axis in AXES]


def _read_calc_force(dataset: Path, episode_number_1based: int, indices: dict[str, list[int]], unit_scale: float) -> dict[str, np.ndarray]:
    episode_index = episode_number_1based - 1
    matches = sorted(dataset.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(f"Cannot find episode_{episode_index:06d}.parquet under {dataset}")

    table = pq.read_table(matches[0], columns=["observation.state"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    thumb = states[:, indices["thumb"]] * unit_scale
    index = states[:, indices["index"]] * unit_scale
    return {
        "thumb": np.linalg.norm(thumb, axis=-1),
        "index": np.linalg.norm(index, axis=-1),
        "frames": np.arange(states.shape[0], dtype=np.float32),
        "progress": np.linspace(0.0, 100.0, states.shape[0], dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-0823", type=Path, default=Path("data/test_0823"))
    parser.add_argument("--test-0823-2", type=Path, default=Path("data/test_0823_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/thumb_index_tactile_curves"))
    parser.add_argument("--unit-scale", type=float, default=0.1, help="Default converts XHand force LSB to N.")
    parser.add_argument("--unit-label", default="N")
    args = parser.parse_args()

    datasets = {
        "0823": args.test_0823.expanduser().resolve(),
        "0823_2": args.test_0823_2.expanduser().resolve(),
    }
    groups = {
        "half bottle": [("0823", 1), ("0823", 2), ("0823_2", 3)],
        "full bottle": [("0823", 3), ("0823_2", 1), ("0823_2", 2)],
        "empty bottle": [("0823_2", 4), ("0823_2", 5)],
    }
    group_titles = {
        "half bottle": "Half Bottle",
        "full bottle": "Full Bottle",
        "empty bottle": "Empty Bottle",
    }

    indices_by_dataset: dict[str, dict[str, list[int]]] = {}
    for name, dataset in datasets.items():
        state_names = _load_state_names(dataset)
        indices_by_dataset[name] = {
            finger: _calc_force_indices(state_names, sensor_id)
            for finger, sensor_id in FINGER_IDS.items()
        }

    curves: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    max_force = 0.0
    for items in groups.values():
        for dataset_name, episode_number in items:
            key = (dataset_name, episode_number)
            curve = _read_calc_force(
                datasets[dataset_name],
                episode_number,
                indices_by_dataset[dataset_name],
                args.unit_scale,
            )
            curves[key] = curve
            max_force = max(max_force, float(np.max(curve["thumb"])), float(np.max(curve["index"])))

    colors = {
        ("0823", 1): "#4c78a8",
        ("0823", 2): "#72b7b2",
        ("0823", 3): "#54a24b",
        ("0823_2", 1): "#f58518",
        ("0823_2", 2): "#e45756",
        ("0823_2", 3): "#b279a2",
        ("0823_2", 4): "#9d755d",
        ("0823_2", 5): "#bab0ac",
    }

    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Thumb/Index calc_force comparison by bottle weight")

    for row, (group_name, items) in enumerate(groups.items()):
        for col, finger in enumerate(("thumb", "index")):
            ax = axes[row, col]
            for dataset_name, episode_number in items:
                curve = curves[(dataset_name, episode_number)]
                label = f"{dataset_name} data {episode_number}"
                ax.plot(
                    curve["progress"],
                    curve[finger],
                    linewidth=1.7,
                    color=colors[(dataset_name, episode_number)],
                    label=label,
                )
            ax.set_title(f"{group_titles[group_name]} - {finger}")
            ax.set_ylabel(f"|F| ({args.unit_label})")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", fontsize=8)
            ax.set_ylim(-0.05 * max_force, max_force * 1.08 if max_force > 0 else 1.0)
            if row == 2:
                ax.set_xlabel("episode progress (%)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "thumb_index_weight_group_comparison.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
