#!/usr/bin/env python3
"""Create XHand index-finger raw tactile taxel videos from LeRobot episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pyarrow.parquet as pq


AXES = ("fx", "fy", "fz")
INDEX_SENSOR_ID = 1
RAW_TAXELS = 120
DEFAULT_LAYOUT_ROOT = Path("/Users/babyna/💡科研/DexTactileIDM/Xhand1交付资料-带触觉")


def _dataset_dirs(root: Path) -> list[Path]:
    if (root / "meta" / "info.json").exists():
        return [root]
    dirs = sorted(path.parent.parent for path in root.rglob("meta/info.json"))
    if not dirs:
        raise FileNotFoundError(f"No LeRobot meta/info.json found under {root}")
    return dirs


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


def _raw_force_indices(state_names: list[str], sensor_id: int = INDEX_SENSOR_ID) -> np.ndarray:
    indices = []
    for taxel in range(RAW_TAXELS):
        for axis in AXES:
            name = f"hand_tactile_sensor_{sensor_id}.raw_force_{taxel}.{axis}"
            if name not in state_names:
                raise ValueError(f"Missing state field: {name}")
            indices.append(state_names.index(name))
    return np.asarray(indices, dtype=np.int64)


def _read_t16_coords(layout_root: Path) -> tuple[np.ndarray, Path]:
    layout_dir = layout_root / "触觉传感器"
    candidates = (
        layout_dir / "points_t16_transformed (1).json",
        layout_dir / "points_t16_transformed (2).json",
        layout_dir / "points.t16 (2).json",
    )
    for path in candidates:
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            points = data.get("measurement_points") or data.get("points")
            if points is None:
                raise ValueError(f"No measurement_points/points in {path}")
            points = sorted(points, key=lambda item: int(item["point"]))
            coords_3d = np.asarray([[float(p["x"]), float(p["y"]), float(p["z"])] for p in points], dtype=np.float32)
            if coords_3d.shape != (RAW_TAXELS, 3):
                raise ValueError(f"Expected {(RAW_TAXELS, 3)} coords in {path}, got {coords_3d.shape}")
            return coords_3d[:, [1, 2]], path
    raise FileNotFoundError(f"Could not find T16 tactile coordinate JSON under {layout_dir}")


def _read_episode_magnitudes(episode_path: Path, raw_indices: np.ndarray, unit_scale: float) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(episode_path, columns=["observation.state", "frame_index"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    raw = states[:, raw_indices].reshape(states.shape[0], RAW_TAXELS, 3) * unit_scale
    magnitudes = np.linalg.norm(raw, axis=-1).astype(np.float32)
    return magnitudes, frame_indices


def _compute_vmax(episode_payloads: Iterable[tuple[np.ndarray, np.ndarray, Path]], percentile: float, minimum: float) -> float:
    values = [mags.reshape(-1) for mags, _frame_indices, _path in episode_payloads]
    if not values:
        return minimum
    merged = np.concatenate(values)
    positive = merged[merged > 0]
    if positive.size == 0:
        return minimum
    return max(float(np.percentile(positive, percentile)), minimum)


def _canvas_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def _init_figure(coords: np.ndarray, cmap: str, vmax: float, unit_label: str) -> tuple[plt.Figure, plt.Axes, object, plt.Text]:
    fig, ax = plt.subplots(figsize=(6.4, 7.6), dpi=140)
    x = coords[:, 0]
    y = coords[:, 1]
    pad_x = max(1.0, 0.08 * float(np.ptp(x)))
    pad_y = max(1.0, 0.08 * float(np.ptp(y)))
    ax.scatter(x, y, s=30, c="#d0d0d0", edgecolors="#9a9a9a", linewidths=0.35, zorder=1)
    scatter = ax.scatter(
        x,
        y,
        s=72,
        c=np.zeros(RAW_TAXELS),
        cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=vmax),
        edgecolors="black",
        linewidths=0.2,
        zorder=2,
    )
    title = ax.set_title("", fontsize=10, pad=10)
    ax.set_aspect("equal")
    ax.set_xlim(float(x.min() - pad_x), float(x.max() + pad_x))
    ax.set_ylim(float(y.min() - pad_y), float(y.max() + pad_y))
    ax.set_xlabel("lateral coordinate (mm)")
    ax.set_ylabel("distal coordinate (mm)")
    ax.grid(True, alpha=0.18)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"taxel |F| ({unit_label})")
    fig.subplots_adjust(top=0.90, right=0.86)
    return fig, ax, scatter, title


def _write_video(
    *,
    output_path: Path,
    coords: np.ndarray,
    frame_sets: list[tuple[str, np.ndarray, np.ndarray]],
    fps: int,
    cmap: str,
    vmax: float,
    unit_label: str,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ax, scatter, title = _init_figure(coords, cmap, vmax, unit_label)
    total_frames = 0
    max_force = 0.0
    mean_force_sum = 0.0
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for label, magnitudes, frame_indices in frame_sets:
            for row, frame_index in enumerate(frame_indices):
                mag = magnitudes[row]
                scatter.set_array(mag)
                scatter.set_sizes(28.0 + 92.0 * np.clip(mag / max(vmax, 1e-6), 0.0, 1.0))
                title.set_text(
                    f"Index raw tactile taxels | {label}\n"
                    f"frame {int(frame_index)} | max {float(np.max(mag)):.2f} {unit_label}"
                )
                writer.append_data(_canvas_rgb(fig))
                total_frames += 1
                max_force = max(max_force, float(np.max(mag)))
                mean_force_sum += float(np.mean(mag))
    plt.close(fig)
    return {
        "output_mp4": str(output_path),
        "frames": total_frames,
        "fps": fps,
        "duration_sec": total_frames / float(fps),
        "max_taxel_force": max_force,
        "mean_taxel_force": mean_force_sum / max(total_frames, 1),
        "vmax": vmax,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/press_button_4_times"))
    parser.add_argument("--layout-root", type=Path, default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/index_taxel_tactile_videos"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--unit-scale", type=float, default=0.1, help="Default converts XHand force LSB to N.")
    parser.add_argument("--unit-label", default="N")
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--vmax-percentile", type=float, default=99.5)
    parser.add_argument("--cmap", default="turbo")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    layout_root = args.layout_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    coords, layout_path = _read_t16_coords(layout_root)
    all_summaries: list[dict[str, object]] = []
    concat_frame_sets: list[tuple[str, np.ndarray, np.ndarray]] = []

    for dataset in _dataset_dirs(dataset_root):
        state_names = _load_state_names(dataset)
        raw_indices = _raw_force_indices(state_names)
        episode_payloads = []
        for episode_path in _episode_files(dataset):
            magnitudes, frame_indices = _read_episode_magnitudes(episode_path, raw_indices, args.unit_scale)
            episode_payloads.append((magnitudes, frame_indices, episode_path))

        vmax = args.vmax if args.vmax is not None else _compute_vmax(episode_payloads, args.vmax_percentile, minimum=1.0)
        dataset_out = output_dir / dataset.name
        for magnitudes, frame_indices, episode_path in episode_payloads:
            episode = _episode_index(episode_path)
            label = f"{dataset.name} episode_{episode:06d}"
            summary = _write_video(
                output_path=dataset_out / f"episode_{episode:06d}_index_taxel_tactile.mp4",
                coords=coords,
                frame_sets=[(label, magnitudes, frame_indices)],
                fps=args.fps,
                cmap=args.cmap,
                vmax=vmax,
                unit_label=args.unit_label,
            )
            summary.update(
                {
                    "dataset": str(dataset),
                    "episode_index": episode,
                    "layout_json": str(layout_path),
                }
            )
            all_summaries.append(summary)
            concat_frame_sets.append((label, magnitudes, frame_indices))

    if concat_frame_sets:
        concat_vmax = args.vmax if args.vmax is not None else _compute_vmax(
            [(mags, frame_indices, Path(label)) for label, mags, frame_indices in concat_frame_sets],
            args.vmax_percentile,
            minimum=1.0,
        )
        all_summary = _write_video(
            output_path=output_dir / "press_button_4_times_index_taxel_tactile_all_episodes.mp4",
            coords=coords,
            frame_sets=concat_frame_sets,
            fps=args.fps,
            cmap=args.cmap,
            vmax=concat_vmax,
            unit_label=args.unit_label,
        )
        all_summary.update({"dataset": str(dataset_root), "episode_index": "all", "layout_json": str(layout_path)})
        all_summaries.append(all_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "index_taxel_tactile_video_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False))
    print(json.dumps({"summary": str(summary_path), "num_videos": len(all_summaries)}, indent=2))


if __name__ == "__main__":
    main()
