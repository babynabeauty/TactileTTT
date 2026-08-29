#!/usr/bin/env python3
"""Visualize one XHand finger's raw tactile contact as official taxel heat points.

Inactive taxels are shown in gray. Active taxels are colored by force magnitude
using a heatmap colormap. The 2D taxel layout is read from the official XHand
tactile point JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.spatial

from openpi.policies import xhand_policy


DEFAULT_LAYOUT_DIR = Path("Xhand1交付资料-带触觉/触觉传感器")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _episode_file(repo: Path, episode_index: int) -> Path:
    candidates = sorted(repo.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find episode_{episode_index:06d}.parquet under {repo}")
    return candidates[0]


def _read_measurement_points(path: Path) -> np.ndarray:
    with path.open() as f:
        data = json.load(f)
    points = data.get("measurement_points") or data.get("points")
    if points is None:
        raise ValueError(f"Could not find measurement_points/points in {path}")
    points = sorted(points, key=lambda item: int(item["point"]))
    coords = np.asarray([[float(p["x"]), float(p["y"]), float(p["z"])] for p in points], dtype=np.float32)
    if coords.shape != (120, 3):
        raise ValueError(f"Expected 120 points in {path}, got {coords.shape}")
    return coords


def _taxel_coords(layout_dir: Path, finger: int) -> tuple[np.ndarray, Path]:
    if finger == 0:
        path = layout_dir / "points_t30_right_hand_transformed.json"
        if not path.exists():
            path = layout_dir / "points.t30 (2).json"
        coords_3d = _read_measurement_points(path)
        coords_2d = coords_3d[:, [0, 1]]  # thumb T30: lateral x, distal y
    else:
        path = layout_dir / "points_t16_transformed (1).json"
        if not path.exists():
            path = layout_dir / "points_t16_transformed (2).json"
        if not path.exists():
            path = layout_dir / "points.t16 (2).json"
        coords_3d = _read_measurement_points(path)
        coords_2d = coords_3d[:, [1, 2]]  # four fingers T16: lateral y, distal z
    return coords_2d.astype(np.float32), path


def _extract_raw_tactile(state: np.ndarray) -> np.ndarray:
    chunks = []
    for sensor_id in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + sensor_id * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        chunks.append(state[start:end].reshape(xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    return np.stack(chunks, axis=0).astype(np.float32)


def _select_frame(df: pd.DataFrame, finger: int, requested: int | None) -> tuple[int, np.ndarray]:
    if requested is not None:
        if requested < 0:
            requested = int(df["frame_index"].max()) + 1 + requested
        row = df[df["frame_index"] == requested]
        if row.empty:
            raise ValueError(f"frame_index={requested} not found in episode.")
        return requested, np.asarray(row.iloc[0]["observation.state"], dtype=np.float32)

    best_frame = None
    best_state = None
    best_score = -np.inf
    for _, row in df.iterrows():
        state = np.asarray(row["observation.state"], dtype=np.float32)
        tactile = _extract_raw_tactile(state)
        score = float(np.linalg.norm(tactile[finger], axis=-1).sum())
        if score > best_score:
            best_score = score
            best_frame = int(row["frame_index"])
            best_state = state
    if best_frame is None or best_state is None:
        raise ValueError("Episode has no frames.")
    return best_frame, best_state


def _draw_contact_heatmap(
    *,
    coords: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str,
    cmap_name: str,
    draw_glow: bool,
    show_title: bool,
) -> None:
    x = coords[:, 0]
    y = coords[:, 1]
    active = magnitude > threshold
    active_values = magnitude[active]
    vmax = float(np.percentile(active_values, 98.0)) if active_values.size else max(float(magnitude.max()), 1.0)
    vmax = max(vmax, threshold + 1e-6)

    pad_x = max(0.8, 0.08 * float(np.ptp(x)))
    pad_y = max(0.8, 0.08 * float(np.ptp(y)))
    xlim = (float(x.min() - pad_x), float(x.max() + pad_x))
    ylim = (float(y.min() - pad_y), float(y.max() + pad_y))

    fig, ax = plt.subplots(figsize=(3.0, 5.2))

    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_pts = coords[hull.vertices]
        ax.fill(hull_pts[:, 0], hull_pts[:, 1], color="#06115a", zorder=0)
    except Exception:
        pass

    if draw_glow and active_values.size:
        grid_x, grid_y = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 260),
            np.linspace(ylim[0], ylim[1], 420),
        )
        sigma = 0.045 * max(float(np.ptp(x)), float(np.ptp(y)))
        sigma = max(sigma, 1e-6)
        heat = np.zeros_like(grid_x, dtype=np.float32)
        for point, value in zip(coords[active], active_values, strict=True):
            dist2 = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2
            heat += float(value) * np.exp(-dist2 / (2.0 * sigma * sigma))
        heat = scipy.ndimage.gaussian_filter(heat, sigma=1.0)

        hull = scipy.spatial.Delaunay(coords)
        inside = hull.find_simplex(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)) >= 0
        inside = inside.reshape(grid_x.shape)
        heat = np.ma.array(heat, mask=~inside)
        ax.imshow(
            heat,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cmap_name,
            vmin=0.0,
            vmax=max(float(np.percentile(heat.compressed(), 98.0)), 1e-6),
            interpolation="bilinear",
            alpha=0.62,
        )

    # Official outline.
    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_pts = coords[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="black", linewidth=2.2, zorder=5)
    except Exception:
        pass

    # Inactive taxels are always gray.
    ax.scatter(
        x[~active],
        y[~active],
        s=18,
        c="#d2d2d2",
        edgecolors="none",
        linewidths=0.0,
        zorder=6,
    )
    scatter = ax.scatter(
        x[active],
        y[active],
        s=48,
        c=magnitude[active],
        cmap=cmap_name,
        vmin=threshold,
        vmax=vmax,
        edgecolors="none",
        linewidths=0.0,
        zorder=7,
    )

    if active_values.size:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("force magnitude", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if show_title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _draw_contact_heatmap_on_ax(
    ax,
    *,
    coords: np.ndarray,
    magnitude: np.ndarray,
    threshold: float,
    cmap_name: str,
    draw_glow: bool,
    show_title: bool,
    title: str,
    point_size_scale: float = 1.0,
) -> None:
    x = coords[:, 0]
    y = coords[:, 1]
    active = magnitude > threshold
    active_values = magnitude[active]
    vmax = float(np.percentile(active_values, 98.0)) if active_values.size else max(float(magnitude.max()), 1.0)
    vmax = max(vmax, threshold + 1e-6)

    pad_x = max(0.8, 0.08 * float(np.ptp(x)))
    pad_y = max(0.8, 0.08 * float(np.ptp(y)))
    xlim = (float(x.min() - pad_x), float(x.max() + pad_x))
    ylim = (float(y.min() - pad_y), float(y.max() + pad_y))

    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_pts = coords[hull.vertices]
        ax.fill(hull_pts[:, 0], hull_pts[:, 1], color="#06115a", zorder=0)
    except Exception:
        pass

    if draw_glow and active_values.size:
        grid_x, grid_y = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 220),
            np.linspace(ylim[0], ylim[1], 340),
        )
        sigma = 0.045 * max(float(np.ptp(x)), float(np.ptp(y)))
        sigma = max(sigma, 1e-6)
        heat = np.zeros_like(grid_x, dtype=np.float32)
        for point, value in zip(coords[active], active_values, strict=True):
            dist2 = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2
            heat += float(value) * np.exp(-dist2 / (2.0 * sigma * sigma))
        heat = scipy.ndimage.gaussian_filter(heat, sigma=1.0)

        hull = scipy.spatial.Delaunay(coords)
        inside = hull.find_simplex(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)) >= 0
        inside = inside.reshape(grid_x.shape)
        heat = np.ma.array(heat, mask=~inside)
        ax.imshow(
            heat,
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cmap_name,
            vmin=0.0,
            vmax=max(float(np.percentile(heat.compressed(), 98.0)), 1e-6),
            interpolation="bilinear",
            alpha=0.62,
        )

    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_pts = coords[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="black", linewidth=1.8, zorder=5)
    except Exception:
        pass

    ax.scatter(
        x[~active],
        y[~active],
        s=14 * point_size_scale,
        c="#d2d2d2",
        edgecolors="none",
        linewidths=0.0,
        zorder=6,
    )
    ax.scatter(
        x[active],
        y[active],
        s=36 * point_size_scale,
        c=magnitude[active],
        cmap=cmap_name,
        vmin=threshold,
        vmax=vmax,
        edgecolors="none",
        linewidths=0.0,
        zorder=7,
    )

    if show_title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def _draw_all_fingers_contact_heatmap(
    *,
    tactile: np.ndarray,
    threshold: float,
    output_path: Path,
    cmap_name: str,
    draw_glow: bool,
    show_title: bool,
    layout_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 4.2), constrained_layout=True)
    for finger, ax in enumerate(axes):
        coords, _ = _taxel_coords(layout_dir, finger)
        magnitude = np.linalg.norm(tactile[finger], axis=-1)
        _draw_contact_heatmap_on_ax(
            ax,
            coords=coords,
            magnitude=magnitude,
            threshold=threshold,
            cmap_name=cmap_name,
            draw_glow=draw_glow,
            show_title=show_title,
            title=FINGER_NAMES[finger],
            point_size_scale=0.85,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="data/grasp_pipette_and_press_w_force_w_depth_0614_good_tactile_7ep")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=None, help="If omitted, choose strongest frame for this finger.")
    parser.add_argument("--finger", type=int, default=1, help="0 thumb, 1 index, 2 middle, 3 ring, 4 little.")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--cmap", default="turbo")
    parser.add_argument("--all-fingers", action="store_true", help="Draw all five fingers in one row.")
    parser.add_argument("--no-colorbar", action="store_true", help="Do not draw the colorbar for single-finger plots.")
    parser.add_argument("--no-glow", action="store_true", help="Disable Gaussian heatmap glow behind active points.")
    parser.add_argument("--show-title", action="store_true", help="Draw a title above the figure.")
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xhand_taxel_contact_heatmaps"))
    args = parser.parse_args()

    if not (0 <= args.finger < len(FINGER_NAMES)):
        raise ValueError(f"finger must be in [0, 4], got {args.finger}")

    repo = Path(args.repo_id)
    episode_path = _episode_file(repo, args.episode_index)
    df = pd.read_parquet(episode_path)
    frame_index, state = _select_frame(df, args.finger, args.frame_index)
    tactile = _extract_raw_tactile(state)
    if args.all_fingers:
        output_path = args.output_dir / f"ep{args.episode_index:06d}_frame{frame_index:06d}_all_fingers_contact_heatmap.png"
        _draw_all_fingers_contact_heatmap(
            tactile=tactile,
            threshold=args.threshold,
            output_path=output_path,
            cmap_name=args.cmap,
            draw_glow=not args.no_glow,
            show_title=args.show_title,
            layout_dir=args.layout_dir,
        )
        summary = {
            "repo_id": str(repo),
            "episode_index": args.episode_index,
            "frame_index": frame_index,
            "threshold": args.threshold,
            "num_active_taxels_per_finger": [
                int(np.sum(np.linalg.norm(tactile[finger], axis=-1) > args.threshold)) for finger in range(5)
            ],
            "max_magnitude_per_finger": [
                float(np.max(np.linalg.norm(tactile[finger], axis=-1))) for finger in range(5)
            ],
            "output_png": str(output_path),
        }
        output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    magnitude = np.linalg.norm(tactile[args.finger], axis=-1)
    coords, layout_path = _taxel_coords(args.layout_dir, args.finger)

    stem = f"{repo.name}_ep{args.episode_index:06d}_frame{frame_index:06d}_{FINGER_NAMES[args.finger]}"
    output_path = args.output_dir / f"{stem}_contact_heatmap.png"
    title = (
        f"{FINGER_NAMES[args.finger]} contact heatmap, "
        f"episode {args.episode_index}, frame {frame_index}"
    )
    if args.no_colorbar:
        fig, ax = plt.subplots(figsize=(3.0, 5.2))
        _draw_contact_heatmap_on_ax(
            ax,
            coords=coords,
            magnitude=magnitude,
            threshold=args.threshold,
            cmap_name=args.cmap,
            draw_glow=not args.no_glow,
            show_title=args.show_title,
            title=title,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
    else:
        _draw_contact_heatmap(
            coords=coords,
            magnitude=magnitude,
            threshold=args.threshold,
            output_path=output_path,
            title=title,
            cmap_name=args.cmap,
            draw_glow=not args.no_glow,
            show_title=args.show_title,
        )

    summary = {
        "repo_id": str(repo),
        "episode_index": args.episode_index,
        "frame_index": frame_index,
        "finger": args.finger,
        "finger_name": FINGER_NAMES[args.finger],
        "threshold": args.threshold,
        "layout_json": str(layout_path),
        "num_active_taxels": int(np.sum(magnitude > args.threshold)),
        "max_magnitude": float(np.max(magnitude)),
        "mean_magnitude": float(np.mean(magnitude)),
        "output_png": str(output_path),
    }
    output_json = output_path.with_suffix(".json")
    output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
