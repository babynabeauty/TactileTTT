#!/usr/bin/env python3
"""Visualize one frame for the pretrained XHand patch tactile encoder heads.

The script always computes and saves the ground-truth patch targets from raw
XHand tactile force. If ``--params`` is provided, it also loads the Stage-1
``xhand_patch_tactile_encoder_pretrain`` model and visualizes the three decoder
head predictions.

The three head targets are:

  1. patch distribution: relative contact strength over 5 patches.
  2. patch contact: binary contact mask for each patch.
  3. patch summary: 7 values per patch; the figure shows its strength channel,
     while the saved ``.npz`` keeps all 7 dimensions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.spatial

from openpi.models import model as _model
from openpi.models.tactile_tokenizer import AdaptiveFingertipPatchTokenizer
from openpi.policies import xhand_policy
from openpi.training import config as _config


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
PATCH_NAMES = ("tip", "center", "base", "left", "right")
DEFAULT_LAYOUT_DIR = Path("Xhand1交付资料-带触觉/触觉传感器")


def _episode_file(repo: Path, episode_index: int) -> Path:
    candidates = sorted(repo.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find episode_{episode_index:06d}.parquet under {repo}")
    return candidates[0]


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
    return np.stack(chunks, axis=0).astype(np.float32)  # [5, 120, 3]


def _load_frame(repo: Path, episode_index: int, frame_index: int) -> tuple[float, np.ndarray]:
    episode_path = _episode_file(repo, episode_index)
    df = pd.read_parquet(episode_path)
    row = df[df["frame_index"] == frame_index]
    if row.empty:
        raise ValueError(f"frame_index={frame_index} not found in {episode_path}")
    item = row.iloc[0]
    timestamp = float(item["timestamp"]) if "timestamp" in item else float(frame_index) / 15.0
    state = np.asarray(item["observation.state"], dtype=np.float32)
    return timestamp, _extract_raw_tactile(state)


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


def _taxel_coords(layout_dir: Path, finger_idx: int) -> tuple[np.ndarray, Path]:
    if finger_idx == 0:
        path = layout_dir / "points_t30_right_hand_transformed.json"
        if not path.exists():
            path = layout_dir / "points.t30 (2).json"
        coords_3d = _read_measurement_points(path)
        coords_2d = coords_3d[:, [0, 1]]
    else:
        path = layout_dir / "points_t16_transformed (1).json"
        if not path.exists():
            path = layout_dir / "points_t16_transformed (2).json"
        if not path.exists():
            path = layout_dir / "points.t16 (2).json"
        coords_3d = _read_measurement_points(path)
        coords_2d = coords_3d[:, [1, 2]]
    return coords_2d.astype(np.float32), path


def _compute_gt_targets(
    tactile: np.ndarray,
    *,
    contact_threshold: float,
    contact_temperature: float,
) -> dict[str, np.ndarray]:
    """Numpy version of PatchInformedFingerTokenizer.patch_reconstruction_targets."""
    forces = tactile[None].astype(np.float32)  # [1, 5, 120, 3]
    magnitude = np.linalg.norm(forces, axis=-1)  # [1, 5, 120]
    temperature = max(contact_temperature, 1e-6)
    gate = 1.0 / (1.0 + np.exp(-((magnitude - contact_threshold) / temperature)))

    patch_ids = np.asarray(AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(5, 120), dtype=np.int32)
    patch_masks = np.eye(5, dtype=np.float32)[patch_ids]  # [F, P, R]
    patch_masks = np.transpose(patch_masks, (0, 2, 1))  # [F, R, P]

    masked_gate = gate[:, :, None, :] * patch_masks[None, :, :, :]  # [1, F, R, P]
    gate_sum = np.sum(masked_gate, axis=-1)
    gate_denom = np.maximum(gate_sum, 1e-6)
    gated_force_mean = np.einsum("bfrp,bfpc->bfrc", masked_gate, forces) / gate_denom[..., None]

    abs_forces = np.abs(forces)
    patch_abs_max = np.max(
        np.where(patch_masks[None, :, :, :, None] > 0, abs_forces[:, :, None, :, :], 0.0),
        axis=-2,
    )
    patch_strength = np.max(
        np.where(patch_masks[None, :, :, :] > 0, magnitude[:, :, None, :], 0.0),
        axis=-1,
    )

    contact = patch_strength > contact_threshold
    active_strength = np.where(contact, patch_strength, 0.0)
    strength_sum = np.sum(active_strength, axis=-1, keepdims=True)
    uniform = np.full_like(active_strength, 1.0 / 5.0)
    distribution = np.where(strength_sum > 1e-6, active_strength / np.maximum(strength_sum, 1e-6), uniform)
    summary = np.concatenate([gated_force_mean, patch_abs_max, patch_strength[..., None]], axis=-1)

    return {
        "target_dist": distribution[0].astype(np.float32),  # [F, R]
        "target_contact": contact[0].astype(np.float32),
        "target_summary": summary[0].astype(np.float32),  # [F, R, 7]
        "target_strength": summary[0, ..., -1].astype(np.float32),
        "target_mean_force": summary[0, ..., :3].astype(np.float32),
        "target_abs_max_force": summary[0, ..., 3:6].astype(np.float32),
        "patch_ids": patch_ids.astype(np.int32),
    }


def _load_model(params_path: Path, config_name: str):
    train_config = _config.get_config(config_name)
    params = _model.restore_params(params_path, restore_type=np.ndarray)
    return train_config.model.load(params)


def _predict_heads(model, tactile: np.ndarray) -> dict[str, np.ndarray]:
    effort = jnp.asarray(tactile[None, None], dtype=jnp.float32)  # [1, 1, 5, 120, 3]
    times = jnp.zeros((1,), dtype=jnp.float32)
    tokens = model.patch_encoder._encode_steps(effort, times, future=False, include_temporal=False)

    dist_logits = model.patch_distribution_head(tokens)
    pred_dist = jax.nn.softmax(dist_logits.astype(jnp.float32), axis=-1)

    pred_summary = model.patch_summary_head(tokens)
    pred_summary = einops.rearrange(
        pred_summary,
        "b t f (r c) -> b t f r c",
        r=model.num_patches,
        c=model.summary_dim,
    )

    contact_logits = model.patch_contact_head(tokens)
    pred_contact = jax.nn.sigmoid(contact_logits.astype(jnp.float32))

    pred_summary_np = np.asarray(pred_summary[0, 0], dtype=np.float32)
    pred_strength = (
        np.linalg.norm(pred_summary_np, axis=-1)
        if getattr(model, "force_only_summary", False)
        else pred_summary_np[..., -1]
    )
    return {
        "pred_dist": np.asarray(pred_dist[0, 0], dtype=np.float32),
        "pred_contact": np.asarray(pred_contact[0, 0], dtype=np.float32),
        "pred_summary": pred_summary_np,
        "pred_strength": pred_strength,
        "pred_mean_force": pred_summary_np[..., :3],
        "pred_abs_max_force": (
            pred_summary_np[..., 3:6]
            if not getattr(model, "force_only_summary", False)
            else np.abs(pred_summary_np)
        ),
    }


def _patch_grid(
    *,
    layout_dir: Path,
    finger_idx: int,
    patch_values: np.ndarray,
    resolution: tuple[int, int] = (190, 260),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    coords, _ = _taxel_coords(layout_dir, finger_idx)
    patch_ids = np.asarray(AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(5, 120)[finger_idx], dtype=np.int32)
    point_values = patch_values[patch_ids].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]
    pad_x = max(0.5, 0.08 * float(np.ptp(x)))
    pad_y = max(0.5, 0.08 * float(np.ptp(y)))
    xlim = (float(x.min() - pad_x), float(x.max() + pad_x))
    ylim = (float(y.min() - pad_y), float(y.max() + pad_y))
    grid_x, grid_y = np.meshgrid(np.linspace(*xlim, resolution[0]), np.linspace(*ylim, resolution[1]))

    tree = scipy.spatial.cKDTree(coords)
    _, nearest = tree.query(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1))
    grid_values = point_values[nearest].reshape(grid_x.shape)
    grid_patch_ids = patch_ids[nearest].reshape(grid_x.shape).astype(np.float32)

    hull = scipy.spatial.Delaunay(coords)
    inside = hull.find_simplex(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)) >= 0
    inside = inside.reshape(grid_x.shape)
    alpha = scipy.ndimage.gaussian_filter(inside.astype(np.float32), sigma=1.0)
    grid_patch_ids = np.where(inside, grid_patch_ids, np.nan)
    return grid_values, grid_patch_ids, alpha, coords, (xlim[0], xlim[1], ylim[0], ylim[1])


def _draw_patch_values(
    ax,
    *,
    layout_dir: Path,
    finger_idx: int,
    values: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    grid_values, grid_patch_ids, alpha, coords, extent = _patch_grid(layout_dir=layout_dir, finger_idx=finger_idx, patch_values=values)
    masked = np.ma.array(grid_values, mask=alpha < 0.2)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1, 1, 1, 0))
    ax.imshow(
        masked,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=max(vmax, vmin + 1e-6),
        interpolation="bilinear",
    )
    ax.contour(
        grid_patch_ids,
        levels=[0.5, 1.5, 2.5, 3.5],
        origin="lower",
        extent=extent,
        colors="black",
        linewidths=1.2,
        alpha=0.95,
        zorder=4,
    )
    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_pts = coords[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="black", linewidth=1.7, zorder=5)
    except Exception:
        pass
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_raw_taxel_force(
    ax,
    *,
    layout_dir: Path,
    finger_idx: int,
    tactile: np.ndarray,
    threshold: float,
    cmap_name: str,
    vmax: float,
    title: str,
) -> None:
    coords, _ = _taxel_coords(layout_dir, finger_idx)
    magnitude = np.linalg.norm(tactile[finger_idx], axis=-1)
    x = coords[:, 0]
    y = coords[:, 1]
    active = magnitude > threshold

    pad_x = max(0.8, 0.08 * float(np.ptp(x)))
    pad_y = max(0.8, 0.08 * float(np.ptp(y)))
    xlim = (float(x.min() - pad_x), float(x.max() + pad_x))
    ylim = (float(y.min() - pad_y), float(y.max() + pad_y))

    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_pts = coords[hull.vertices]
        ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=plt.get_cmap(cmap_name)(0.0), zorder=0)
    except Exception:
        pass

    if np.any(active):
        grid_x, grid_y = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 220),
            np.linspace(ylim[0], ylim[1], 300),
        )
        sigma = max(0.045 * max(float(np.ptp(x)), float(np.ptp(y))), 1e-6)
        heat = np.zeros_like(grid_x, dtype=np.float32)
        for point, value in zip(coords[active], magnitude[active], strict=True):
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
            vmax=max(vmax, 1e-6),
            interpolation="bilinear",
            alpha=0.62,
        )

    try:
        hull = scipy.spatial.ConvexHull(coords)
        hull_pts = coords[hull.vertices]
        hull_pts = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_pts[:, 0], hull_pts[:, 1], color="black", linewidth=1.7, zorder=5)
    except Exception:
        pass

    ax.scatter(x[~active], y[~active], s=12, c="#d2d2d2", edgecolors="none", linewidths=0.0, zorder=6)
    scatter = ax.scatter(
        x[active],
        y[active],
        s=32,
        c=magnitude[active],
        cmap=cmap_name,
        vmin=threshold,
        vmax=max(vmax, threshold + 1e-6),
        edgecolors="none",
        linewidths=0.0,
        zorder=7,
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return scatter


def _draw_patch_values_with_text(
    ax,
    *,
    layout_dir: Path,
    finger_idx: int,
    values: np.ndarray,
    text_values: np.ndarray,
    text_format: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title: str,
) -> None:
    _draw_patch_values(
        ax,
        layout_dir=layout_dir,
        finger_idx=finger_idx,
        values=values,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        title=title,
    )
    coords, _ = _taxel_coords(layout_dir, finger_idx)
    patch_ids = np.asarray(AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(5, 120)[finger_idx], dtype=np.int32)
    for patch_id in range(5):
        patch_coords = coords[patch_ids == patch_id]
        center = np.mean(patch_coords, axis=0)
        text = text_format.format(float(text_values[patch_id]))
        ax.text(
            float(center[0]),
            float(center[1]),
            text,
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "black", "edgecolor": "none", "alpha": 0.42},
        )


def _plot_paper_summary(
    *,
    tactile: np.ndarray,
    arrays: dict[str, np.ndarray],
    output_path: Path,
    layout_dir: Path,
    finger_idx: int,
    title: str,
    raw_threshold: float,
) -> None:
    distribution = arrays["target_dist"][finger_idx]
    strength = arrays["target_strength"][finger_idx]
    raw_mag = np.linalg.norm(tactile[finger_idx], axis=-1)
    raw_vmax = float(max(np.percentile(raw_mag[raw_mag > raw_threshold], 98.0), np.max(raw_mag), raw_threshold + 1e-6)) if np.any(raw_mag > raw_threshold) else float(max(np.max(raw_mag), 1.0))
    strength_vmax = float(max(np.max(strength), 1e-6))

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), constrained_layout=True)
    raw_scatter = _draw_raw_taxel_force(
        axes[0],
        layout_dir=layout_dir,
        finger_idx=finger_idx,
        tactile=tactile,
        threshold=raw_threshold,
        cmap_name="magma",
        vmax=raw_vmax,
        title="Raw taxel force",
    )
    _draw_patch_values_with_text(
        axes[1],
        layout_dir=layout_dir,
        finger_idx=finger_idx,
        values=distribution,
        text_values=100.0 * distribution,
        text_format="{:.0f}%",
        cmap_name="viridis",
        vmin=0.0,
        vmax=1.0,
        title="Patch distribution\nrelative, sum=1",
    )
    _draw_patch_values_with_text(
        axes[2],
        layout_dir=layout_dir,
        finger_idx=finger_idx,
        values=strength,
        text_values=strength,
        text_format="{:.1f}",
        cmap_name="magma",
        vmin=0.0,
        vmax=strength_vmax,
        title="Patch strength\nabsolute force",
    )

    if raw_scatter is not None:
        cbar = fig.colorbar(raw_scatter, ax=axes[0], fraction=0.052, pad=0.02)
        cbar.set_label("force magnitude", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    sm_dist = matplotlib.cm.ScalarMappable(cmap="viridis", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0))
    sm_dist.set_array([])
    cbar = fig.colorbar(sm_dist, ax=axes[1], fraction=0.052, pad=0.02)
    cbar.set_label("relative ratio", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    sm_strength = matplotlib.cm.ScalarMappable(
        cmap="magma",
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(strength_vmax, 1e-6)),
    )
    sm_strength.set_array([])
    cbar = fig.colorbar(sm_strength, ax=axes[2], fraction=0.052, pad=0.02)
    cbar.set_label("force magnitude", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _plot_heads(
    *,
    arrays: dict[str, np.ndarray],
    prefix: str,
    output_path: Path,
    layout_dir: Path,
    title: str,
) -> None:
    dist = arrays[f"{prefix}_dist"]
    contact = arrays[f"{prefix}_contact"]
    strength = arrays[f"{prefix}_strength"]
    strength_vmax = float(max(np.percentile(strength, 98.0), np.max(strength), 1e-6))

    rows = [
        ("Patch distribution", dist, "viridis", 0.0, 1.0),
        ("Patch contact", contact, "magma", 0.0, 1.0),
        ("Patch strength", strength, "magma", 0.0, strength_vmax),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(12.5, 7.4), constrained_layout=True)
    for row_idx, (row_name, values, cmap_name, vmin, vmax) in enumerate(rows):
        for finger_idx, finger_name in enumerate(FINGER_NAMES):
            _draw_patch_values(
                axes[row_idx, finger_idx],
                layout_dir=layout_dir,
                finger_idx=finger_idx,
                values=values[finger_idx],
                cmap_name=cmap_name,
                vmin=vmin,
                vmax=vmax,
                title=finger_name if row_idx == 0 else "",
            )
            if finger_idx == 0:
                axes[row_idx, finger_idx].text(
                    -0.18,
                    0.5,
                    row_name,
                    transform=axes[row_idx, finger_idx].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
        sm = matplotlib.cm.ScalarMappable(
            cmap=cmap_name,
            norm=matplotlib.colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6)),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=axes[row_idx, :], fraction=0.018, pad=0.01)

    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _plot_comparison(
    *,
    arrays: dict[str, np.ndarray],
    output_path: Path,
    layout_dir: Path,
    title: str,
) -> None:
    panels = [
        ("GT distribution", "target_dist", "viridis", 0.0, 1.0),
        ("Pred distribution", "pred_dist", "viridis", 0.0, 1.0),
        ("GT contact", "target_contact", "magma", 0.0, 1.0),
        ("Pred contact", "pred_contact", "magma", 0.0, 1.0),
        (
            "GT strength",
            "target_strength",
            "magma",
            0.0,
            float(max(np.max(arrays["target_strength"]), np.max(arrays["pred_strength"]), 1e-6)),
        ),
        (
            "Pred strength",
            "pred_strength",
            "magma",
            0.0,
            float(max(np.max(arrays["target_strength"]), np.max(arrays["pred_strength"]), 1e-6)),
        ),
    ]
    fig, axes = plt.subplots(6, 5, figsize=(12.5, 13.5), constrained_layout=True)
    for row_idx, (row_name, key, cmap_name, vmin, vmax) in enumerate(panels):
        values = arrays[key]
        for finger_idx, finger_name in enumerate(FINGER_NAMES):
            _draw_patch_values(
                axes[row_idx, finger_idx],
                layout_dir=layout_dir,
                finger_idx=finger_idx,
                values=values[finger_idx],
                cmap_name=cmap_name,
                vmin=vmin,
                vmax=vmax,
                title=finger_name if row_idx == 0 else "",
            )
            if finger_idx == 0:
                axes[row_idx, finger_idx].text(
                    -0.18,
                    0.5,
                    row_name,
                    transform=axes[row_idx, finger_idx].transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=Path, default=Path("data/grasp_pipette_and_press_w_force_w_depth_0614_good_tactile_7ep"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=321)
    parser.add_argument("--params", type=Path, default=None, help="Optional Stage-1 patch encoder params path.")
    parser.add_argument("--config-name", default="xhand_patch_tactile_encoder_pretrain")
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--contact-temperature", type=float, default=0.5)
    parser.add_argument("--layout-dir", type=Path, default=DEFAULT_LAYOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/single_frame_patch_encoder_heads"))
    parser.add_argument("--paper-finger", type=int, default=1, help="Finger index for the paper summary figure.")
    parser.add_argument("--raw-threshold", type=float, default=5.0)
    args = parser.parse_args()

    repo = args.repo_id
    timestamp, tactile = _load_frame(repo, args.episode_index, args.frame_index)
    arrays = _compute_gt_targets(
        tactile,
        contact_threshold=args.contact_threshold,
        contact_temperature=args.contact_temperature,
    )

    if args.params is not None:
        model = _load_model(args.params, args.config_name)
        arrays.update(_predict_heads(model, tactile))

    stem = f"{repo.name}_ep{args.episode_index:06d}_frame{args.frame_index:06d}"
    output_dir = args.output_dir / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{stem}_patch_encoder_heads_data.npz"
    np.savez_compressed(
        npz_path,
        raw_tactile=tactile,
        raw_magnitude=np.linalg.norm(tactile, axis=-1),
        timestamp=np.asarray(timestamp, dtype=np.float32),
        **arrays,
    )

    gt_png = output_dir / f"{stem}_gt_patch_encoder_heads.png"
    _plot_heads(
        arrays=arrays,
        prefix="target",
        output_path=gt_png,
        layout_dir=args.layout_dir,
        title=f"GT patch encoder head targets, episode {args.episode_index}, frame {args.frame_index}",
    )

    paper_png = output_dir / f"{stem}_{FINGER_NAMES[args.paper_finger]}_paper_patch_summary.png"
    _plot_paper_summary(
        tactile=tactile,
        arrays=arrays,
        output_path=paper_png,
        layout_dir=args.layout_dir,
        finger_idx=args.paper_finger,
        title=f"{FINGER_NAMES[args.paper_finger].capitalize()} tactile patch representation",
        raw_threshold=args.raw_threshold,
    )

    pred_png = None
    comparison_png = None
    if args.params is not None:
        pred_png = output_dir / f"{stem}_pred_patch_encoder_heads.png"
        _plot_heads(
            arrays=arrays,
            prefix="pred",
            output_path=pred_png,
            layout_dir=args.layout_dir,
            title=f"Predicted patch encoder heads, episode {args.episode_index}, frame {args.frame_index}",
        )
        comparison_png = output_dir / f"{stem}_gt_vs_pred_patch_encoder_heads.png"
        _plot_comparison(
            arrays=arrays,
            output_path=comparison_png,
            layout_dir=args.layout_dir,
            title=f"GT vs predicted patch encoder heads, episode {args.episode_index}, frame {args.frame_index}",
        )

    summary = {
        "repo_id": str(repo),
        "episode_index": args.episode_index,
        "frame_index": args.frame_index,
        "timestamp": timestamp,
        "params": str(args.params) if args.params is not None else None,
        "contact_threshold": args.contact_threshold,
        "contact_temperature": args.contact_temperature,
        "active_raw_taxels_threshold_1": int(np.sum(np.linalg.norm(tactile, axis=-1) > 1.0)),
        "active_raw_taxels_threshold_5": int(np.sum(np.linalg.norm(tactile, axis=-1) > 5.0)),
        "npz": str(npz_path),
        "gt_png": str(gt_png),
        "paper_png": str(paper_png),
        "pred_png": str(pred_png) if pred_png is not None else None,
        "comparison_png": str(comparison_png) if comparison_png is not None else None,
    }
    json_path = output_dir / f"{stem}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
