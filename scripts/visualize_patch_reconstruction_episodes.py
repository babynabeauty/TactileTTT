#!/usr/bin/env python3
"""Visualize held-out XHand patch reconstruction over complete episodes.

For one selected validation episode per task, every frame is rendered as:

    Raw tactile heatmap | GT patch strength/contact | Predicted strength/contact

The raw column stays in the sensor's physical scale. Encoder inputs and patch
targets use the exact effort mean/std normalization used during Stage-1
training. This distinction is intentional and recorded in each episode's
metadata.json.
"""

from __future__ import annotations

import argparse
import collections.abc
import json
import pathlib
import random
import shutil
import subprocess
from collections import defaultdict

import einops
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training.data_loader import _arrow_column_to_numpy

import visualize_single_frame_patch_encoder_heads as _single_vis


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _load_episode_filter(path: pathlib.Path) -> set[int]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict):
        episodes = (
            data.get("episodes")
            or data.get("episode_indices")
            or data.get("episode_index")
            or data.get("val")
        )
    else:
        raise ValueError(f"Unsupported episode filter format: {path}")
    if episodes is None:
        raise ValueError(f"No episode list found in {path}")
    return {int(episode) for episode in episodes}


def _task_metadata(repo: pathlib.Path, selected_episodes: set[int]) -> dict[object, dict]:
    task_name_to_index: dict[str, int] = {}
    task_index_to_name: dict[int, str] = {}
    tasks_path = repo / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with tasks_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "task" in row and "task_index" in row:
                    task_name = str(row["task"])
                    task_index = int(row["task_index"])
                    task_name_to_index[task_name] = task_index
                    task_index_to_name[task_index] = task_name

    grouped: dict[object, dict] = {}
    episodes_path = repo / "meta" / "episodes.jsonl"
    with episodes_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            episode = int(row.get("episode_index", row.get("index")))
            if episode not in selected_episodes:
                continue
            if "task_index" in row:
                task_key: object = int(row["task_index"])
            elif row.get("tasks"):
                task_name = str(row["tasks"][0])
                task_key = task_name_to_index.get(task_name, task_name)
            elif "task" in row:
                task_name = str(row["task"])
                task_key = task_name_to_index.get(task_name, task_name)
            else:
                task_key = 0
            task_name = task_index_to_name.get(task_key, None) if isinstance(task_key, int) else str(task_key)
            if task_name is None:
                if row.get("tasks"):
                    task_name = str(row["tasks"][0])
                elif "task" in row:
                    task_name = str(row["task"])
                else:
                    task_name = f"task_{task_key}"
            item = grouped.setdefault(task_key, {"task_name": task_name, "episodes": []})
            item["episodes"].append(episode)

    for item in grouped.values():
        item["episodes"] = sorted(set(item["episodes"]))
    return grouped


def _episode_file(repo: pathlib.Path, episode_index: int) -> pathlib.Path:
    candidates = sorted(repo.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"episode_{episode_index:06d}.parquet not found under {repo}")
    return candidates[0]


def _read_episode(repo: pathlib.Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _episode_file(repo, episode_index)
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    columns = ["observation.state"]
    if "frame_index" in available:
        columns.append("frame_index")
    if "timestamp" in available:
        columns.append("timestamp")
    table = parquet.read(columns=columns)
    states = _arrow_column_to_numpy(table["observation.state"]).astype(np.float32)
    tactile = np.stack([_single_vis._extract_raw_tactile(state) for state in states], axis=0)
    frame_indices = (
        np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if "frame_index" in table.column_names
        else np.arange(states.shape[0], dtype=np.int64)
    )
    timestamps = (
        np.asarray(table["timestamp"].to_numpy(), dtype=np.float32)
        if "timestamp" in table.column_names
        else frame_indices.astype(np.float32) / 15.0
    )
    return tactile, frame_indices, timestamps


def _contact_richness(tactile: np.ndarray, threshold: float) -> float:
    magnitude = np.linalg.norm(tactile, axis=-1)
    frame_energy = np.sum(np.where(magnitude > threshold, magnitude, 0.0), axis=(1, 2))
    keep = max(1, int(np.ceil(0.1 * frame_energy.size)))
    return float(np.mean(np.partition(frame_energy, -keep)[-keep:]))


def _select_episodes(
    repo: pathlib.Path,
    grouped: dict[object, dict],
    *,
    selection: str,
    seed: int,
    raw_contact_threshold: float,
) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for task_key, item in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        candidates = list(item["episodes"])
        if not candidates:
            continue
        scores = {}
        if selection == "first":
            episode = candidates[0]
        elif selection == "random":
            episode = rng.choice(candidates)
        elif selection == "max_contact":
            for candidate in candidates:
                tactile, _, _ = _read_episode(repo, candidate)
                scores[candidate] = _contact_richness(tactile, raw_contact_threshold)
            episode = max(candidates, key=lambda candidate: (scores[candidate], -candidate))
        else:
            raise ValueError(f"Unsupported selection mode: {selection}")
        selected.append(
            {
                "task_key": task_key,
                "task_name": item["task_name"],
                "episode_index": episode,
                "candidate_episodes": candidates,
                "contact_richness": scores.get(episode),
            }
        )
    return selected


def _normalize_tactile(tactile: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    std = np.asarray(std, dtype=np.float32).reshape(-1)
    feature_shape = tactile.shape[1:]
    if mean.size == feature_shape[-1]:
        return ((tactile - mean) / (std + 1e-6)).astype(np.float32)
    if mean.size == feature_shape[-2] * feature_shape[-1]:
        flat = tactile.reshape(tactile.shape[0], feature_shape[0], -1)
        return ((flat - mean) / (std + 1e-6)).reshape(tactile.shape).astype(np.float32)
    if mean.size == int(np.prod(feature_shape)):
        flat = tactile.reshape(tactile.shape[0], -1)
        return ((flat - mean) / (std + 1e-6)).reshape(tactile.shape).astype(np.float32)
    raise ValueError(f"Cannot apply effort stats of length {mean.size} to tactile shape {feature_shape}")


def _load_effort_norm(assets_dir: pathlib.Path, asset_id: str) -> tuple[np.ndarray, np.ndarray, pathlib.Path]:
    norm_dir = assets_dir.resolve() / asset_id
    norm_stats = _normalize.load(norm_dir)
    if "effort" not in norm_stats:
        raise KeyError(f"No effort statistics in {norm_dir / 'norm_stats.json'}")
    effort = norm_stats["effort"]
    return np.asarray(effort.mean), np.asarray(effort.std), norm_dir / "norm_stats.json"


def _iter_tree_leaves(tree, path=()):
    if isinstance(tree, collections.abc.Mapping):
        for key, value in tree.items():
            yield from _iter_tree_leaves(value, (*path, str(key)))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from _iter_tree_leaves(value, (*path, str(index)))
    else:
        yield path, tree


def _infer_checkpoint_encoder_type(params) -> str | None:
    leaves = list(_iter_tree_leaves(params))
    paths = ("/".join(path) for path, _ in leaves)
    if any("patch_encoder/patch_stat_proj_in" in path for path in paths):
        return "patch_informed"

    paths = ("/".join(path) for path, _ in leaves)
    if any("patch_encoder/point_score" in path for path in paths):
        return "raw_spatial"

    for path, value in leaves:
        if "patch_encoder/force_proj_in/kernel" not in "/".join(path):
            continue
        shape = getattr(value, "shape", ())
        if shape and int(shape[0]) == 3:
            return "raw_spatial"
        if shape and int(shape[0]) > 3:
            return "raw_mlp"
    return None


def _infer_checkpoint_head_type(params) -> str | None:
    leaves = list(_iter_tree_leaves(params))
    paths = ["/".join(path) for path, _ in leaves]
    if any("patch_force_mean_head/" in path for path in paths):
        return "mean_force"
    if any("patch_strength_head/" in path for path in paths):
        return "strength"
    for path, value in leaves:
        if "patch_summary_head/kernel" not in "/".join(path):
            continue
        shape = getattr(value, "shape", ())
        if shape and int(shape[-1]) == 15:
            return "force_three_head"
    if any("patch_distribution_head/" in path for path in paths):
        return "full_heads"
    return None


def _load_model(config_name: str, params: pathlib.Path):
    config = _config.get_config(config_name)
    restored = _model.restore_params(params, restore_type=np.ndarray)
    expected_type = getattr(config.model, "pretrain_tactile_encoder", None)
    checkpoint_type = _infer_checkpoint_encoder_type(restored)
    if expected_type is not None and checkpoint_type is not None and expected_type != checkpoint_type:
        config_by_type = {
            "patch_informed": "xhand_patch_tactile_encoder_pretrain",
            "raw_spatial": "xhand_raw_spatial_tactile_encoder_pretrain",
            "raw_mlp": "xhand_raw_mlp_tactile_encoder_pretrain",
        }
        raise ValueError(
            f"Encoder checkpoint/config mismatch: checkpoint contains {checkpoint_type!r} parameters, "
            f"but --config-name={config_name!r} builds {expected_type!r}. "
            f"Use --config-name {config_by_type[checkpoint_type]}."
        )
    checkpoint_head_type = _infer_checkpoint_head_type(restored)
    config_head_type = (
        "mean_force"
        if isinstance(config.model, pi0_config.XHandPatchMeanForceEncoderPretrainConfig)
        else "strength"
        if isinstance(config.model, pi0_config.XHandPatchStrengthEncoderPretrainConfig)
        else "force_three_head"
        if isinstance(config.model, pi0_config.XHandPatchForceThreeHeadEncoderPretrainConfig)
        else "full_heads"
    )
    if checkpoint_head_type is not None and checkpoint_head_type != config_head_type:
        config_by_head_type = {
            "mean_force": "xhand_patch_mean_force_contact_distribution_encoder_pretrain",
            "strength": "xhand_patch_strength_contact_encoder_pretrain",
            "force_three_head": "xhand_patch_force_three_head_encoder_pretrain",
            "full_heads": "xhand_patch_tactile_encoder_pretrain",
        }
        raise ValueError(
            f"Decoder checkpoint/config mismatch: checkpoint contains {checkpoint_head_type!r} heads, "
            f"but --config-name={config_name!r} builds {config_head_type!r} heads. "
            f"Use --config-name {config_by_head_type[checkpoint_head_type]}."
        )
    model = config.model.load(restored)
    model.eval()
    return model


def _model_batch(model, normalized_tactile: jax.Array) -> dict[str, jax.Array]:
    effort = normalized_tactile[:, None, ...].astype(jnp.float32)
    times = jnp.zeros((1,), dtype=jnp.float32)
    tokens = model.patch_encoder._encode_steps(effort, times, future=False, include_temporal=False)
    target_dist, target_summary, target_contact = model.patch_encoder.patch_reconstruction_targets(effort)
    force_only_summary = getattr(model, "force_only_summary", False)
    target_force = target_summary[..., : model.dim_per_point]
    if force_only_summary:
        target_force = target_force * target_contact[..., None]
    target_strength = (
        jnp.linalg.norm(target_force, axis=-1)
        if force_only_summary
        else target_summary[..., -1]
    )

    if hasattr(model, "patch_force_mean_head"):
        pred_force = einops.rearrange(
            model.patch_force_mean_head(tokens),
            "b t f (r c) -> b t f r c",
            r=model.num_patches,
            c=model.dim_per_point,
        ).astype(jnp.float32)
        pred_strength = jnp.linalg.norm(pred_force, axis=-1)
        pred_dist = pred_strength / jnp.maximum(jnp.sum(pred_strength, axis=-1, keepdims=True), 1e-6)
        pred_contact = jax.nn.sigmoid(
            (pred_strength - model.contact_threshold) / max(model.contact_temperature, 1e-6)
        )
        pred_summary = jnp.concatenate(
            [pred_force, jnp.abs(pred_force), pred_strength[..., None]],
            axis=-1,
        )
    elif hasattr(model, "patch_strength_head"):
        pred_strength = jax.nn.softplus(model.patch_strength_head(tokens).astype(jnp.float32))
        pred_dist = pred_strength / jnp.maximum(jnp.sum(pred_strength, axis=-1, keepdims=True), 1e-6)
        pred_contact = jax.nn.sigmoid(
            (pred_strength - model.contact_threshold) / max(model.contact_temperature, 1e-6)
        )
        zeros = jnp.zeros((*pred_strength.shape, 2 * model.dim_per_point), dtype=jnp.float32)
        pred_summary = jnp.concatenate([zeros, pred_strength[..., None]], axis=-1)
    else:
        pred_dist = jax.nn.softmax(model.patch_distribution_head(tokens).astype(jnp.float32), axis=-1)
        pred_summary = einops.rearrange(
            model.patch_summary_head(tokens),
            "b t f (r c) -> b t f r c",
            r=model.num_patches,
            c=model.summary_dim,
        ).astype(jnp.float32)
        pred_contact = jax.nn.sigmoid(model.patch_contact_head(tokens).astype(jnp.float32))
        pred_strength = (
            jnp.linalg.norm(pred_summary, axis=-1)
            if getattr(model, "force_only_summary", False)
            else pred_summary[..., -1]
        )
    pred_force = pred_summary[..., : model.dim_per_point]
    target_summary_output = (
        target_force
        if force_only_summary
        else target_summary
    )
    return {
        "target_dist": target_dist[:, 0].astype(jnp.float32),
        "target_summary": target_summary_output[:, 0].astype(jnp.float32),
        "target_contact": target_contact[:, 0].astype(jnp.float32),
        "target_strength": target_strength[:, 0].astype(jnp.float32),
        "target_force": target_force[:, 0].astype(jnp.float32),
        "pred_dist": pred_dist[:, 0],
        "pred_summary": pred_summary[:, 0],
        "pred_contact": pred_contact[:, 0],
        "pred_strength": pred_strength[:, 0],
        "pred_force": pred_force[:, 0],
    }


def _predict_episode(model, normalized_tactile: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    predict = jax.jit(lambda batch: _model_batch(model, batch))
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, normalized_tactile.shape[0], batch_size):
        batch = jnp.asarray(normalized_tactile[start : start + batch_size], dtype=jnp.float32)
        output = jax.device_get(predict(batch))
        for key, value in output.items():
            chunks[key].append(np.asarray(value, dtype=np.float32))
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def _patch_centers(layout_dir: pathlib.Path, finger_idx: int) -> tuple[np.ndarray, np.ndarray]:
    coords, _ = _single_vis._taxel_coords(layout_dir, finger_idx)
    patch_ids = np.asarray(
        _single_vis.AdaptiveFingertipPatchTokenizer._official_xhand_patch_ids(5, 120)[finger_idx],
        dtype=np.int32,
    )
    centers = np.stack([np.mean(coords[patch_ids == patch_id], axis=0) for patch_id in range(5)], axis=0)
    return centers, patch_ids


def _annotate_patch_panel(
    ax,
    *,
    layout_dir: pathlib.Path,
    finger_idx: int,
    strength: np.ndarray,
    contact: np.ndarray,
    predicted: bool,
    fontsize: float,
) -> None:
    centers, _ = _patch_centers(layout_dir, finger_idx)
    for patch_id, center in enumerate(centers):
        if predicted:
            text = f"p={float(contact[patch_id]):.2f}\ns={float(strength[patch_id]):.2f}"
        else:
            symbol = "C" if float(contact[patch_id]) >= 0.5 else "-"
            text = f"{symbol}\ns={float(strength[patch_id]):.2f}"
        ax.text(
            float(center[0]),
            float(center[1]),
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "black", "edgecolor": "none", "alpha": 0.62},
            zorder=8,
        )


def _plot_frame(
    *,
    raw_tactile: np.ndarray,
    arrays: dict[str, np.ndarray],
    frame_position: int,
    actual_frame_index: int,
    timestamp: float,
    task_name: str,
    episode_index: int,
    layout_dir: pathlib.Path,
    output_path: pathlib.Path,
    raw_threshold: float,
    raw_vmax: float,
    strength_vmax: float,
    show_patch_labels: bool,
    patch_label_fontsize: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(5, 4, figsize=(11.4, 12.4), constrained_layout=True)
    for finger_idx, finger_name in enumerate(FINGER_NAMES):
        _single_vis._draw_raw_taxel_force(
            axes[finger_idx, 0],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            tactile=raw_tactile[frame_position],
            threshold=raw_threshold,
            cmap_name="magma",
            vmax=raw_vmax,
            title="",
        )
        target_strength = arrays["target_strength"][frame_position, finger_idx]
        target_contact = arrays["target_contact"][frame_position, finger_idx]
        pred_strength = arrays["pred_strength"][frame_position, finger_idx]
        pred_contact = arrays["pred_contact"][frame_position, finger_idx]
        pred_distribution = arrays["pred_dist"][frame_position, finger_idx]
        target_display_strength = np.where(target_contact >= 0.5, np.maximum(target_strength, 0.0), 0.0)
        pred_display_strength = np.where(pred_contact >= 0.5, np.maximum(pred_strength, 0.0), 0.0)

        _single_vis._draw_patch_values(
            axes[finger_idx, 1],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            values=target_display_strength,
            cmap_name="magma",
            vmin=0.0,
            vmax=strength_vmax,
            title="",
        )
        if show_patch_labels:
            _annotate_patch_panel(
                axes[finger_idx, 1],
                layout_dir=layout_dir,
                finger_idx=finger_idx,
                strength=target_strength,
                contact=target_contact,
                predicted=False,
                fontsize=patch_label_fontsize,
            )

        _single_vis._draw_patch_values(
            axes[finger_idx, 2],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            values=pred_display_strength,
            cmap_name="magma",
            vmin=0.0,
            vmax=strength_vmax,
            title="",
        )
        if show_patch_labels:
            _annotate_patch_panel(
                axes[finger_idx, 2],
                layout_dir=layout_dir,
                finger_idx=finger_idx,
                strength=pred_strength,
                contact=pred_contact,
                predicted=True,
                fontsize=patch_label_fontsize,
            )

        _single_vis._draw_patch_values(
            axes[finger_idx, 3],
            layout_dir=layout_dir,
            finger_idx=finger_idx,
            values=pred_distribution,
            cmap_name="viridis",
            vmin=0.0,
            vmax=1.0,
            title="",
        )

        axes[finger_idx, 0].text(
            -0.08,
            0.5,
            finger_name,
            transform=axes[finger_idx, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    axes[0, 0].set_title("Raw tactile\nheatmap", fontsize=10, fontweight="bold")
    axes[0, 1].set_title("GT patch\nforce magnitude", fontsize=10, fontweight="bold")
    axes[0, 2].set_title("Predicted patch\nforce magnitude", fontsize=10, fontweight="bold")
    axes[0, 3].set_title("Predicted patch\ndistribution", fontsize=10, fontweight="bold")

    raw_map = matplotlib.cm.ScalarMappable(
        cmap="magma", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(raw_vmax, 1e-6))
    )
    strength_map = matplotlib.cm.ScalarMappable(
        cmap="magma", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(strength_vmax, 1e-6))
    )
    distribution_map = matplotlib.cm.ScalarMappable(
        cmap="viridis", norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    )
    raw_map.set_array([])
    strength_map.set_array([])
    distribution_map.set_array([])
    raw_bar = fig.colorbar(raw_map, ax=axes[:, 0], fraction=0.018, pad=0.01)
    raw_bar.set_label("raw force magnitude", fontsize=8)
    patch_bar = fig.colorbar(strength_map, ax=axes[:, 1:3], fraction=0.012, pad=0.01)
    patch_bar.set_label("contact-masked patch force magnitude", fontsize=8)
    distribution_bar = fig.colorbar(distribution_map, ax=axes[:, 3], fraction=0.018, pad=0.01)
    distribution_bar.set_label("predicted patch distribution", fontsize=8)

    fig.suptitle(
        f"{task_name}\nepisode {episode_index}, frame {actual_frame_index}, t={timestamp:.2f}s",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.002,
        "Color encodes contact-masked patch force magnitude; dark patches indicate no contact or negligible force.",
        ha="center",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _save_individual_panels(
    *,
    raw_tactile: np.ndarray,
    arrays: dict[str, np.ndarray],
    frame_position: int,
    actual_frame_index: int,
    layout_dir: pathlib.Path,
    output_dir: pathlib.Path,
    raw_threshold: float,
    raw_vmax: float,
    strength_vmax: float,
    show_patch_labels: bool,
    patch_label_fontsize: float,
    dpi: int,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_records: list[dict[str, object]] = []
    panel_specs = (
        ("raw_tactile", "Raw tactile"),
        ("gt_strength_contact", "GT force magnitude"),
        ("pred_strength_contact", "Predicted force magnitude"),
        ("pred_distribution", "Predicted distribution"),
    )

    for finger_idx, finger_name in enumerate(FINGER_NAMES):
        target_strength = arrays["target_strength"][frame_position, finger_idx]
        target_contact = arrays["target_contact"][frame_position, finger_idx]
        pred_strength = arrays["pred_strength"][frame_position, finger_idx]
        pred_contact = arrays["pred_contact"][frame_position, finger_idx]
        pred_distribution = arrays["pred_dist"][frame_position, finger_idx]
        target_display_strength = np.where(
            target_contact >= 0.5, np.maximum(target_strength, 0.0), 0.0
        )
        pred_display_strength = np.where(
            pred_contact >= 0.5, np.maximum(pred_strength, 0.0), 0.0
        )

        for panel_index, (panel_name, panel_title) in enumerate(panel_specs):
            fig, ax = plt.subplots(figsize=(3.0, 3.8), constrained_layout=True)
            if panel_name == "raw_tactile":
                _single_vis._draw_raw_taxel_force(
                    ax,
                    layout_dir=layout_dir,
                    finger_idx=finger_idx,
                    tactile=raw_tactile[frame_position],
                    threshold=raw_threshold,
                    cmap_name="magma",
                    vmax=raw_vmax,
                    title=panel_title,
                )
                color_map = matplotlib.cm.ScalarMappable(
                    cmap="magma",
                    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(raw_vmax, 1e-6)),
                )
                color_label = "raw force magnitude"
            elif panel_name == "gt_strength_contact":
                _single_vis._draw_patch_values(
                    ax,
                    layout_dir=layout_dir,
                    finger_idx=finger_idx,
                    values=target_display_strength,
                    cmap_name="magma",
                    vmin=0.0,
                    vmax=strength_vmax,
                    title=panel_title,
                )
                if show_patch_labels:
                    _annotate_patch_panel(
                        ax,
                        layout_dir=layout_dir,
                        finger_idx=finger_idx,
                        strength=target_strength,
                        contact=target_contact,
                        predicted=False,
                        fontsize=patch_label_fontsize,
                    )
                color_map = matplotlib.cm.ScalarMappable(
                    cmap="magma",
                    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(strength_vmax, 1e-6)),
                )
                color_label = "contact-masked force magnitude"
            elif panel_name == "pred_strength_contact":
                _single_vis._draw_patch_values(
                    ax,
                    layout_dir=layout_dir,
                    finger_idx=finger_idx,
                    values=pred_display_strength,
                    cmap_name="magma",
                    vmin=0.0,
                    vmax=strength_vmax,
                    title=panel_title,
                )
                if show_patch_labels:
                    _annotate_patch_panel(
                        ax,
                        layout_dir=layout_dir,
                        finger_idx=finger_idx,
                        strength=pred_strength,
                        contact=pred_contact,
                        predicted=True,
                        fontsize=patch_label_fontsize,
                    )
                color_map = matplotlib.cm.ScalarMappable(
                    cmap="magma",
                    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(strength_vmax, 1e-6)),
                )
                color_label = "contact-masked force magnitude"
            else:
                _single_vis._draw_patch_values(
                    ax,
                    layout_dir=layout_dir,
                    finger_idx=finger_idx,
                    values=pred_distribution,
                    cmap_name="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    title=panel_title,
                )
                color_map = matplotlib.cm.ScalarMappable(
                    cmap="viridis",
                    norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
                )
                color_label = "patch distribution"

            color_map.set_array([])
            color_bar = fig.colorbar(color_map, ax=ax, fraction=0.052, pad=0.02)
            color_bar.set_label(color_label, fontsize=7)
            color_bar.ax.tick_params(labelsize=7)
            fig.suptitle(f"{finger_name} | frame {actual_frame_index}", fontsize=10)
            output_path = output_dir / f"{finger_idx:02d}_{finger_name}_{panel_index:02d}_{panel_name}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
            plt.close(fig)
            panel_records.append(
                {
                    "finger_index": finger_idx,
                    "finger": finger_name,
                    "panel": panel_name,
                    "path": str(output_path),
                }
            )

    return panel_records


def _make_video(frames_dir: pathlib.Path, output_path: pathlib.Path, fps: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found; skipping video generation", flush=True)
        return False
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=pathlib.Path, default=pathlib.Path("data/taskall-2"))
    parser.add_argument("--params", type=pathlib.Path, required=True)
    parser.add_argument("--config-name", default="xhand_patch_tactile_encoder_pretrain")
    parser.add_argument(
        "--filter-path",
        type=pathlib.Path,
        default=pathlib.Path("outputs/episode_splits/taskall-2_encoder_final_10pct_seed42/val_episodes.json"),
    )
    parser.add_argument("--assets-dir", type=pathlib.Path, default=pathlib.Path("assets/pi0_xhand_tactile_structured_raw_dual_ae"))
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/patch_reconstruction_visualization"))
    parser.add_argument("--layout-dir", type=pathlib.Path, default=_single_vis.DEFAULT_LAYOUT_DIR)
    parser.add_argument("--selection", choices=("first", "random", "max_contact"), default="max_contact")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--save-individual-panels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--raw-contact-threshold", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--show-patch-labels",
        action="store_true",
        help="Overlay per-patch C/p and force values; disabled by default for a cleaner heatmap.",
    )
    parser.add_argument(
        "--patch-label-fontsize",
        type=float,
        default=8.0,
        help="Font size used when --show-patch-labels is enabled (default: 8.0).",
    )
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--make-video", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.frame_stride <= 0 or args.patch_label_fontsize <= 0:
        raise ValueError("batch-size, frame-stride, and patch-label-fontsize must be positive")
    if args.frame_index is not None and args.episode_index is None:
        raise ValueError("--frame-index requires --episode-index")
    repo = args.repo_id.expanduser().resolve()
    filter_path = args.filter_path.expanduser().resolve()
    params = args.params.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    layout_dir = args.layout_dir.expanduser().resolve()
    asset_id = args.asset_id or repo.name

    held_out = {args.episode_index} if args.episode_index is not None else _load_episode_filter(filter_path)
    grouped = _task_metadata(repo, held_out)
    if args.episode_index is None:
        selected = _select_episodes(
            repo,
            grouped,
            selection=args.selection,
            seed=args.seed,
            raw_contact_threshold=args.raw_contact_threshold,
        )
    else:
        selected = []
        for task_key, item in grouped.items():
            if args.episode_index in item["episodes"]:
                selected.append(
                    {
                        "task_key": task_key,
                        "task_name": item["task_name"],
                        "episode_index": args.episode_index,
                        "candidate_episodes": [args.episode_index],
                        "contact_richness": None,
                    }
                )
                break
    if not selected:
        raise RuntimeError("No validation episodes were selected")

    effort_mean, effort_std, norm_path = _load_effort_norm(args.assets_dir, asset_id)
    model = _load_model(args.config_name, params)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_id": str(repo),
        "params": str(params),
        "config_name": args.config_name,
        "filter_path": str(filter_path),
        "normalization": str(norm_path),
        "selection": args.selection,
        "requested_episode_index": args.episode_index,
        "requested_frame_index": args.frame_index,
        "save_individual_panels": args.save_individual_panels,
        "seed": args.seed,
        "episodes": selected,
    }
    (output_root / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for selected_item in selected:
        episode_index = int(selected_item["episode_index"])
        task_name = str(selected_item["task_name"])
        task_key = selected_item["task_key"]
        print(f"Processing task={task_key} episode={episode_index}: {task_name}", flush=True)
        raw_tactile, frame_indices, timestamps = _read_episode(repo, episode_index)
        normalized_tactile = _normalize_tactile(raw_tactile, effort_mean, effort_std)
        arrays = _predict_episode(model, normalized_tactile, args.batch_size)

        episode_dir = output_root / f"task_{str(task_key).replace('/', '_')}_episode_{episode_index:06d}"
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            episode_dir / "patch_reconstruction_data.npz",
            raw_tactile=raw_tactile,
            normalized_tactile=normalized_tactile,
            frame_indices=frame_indices,
            timestamps=timestamps,
            **arrays,
        )

        raw_magnitude = np.linalg.norm(raw_tactile, axis=-1)
        active_raw = raw_magnitude[raw_magnitude > args.raw_contact_threshold]
        raw_vmax = float(np.percentile(active_raw, 99.0)) if active_raw.size else 1.0
        target_display_strength = np.where(
            arrays["target_contact"] >= 0.5, np.maximum(arrays["target_strength"], 0.0), 0.0
        )
        pred_display_strength = np.where(
            arrays["pred_contact"] >= 0.5, np.maximum(arrays["pred_strength"], 0.0), 0.0
        )
        combined_strength = np.concatenate(
            [target_display_strength.reshape(-1), pred_display_strength.reshape(-1)]
        )
        positive_strength = combined_strength[combined_strength > 0]
        strength_vmax = float(np.percentile(positive_strength, 99.0)) if positive_strength.size else 1.0

        if args.frame_index is None:
            frame_positions = list(range(0, raw_tactile.shape[0], args.frame_stride))
        else:
            matches = np.flatnonzero(frame_indices == args.frame_index)
            if matches.size == 0:
                raise ValueError(
                    f"Frame index {args.frame_index} not found in episode {episode_index}; "
                    f"available range is {int(frame_indices.min())}..{int(frame_indices.max())}."
                )
            frame_positions = [int(matches[0])]

        rendered = []
        for output_frame, frame_position in enumerate(frame_positions):
            actual_frame_index = int(frame_indices[frame_position])
            output_name_index = actual_frame_index if args.frame_index is not None else output_frame
            output_path = frames_dir / f"frame_{output_name_index:06d}.png"
            _plot_frame(
                raw_tactile=raw_tactile,
                arrays=arrays,
                frame_position=frame_position,
                actual_frame_index=actual_frame_index,
                timestamp=float(timestamps[frame_position]),
                task_name=task_name,
                episode_index=episode_index,
                layout_dir=layout_dir,
                output_path=output_path,
                raw_threshold=args.raw_contact_threshold,
                raw_vmax=raw_vmax,
                strength_vmax=strength_vmax,
                show_patch_labels=args.show_patch_labels,
                patch_label_fontsize=args.patch_label_fontsize,
                dpi=args.dpi,
            )
            panel_records = []
            if args.save_individual_panels:
                panel_records = _save_individual_panels(
                    raw_tactile=raw_tactile,
                    arrays=arrays,
                    frame_position=frame_position,
                    actual_frame_index=actual_frame_index,
                    layout_dir=layout_dir,
                    output_dir=episode_dir / "panels" / f"frame_{actual_frame_index:06d}",
                    raw_threshold=args.raw_contact_threshold,
                    raw_vmax=raw_vmax,
                    strength_vmax=strength_vmax,
                    show_patch_labels=args.show_patch_labels,
                    patch_label_fontsize=args.patch_label_fontsize,
                    dpi=args.dpi,
                )
                panel_data = {
                    "episode_index": episode_index,
                    "frame_index": actual_frame_index,
                    "timestamp": float(timestamps[frame_position]),
                    "finger_names": FINGER_NAMES,
                    "target_strength": arrays["target_strength"][frame_position].tolist(),
                    "target_contact": arrays["target_contact"][frame_position].tolist(),
                    "target_force_xyz": arrays["target_force"][frame_position].tolist(),
                    "predicted_strength": arrays["pred_strength"][frame_position].tolist(),
                    "predicted_contact_probability": arrays["pred_contact"][frame_position].tolist(),
                    "predicted_distribution": arrays["pred_dist"][frame_position].tolist(),
                    "predicted_force_xyz": arrays["pred_force"][frame_position].tolist(),
                    "panels": panel_records,
                }
                panel_manifest = (
                    episode_dir / "panels" / f"frame_{actual_frame_index:06d}" / "panel_manifest.json"
                )
                panel_manifest.write_text(json.dumps(panel_data, indent=2, ensure_ascii=False))

            rendered.append(
                {
                    "output_frame": output_frame,
                    "source_position": frame_position,
                    "frame_index": actual_frame_index,
                    "timestamp": float(timestamps[frame_position]),
                    "path": str(output_path),
                    "panels": panel_records,
                }
            )
            if output_frame == 0 or (output_frame + 1) % 50 == 0:
                print(f"  rendered {output_frame + 1} frames", flush=True)

        video_path = episode_dir / "patch_reconstruction.mp4"
        video_created = (
            args.make_video
            and args.frame_index is None
            and _make_video(frames_dir, video_path, args.fps / args.frame_stride)
        )
        metadata = {
            **selected_item,
            "num_source_frames": int(raw_tactile.shape[0]),
            "num_rendered_frames": len(rendered),
            "frame_stride": args.frame_stride,
            "raw_contact_threshold": args.raw_contact_threshold,
            "raw_vmax": raw_vmax,
            "normalized_strength_vmax": strength_vmax,
            "show_patch_labels": args.show_patch_labels,
            "patch_label_fontsize": args.patch_label_fontsize,
            "gt_color_value": "where(target_contact >= 0.5, max(target_strength, 0), 0)",
            "predicted_color_value": "where(predicted_contact_probability >= 0.5, max(predicted_strength, 0), 0)",
            "distribution_color_value": "softmax patch distribution in [0, 1]",
            "force_array_shape": "[frames, fingers, patches, xyz]",
            "force_array_keys": ["target_force", "pred_force"],
            "normalization": str(norm_path),
            "frames": rendered,
            "video": str(video_path) if video_created else None,
        }
        (episode_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    print(f"Saved patch reconstruction visualizations to {output_root}", flush=True)


if __name__ == "__main__":
    main()
