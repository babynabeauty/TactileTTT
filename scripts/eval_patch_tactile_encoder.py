#!/usr/bin/env python3
"""Evaluate pretrained XHand patch tactile encoder decoder heads.

This is a fast local-parquet evaluator for the Stage-1
Full-head and strength/contact/zero Stage-1 checkpoints are supported. The
evaluator reads raw tactile force
from `observation.state`, runs the patch-informed encoder plus its three
pretraining heads, and reports whether one finger token preserves local
5-patch contact structure.

Example:
  env/.venv/bin/python scripts/eval_patch_tactile_encoder.py \
    --repo-id data/task12345-2 \
    --params checkpoints/xhand_patch_tactile_encoder_pretrain/<exp>/<step>/params \
    --filter-path outputs/episode_splits/task12345-2_recursive_revision/val_episodes.json \
    --output-dir outputs/patch_encoder_eval/task12345_2 \
    --max-frames 20000 \
    --batch-size 256
"""

from __future__ import annotations

import csv
import collections.abc
import dataclasses
import json
import logging
import pathlib

import einops
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tyro

from openpi.models import model as _model
from openpi.policies import xhand_policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training.data_loader import _arrow_column_to_numpy


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str
    params: str
    output_dir: str = "outputs/patch_encoder_eval"
    config_name: str = "xhand_patch_tactile_encoder_pretrain"
    assets_dir: str = "assets/pi0_xhand_tactile_structured_raw_dual_ae"
    asset_id: str | None = None
    filter_path: str | None = None
    batch_size: int = 256
    max_frames: int | None = 20000
    frame_stride: int = 1
    seed: int = 42


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _episode_filter(path: str | None) -> set[int] | None:
    if path is None:
        return None
    with pathlib.Path(path).expanduser().open() as f:
        data = json.load(f)
    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict):
        episodes = (
            data.get("episodes")
            or data.get("episode_indices")
            or data.get("episode_index")
            or data.get("val")
            or data.get("train")
        )
    else:
        raise ValueError(f"Unsupported episode filter format: {path}")
    if episodes is None:
        raise ValueError(f"Could not find episode list in {path}")
    return {int(ep) for ep in episodes}


def _episode_files(repo: pathlib.Path, episode_indices: set[int] | None) -> list[pathlib.Path]:
    info_path = repo / "meta" / "info.json"
    with info_path.open() as f:
        info = json.load(f)
    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    glob_pattern = data_pattern.replace("{episode_chunk:03d}", "*").replace("{episode_index:06d}", "*")
    files = sorted(repo.glob(glob_pattern))
    if not files:
        files = sorted((repo / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {repo}")
    if episode_indices is None:
        return files
    selected = []
    for path in files:
        try:
            episode = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if episode in episode_indices:
            selected.append(path)
    return selected


def _extract_raw_tactile_from_state(states: np.ndarray) -> np.ndarray:
    chunks = []
    for sensor_id in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + sensor_id * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        chunks.append(states[:, start:end].reshape(states.shape[0], xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    return np.stack(chunks, axis=1).astype(np.float32)  # [N, 5, 120, 3]


def _normalize_tactile(tactile: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Match transforms.Normalize for common raw-tactile norm-stat layouts."""
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
    raise ValueError(
        f"Cannot apply effort norm stats of length {mean.size} to tactile shape {feature_shape}."
    )


def _load_effort_norm(args: Args) -> tuple[np.ndarray, np.ndarray]:
    asset_id = args.asset_id or pathlib.Path(args.repo_id.rstrip("/")).name
    norm_dir = pathlib.Path(args.assets_dir).expanduser().resolve() / asset_id
    norm_stats = _normalize.load(norm_dir)
    if "effort" not in norm_stats:
        raise KeyError(f"No 'effort' norm stats found in {norm_dir / 'norm_stats.json'}")
    effort_stats = norm_stats["effort"]
    logging.info("Using effort normalization from %s", norm_dir / "norm_stats.json")
    return np.asarray(effort_stats.mean), np.asarray(effort_stats.std)


def _iter_tactile_batches(args: Args):
    repo = pathlib.Path(args.repo_id).expanduser().resolve()
    episode_indices = _episode_filter(args.filter_path)
    files = _episode_files(repo, episode_indices)
    effort_mean, effort_std = _load_effort_norm(args)
    logging.info("Reading %d parquet episodes from %s", len(files), repo)

    sampled_positions: dict[pathlib.Path, np.ndarray] | None = None
    if args.max_frames is not None:
        frame_counts = np.asarray(
            [
                (pq.ParquetFile(path).metadata.num_rows + args.frame_stride - 1)
                // args.frame_stride
                for path in files
            ],
            dtype=np.int64,
        )
        total_available = int(np.sum(frame_counts))
        sample_count = min(args.max_frames, total_available)
        if sample_count < total_available:
            rng = np.random.default_rng(args.seed)
            sampled_global = np.sort(rng.choice(total_available, size=sample_count, replace=False))
            offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(frame_counts)])
            sampled_positions = {}
            for file_index, path in enumerate(files):
                in_file = sampled_global[
                    (sampled_global >= offsets[file_index])
                    & (sampled_global < offsets[file_index + 1])
                ]
                if in_file.size:
                    sampled_positions[path] = in_file - offsets[file_index]
            logging.info(
                "Uniformly sampled %d of %d held-out frames across %d episodes.",
                sample_count,
                total_available,
                len(files),
            )

    buffer = []
    for parquet_file in files:
        if sampled_positions is not None and parquet_file not in sampled_positions:
            continue
        table = pq.read_table(parquet_file, columns=["observation.state"])
        states = _arrow_column_to_numpy(table["observation.state"]).astype(np.float32)
        states = states[:: args.frame_stride]
        if sampled_positions is not None:
            states = states[sampled_positions[parquet_file]]
        tactile = _extract_raw_tactile_from_state(states)
        tactile = _normalize_tactile(tactile, effort_mean, effort_std)
        for frame in tactile:
            buffer.append(frame)
            if len(buffer) == args.batch_size:
                yield np.stack(buffer, axis=0)
                buffer = []
    if buffer:
        yield np.stack(buffer, axis=0)


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


def _load_model(config_name: str, params_path: str):
    train_config = _config.get_config(config_name)
    params = _model.restore_params(pathlib.Path(params_path).expanduser(), restore_type=np.ndarray)
    expected_type = getattr(train_config.model, "pretrain_tactile_encoder", None)
    checkpoint_type = _infer_checkpoint_encoder_type(params)
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
    return train_config.model.load(params)


def _sigmoid_bce_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    logits = logits.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
    error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
    return jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)


def _pearson(x: jax.Array, y: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32).reshape(-1)
    y = y.astype(jnp.float32).reshape(-1)
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    return jnp.sum(x * y) / jnp.maximum(jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y)), 1e-6)


def _contact_counts(pred_prob: jax.Array, target: jax.Array) -> dict[str, jax.Array]:
    pred = pred_prob >= 0.5
    true = target >= 0.5
    tp = jnp.sum(jnp.logical_and(pred, true))
    fp = jnp.sum(jnp.logical_and(pred, jnp.logical_not(true)))
    fn = jnp.sum(jnp.logical_and(jnp.logical_not(pred), true))
    tn = jnp.sum(jnp.logical_and(jnp.logical_not(pred), jnp.logical_not(true)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _contact_stats(pred_prob: jax.Array, target: jax.Array) -> dict[str, jax.Array]:
    counts = _contact_counts(pred_prob, target)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / jnp.maximum(tp + fp, 1.0)
    recall = tp / jnp.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / jnp.maximum(precision + recall, 1e-6)
    accuracy = (tp + tn) / jnp.maximum(tp + fp + fn + tn, 1.0)
    return {
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": f1,
        "contact_accuracy": accuracy,
    }


def _pearson_sufficient_stats(x: jax.Array, y: jax.Array) -> dict[str, jax.Array]:
    x = x.astype(jnp.float32).reshape(-1)
    y = y.astype(jnp.float32).reshape(-1)
    return {
        "n": jnp.asarray(x.size, dtype=jnp.float32),
        "sum_x": jnp.sum(x),
        "sum_y": jnp.sum(y),
        "sum_x2": jnp.sum(jnp.square(x)),
        "sum_y2": jnp.sum(jnp.square(y)),
        "sum_xy": jnp.sum(x * y),
    }


def _add_global_aggregate_stats(
    metrics: dict[str, jax.Array],
    prefix: str,
    pred_contact: jax.Array,
    target_contact: jax.Array,
    pred_strength: jax.Array,
    target_strength: jax.Array,
) -> None:
    for name, value in _contact_counts(pred_contact, target_contact).items():
        metrics[f"_aggregate/{prefix}/contact/{name}"] = value
    for name, value in _pearson_sufficient_stats(pred_strength, target_strength).items():
        metrics[f"_aggregate/{prefix}/strength/{name}"] = value


def _add_force_aggregate_stats(
    metrics: dict[str, jax.Array],
    prefix: str,
    pred_force: jax.Array,
    target_force: jax.Array,
    target_contact: jax.Array,
) -> None:
    active = target_contact >= 0.5
    inactive = jnp.logical_not(active)
    vector_error = jnp.linalg.norm(pred_force - target_force, axis=-1)
    pred_magnitude = jnp.linalg.norm(pred_force, axis=-1)
    target_magnitude = jnp.linalg.norm(target_force, axis=-1)
    direction_valid = jnp.logical_and(active, target_magnitude > 1e-6)
    cosine = jnp.sum(pred_force * target_force, axis=-1) / jnp.maximum(
        pred_magnitude * target_magnitude,
        1e-6,
    )
    abs_axis_error = jnp.abs(pred_force - target_force)
    base = f"_aggregate/{prefix}/force"
    metrics[f"{base}/count"] = jnp.asarray(vector_error.size, dtype=jnp.float32)
    metrics[f"{base}/vector_l2_sum"] = jnp.sum(vector_error)
    metrics[f"{base}/active_count"] = jnp.sum(active)
    metrics[f"{base}/active_vector_l2_sum"] = jnp.sum(jnp.where(active, vector_error, 0.0))
    metrics[f"{base}/active_pred_magnitude_sum"] = jnp.sum(
        jnp.where(active, pred_magnitude, 0.0)
    )
    metrics[f"{base}/active_target_magnitude_sum"] = jnp.sum(
        jnp.where(active, target_magnitude, 0.0)
    )
    metrics[f"{base}/direction_count"] = jnp.sum(direction_valid)
    metrics[f"{base}/direction_cosine_sum"] = jnp.sum(jnp.where(direction_valid, cosine, 0.0))
    metrics[f"{base}/inactive_count"] = jnp.sum(inactive)
    metrics[f"{base}/inactive_magnitude_sum"] = jnp.sum(jnp.where(inactive, pred_magnitude, 0.0))
    for axis, axis_name in enumerate(("x", "y", "z")):
        metrics[f"{base}/{axis_name}_abs_sum"] = jnp.sum(abs_axis_error[..., axis])
        metrics[f"{base}/active_{axis_name}_abs_sum"] = jnp.sum(
            jnp.where(active, abs_axis_error[..., axis], 0.0)
        )


def _eval_batch(model, tactile: jax.Array) -> dict[str, jax.Array]:
    effort = tactile[:, None, ...].astype(jnp.float32)  # [B,1,5,120,3]
    times = jnp.zeros((1,), dtype=jnp.float32)
    tokens = model.patch_encoder._encode_steps(effort, times, future=False, include_temporal=False)

    target_dist, target_summary, target_contact = model.patch_encoder.patch_reconstruction_targets(effort)
    target_dist = target_dist.astype(jnp.float32)
    target_summary = target_summary.astype(jnp.float32)
    target_contact = target_contact.astype(jnp.float32)
    target_force = target_summary[..., : model.dim_per_point] * target_contact[..., None]
    target_strength = jnp.linalg.norm(target_force, axis=-1)
    eps = 1e-6

    if hasattr(model, "patch_strength_head"):
        # Legacy scalar-strength objective. It has no 3D-force prediction, but
        # remains supported by this evaluator for backwards compatibility.
        target_strength = target_summary[..., -1]
        pred_strength = jax.nn.softplus(model.patch_strength_head(tokens).astype(jnp.float32))
        threshold = jnp.asarray(model.contact_threshold, dtype=jnp.float32)
        temperature = jnp.asarray(max(model.contact_temperature, 1e-6), dtype=jnp.float32)
        contact_logits = (pred_strength - threshold) / temperature
        pred_contact = jax.nn.sigmoid(contact_logits)
        contact_bce = jnp.mean(_sigmoid_bce_with_logits(contact_logits, target_contact))
        contact = _contact_stats(pred_contact, target_contact)
        strength_mae = jnp.mean(jnp.abs(pred_strength - target_strength))
        strength_smooth_l1 = jnp.mean(_smooth_l1(pred_strength, target_strength))
        strength_corr = _pearson(pred_strength, target_strength)
        inactive = 1.0 - target_contact
        inactive_denom = jnp.maximum(jnp.sum(inactive), 1.0)
        zero_patch_mean = jnp.sum(pred_strength * inactive) / inactive_denom
        metrics = {
            "contact_bce": contact_bce,
            "strength_mae": strength_mae,
            "strength_smooth_l1": strength_smooth_l1,
            "strength_pearson": strength_corr,
            "zero_patch_strength_mean": zero_patch_mean,
            "pred_contact_ratio": jnp.mean((pred_contact >= 0.5).astype(jnp.float32)),
            "target_contact_ratio": jnp.mean(target_contact),
            "pred_strength_mean": jnp.mean(pred_strength),
            "target_strength_mean": jnp.mean(target_strength),
            **contact,
        }
    elif hasattr(model, "patch_force_mean_head"):
        # Mean-force family: contact and distribution are derived from the
        # single 3D-force decoder rather than predicted by independent heads.
        pred_force = einops.rearrange(
            model.patch_force_mean_head(tokens),
            "b t f (r c) -> b t f r c",
            r=model.num_patches,
            c=model.dim_per_point,
        ).astype(jnp.float32)
        pred_strength = jnp.linalg.norm(pred_force, axis=-1)
        threshold = jnp.asarray(model.contact_threshold, dtype=jnp.float32)
        temperature = jnp.asarray(max(model.contact_temperature, 1e-6), dtype=jnp.float32)
        contact_logits = (pred_strength - threshold) / temperature
        pred_contact = jax.nn.sigmoid(contact_logits)
        pred_dist = pred_strength / jnp.maximum(jnp.sum(pred_strength, axis=-1, keepdims=True), eps)
        summary_smooth_l1 = jnp.mean(_smooth_l1(pred_force, target_force))
    else:
        # Full-head and compact force-three-head families both expose three
        # independent heads. Their middle head always starts with mean XYZ.
        dist_logits = model.patch_distribution_head(tokens)
        pred_dist = jax.nn.softmax(dist_logits.astype(jnp.float32), axis=-1)

        summary_pred = model.patch_summary_head(tokens)
        summary_pred = einops.rearrange(
            summary_pred,
            "b t f (r c) -> b t f r c",
            r=model.num_patches,
            c=model.summary_dim,
        ).astype(jnp.float32)
        pred_force = summary_pred[..., : model.dim_per_point]
        contact_logits = model.patch_contact_head(tokens).astype(jnp.float32)
        pred_contact = jax.nn.sigmoid(contact_logits)
        pred_strength = jnp.linalg.norm(pred_force, axis=-1)
        target_summary_for_loss = (
            target_force
            if getattr(model, "force_only_summary", False)
            else target_summary
        )
        summary_smooth_l1 = jnp.mean(_smooth_l1(summary_pred, target_summary_for_loss))

    if not hasattr(model, "patch_strength_head"):
        pred_force_dist = pred_strength / jnp.maximum(
            jnp.sum(pred_strength, axis=-1, keepdims=True),
            eps,
        )
        target_force_sum = jnp.sum(target_strength, axis=-1, keepdims=True)
        uniform = jnp.full_like(target_strength, 1.0 / float(model.num_patches))
        target_force_dist = jnp.where(
            target_force_sum > eps,
            target_strength / jnp.maximum(target_force_sum, eps),
            uniform,
        )
        dist_ce = -jnp.mean(jnp.sum(target_dist * jnp.log(jnp.maximum(pred_dist, eps)), axis=-1))
        dist_kl = jnp.mean(
            jnp.sum(
                target_dist
                * (jnp.log(jnp.maximum(target_dist, eps)) - jnp.log(jnp.maximum(pred_dist, eps))),
                axis=-1,
            )
        )
        force_dist_kl_per_item = jnp.sum(
            target_force_dist
            * (
                jnp.log(jnp.maximum(target_force_dist, eps))
                - jnp.log(jnp.maximum(pred_force_dist, eps))
            ),
            axis=-1,
        )
        has_force_distribution = (target_force_sum[..., 0] > eps).astype(jnp.float32)
        force_dist_kl = (
            jnp.sum(force_dist_kl_per_item * has_force_distribution)
            / jnp.maximum(jnp.sum(has_force_distribution), 1.0)
        )
        contact_bce = jnp.mean(_sigmoid_bce_with_logits(contact_logits, target_contact))
        contact = _contact_stats(pred_contact, target_contact)
        strength_mae = jnp.mean(jnp.abs(pred_strength - target_strength))
        strength_smooth_l1 = jnp.mean(_smooth_l1(pred_strength, target_strength))
        strength_corr = _pearson(pred_strength, target_strength)
        metrics = {
            "distribution_ce": dist_ce,
            "distribution_kl": dist_kl,
            "force_distribution_kl": force_dist_kl,
            "contact_bce": contact_bce,
            "strength_mae": strength_mae,
            "strength_smooth_l1": strength_smooth_l1,
            "summary_smooth_l1": summary_smooth_l1,
            "strength_pearson": strength_corr,
            "pred_contact_ratio": jnp.mean((pred_contact >= 0.5).astype(jnp.float32)),
            "target_contact_ratio": jnp.mean(target_contact),
            "pred_strength_mean": jnp.mean(pred_strength),
            "target_strength_mean": jnp.mean(target_strength),
            **contact,
        }
        metrics["_aggregate/global/distribution/force_kl_sum"] = jnp.sum(
            force_dist_kl_per_item * has_force_distribution
        )
        metrics["_aggregate/global/distribution/force_count"] = jnp.sum(has_force_distribution)
        _add_force_aggregate_stats(
            metrics,
            "global",
            pred_force,
            target_force,
            target_contact,
        )

    for finger_idx, finger_name in enumerate(FINGER_NAMES):
        finger_contact = _contact_stats(pred_contact[:, :, finger_idx], target_contact[:, :, finger_idx])
        metrics[f"finger/{finger_name}/contact_f1"] = finger_contact["contact_f1"]
        metrics[f"finger/{finger_name}/contact_accuracy"] = finger_contact["contact_accuracy"]
        metrics[f"finger/{finger_name}/strength_mae"] = jnp.mean(
            jnp.abs(pred_strength[:, :, finger_idx] - target_strength[:, :, finger_idx])
        )
        metrics[f"finger/{finger_name}/strength_pearson"] = _pearson(
            pred_strength[:, :, finger_idx],
            target_strength[:, :, finger_idx],
        )
        _add_global_aggregate_stats(
            metrics,
            f"finger/{finger_name}",
            pred_contact[:, :, finger_idx],
            target_contact[:, :, finger_idx],
            pred_strength[:, :, finger_idx],
            target_strength[:, :, finger_idx],
        )
        if not hasattr(model, "patch_strength_head"):
            _add_force_aggregate_stats(
                metrics,
                f"finger/{finger_name}",
                pred_force[:, :, finger_idx],
                target_force[:, :, finger_idx],
                target_contact[:, :, finger_idx],
            )
    _add_global_aggregate_stats(
        metrics,
        "global",
        pred_contact,
        target_contact,
        pred_strength,
        target_strength,
    )
    return metrics


def _finalize_contact_metrics(aggregate: dict[str, float], prefix: str) -> dict[str, float]:
    base = f"_aggregate/{prefix}/contact"
    tp = aggregate[f"{base}/tp"]
    fp = aggregate[f"{base}/fp"]
    fn = aggregate[f"{base}/fn"]
    tn = aggregate[f"{base}/tn"]
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "contact_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1.0),
    }


def _finalize_pearson(aggregate: dict[str, float], prefix: str) -> float:
    base = f"_aggregate/{prefix}/strength"
    n = aggregate[f"{base}/n"]
    sum_x = aggregate[f"{base}/sum_x"]
    sum_y = aggregate[f"{base}/sum_y"]
    covariance = aggregate[f"{base}/sum_xy"] - sum_x * sum_y / max(n, 1.0)
    variance_x = max(aggregate[f"{base}/sum_x2"] - sum_x * sum_x / max(n, 1.0), 0.0)
    variance_y = max(aggregate[f"{base}/sum_y2"] - sum_y * sum_y / max(n, 1.0), 0.0)
    return covariance / max(np.sqrt(variance_x * variance_y), 1e-12)


def _finalize_force_metrics(aggregate: dict[str, float], prefix: str) -> dict[str, float]:
    base = f"_aggregate/{prefix}/force"
    count = max(aggregate[f"{base}/count"], 1.0)
    active_count = max(aggregate[f"{base}/active_count"], 1.0)
    direction_count = max(aggregate[f"{base}/direction_count"], 1.0)
    inactive_count = max(aggregate[f"{base}/inactive_count"], 1.0)
    active_pred_magnitude_mean = aggregate[f"{base}/active_pred_magnitude_sum"] / active_count
    active_target_magnitude_mean = aggregate[f"{base}/active_target_magnitude_sum"] / active_count
    return {
        "force_vector_l2": aggregate[f"{base}/vector_l2_sum"] / count,
        "active_force_vector_l2": aggregate[f"{base}/active_vector_l2_sum"] / active_count,
        "active_pred_force_magnitude_mean": active_pred_magnitude_mean,
        "active_target_force_magnitude_mean": active_target_magnitude_mean,
        "active_force_magnitude_ratio": (
            active_pred_magnitude_mean / max(active_target_magnitude_mean, 1e-12)
        ),
        "active_force_direction_cosine": aggregate[f"{base}/direction_cosine_sum"] / direction_count,
        "force_x_mae": aggregate[f"{base}/x_abs_sum"] / count,
        "force_y_mae": aggregate[f"{base}/y_abs_sum"] / count,
        "force_z_mae": aggregate[f"{base}/z_abs_sum"] / count,
        "active_force_x_mae": aggregate[f"{base}/active_x_abs_sum"] / active_count,
        "active_force_y_mae": aggregate[f"{base}/active_y_abs_sum"] / active_count,
        "active_force_z_mae": aggregate[f"{base}/active_z_abs_sum"] / active_count,
        "inactive_force_magnitude_mean": aggregate[f"{base}/inactive_magnitude_sum"] / inactive_count,
    }


def _write_outputs(output_dir: pathlib.Path, args: Args, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, ensure_ascii=False))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (output_dir / "metrics_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(metrics):
            writer.writerow({"metric": key, "value": metrics[key]})


def _plot_outputs(output_dir: pathlib.Path, metrics: dict[str, float]) -> None:
    f1 = [metrics.get(f"finger/{name}/contact_f1", np.nan) for name in FINGER_NAMES]
    mae = [metrics.get(f"finger/{name}/strength_mae", np.nan) for name in FINGER_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=180)
    axes[0].bar(FINGER_NAMES, f1, color="#2f6f9f")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("contact F1")
    axes[0].set_title("Patch Contact")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(FINGER_NAMES, mae, color="#d07c2c")
    axes[1].set_ylabel("strength MAE")
    axes[1].set_title("Patch Strength")
    axes[1].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_finger_patch_metrics.png")
    plt.close(fig)


def main(args: Args) -> None:
    init_logging()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided.")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    logging.info("Loading patch encoder checkpoint: %s", args.params)
    model = _load_model(args.config_name, args.params)
    model.eval()

    peval = jax.jit(lambda batch: _eval_batch(model, batch))
    sums: dict[str, float] = {}
    aggregate: dict[str, float] = {}
    total_frames = 0
    batches = 0
    for tactile_np in _iter_tactile_batches(args):
        metrics = peval(jnp.asarray(tactile_np, dtype=jnp.float32))
        metrics_np = jax.device_get(metrics)
        batch_size = tactile_np.shape[0]
        for key, value in metrics_np.items():
            scalar = float(np.asarray(value))
            if key.startswith("_aggregate/"):
                aggregate[key] = aggregate.get(key, 0.0) + scalar
            else:
                sums[key] = sums.get(key, 0.0) + scalar * batch_size
        total_frames += batch_size
        batches += 1
        if batches == 1 or batches % 20 == 0:
            logging.info("Processed %d frames.", total_frames)

    if total_frames == 0:
        raise RuntimeError("No frames were evaluated.")
    metrics = {key: value / total_frames for key, value in sorted(sums.items())}
    metrics.update(_finalize_contact_metrics(aggregate, "global"))
    metrics["strength_pearson"] = _finalize_pearson(aggregate, "global")
    force_dist_count_key = "_aggregate/global/distribution/force_count"
    if force_dist_count_key in aggregate:
        metrics["force_distribution_kl"] = (
            aggregate["_aggregate/global/distribution/force_kl_sum"]
            / max(aggregate[force_dist_count_key], 1.0)
        )
    if "_aggregate/global/force/count" in aggregate:
        metrics.update(_finalize_force_metrics(aggregate, "global"))
    for finger_name in FINGER_NAMES:
        finger_contact = _finalize_contact_metrics(aggregate, f"finger/{finger_name}")
        for metric_name, value in finger_contact.items():
            metrics[f"finger/{finger_name}/{metric_name}"] = value
        metrics[f"finger/{finger_name}/strength_pearson"] = _finalize_pearson(
            aggregate, f"finger/{finger_name}"
        )
        if f"_aggregate/finger/{finger_name}/force/count" in aggregate:
            for metric_name, value in _finalize_force_metrics(
                aggregate,
                f"finger/{finger_name}",
            ).items():
                metrics[f"finger/{finger_name}/{metric_name}"] = value
    metrics["eval_frames"] = float(total_frames)
    metrics["eval_batches"] = float(batches)
    _write_outputs(output_dir, args, metrics)
    _plot_outputs(output_dir, metrics)
    logging.info("Saved patch encoder metrics to %s", output_dir)
    summary = (
        "Summary: contact_f1=%.4f precision=%.4f recall=%.4f "
        "strength_mae=%.4f strength_pearson=%.4f"
    )
    logging.info(
        summary,
        metrics["contact_f1"],
        metrics["contact_precision"],
        metrics["contact_recall"],
        metrics["strength_mae"],
        metrics["strength_pearson"],
    )
    if "distribution_kl" in metrics:
        logging.info("Distribution KL: %.4f", metrics["distribution_kl"])
    if "active_force_vector_l2" in metrics:
        logging.info(
            "3D force: active_vector_l2=%.4f direction_cosine=%.4f "
            "active_magnitude=(pred=%.4f, target=%.4f, ratio=%.4f) "
            "active_xyz_mae=(%.4f, %.4f, %.4f) inactive_magnitude=%.4f",
            metrics["active_force_vector_l2"],
            metrics["active_force_direction_cosine"],
            metrics["active_pred_force_magnitude_mean"],
            metrics["active_target_force_magnitude_mean"],
            metrics["active_force_magnitude_ratio"],
            metrics["active_force_x_mae"],
            metrics["active_force_y_mae"],
            metrics["active_force_z_mae"],
            metrics["inactive_force_magnitude_mean"],
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
