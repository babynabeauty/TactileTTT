#!/usr/bin/env python3
"""Evaluate the patch mean-force XHand tactile encoder pretraining checkpoint.

This evaluator is for the patch mean-force pretraining variants, whose only
decoder head predicts per-finger, per-patch mean 3D force:

  raw tactile [B, 5, 120, 3] -> finger token -> [B, 5 fingers, 5 patches, 3]

It reads local LeRobot parquet files directly, so it does not decode videos.

Example:
  env/.venv/bin/python scripts/eval_patch_mean_force_encoder.py \
    --repo-id data/taskall-2 \
    --params checkpoints/xhand_patch_mean_force_encoder_pretrain/<exp>/19999/params \
    --output-dir outputs/patch_mean_force_encoder_eval/taskall-2 \
    --max-frames 20000 \
    --batch-size 256
"""

from __future__ import annotations

import csv
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
from openpi.training import config as _config
from openpi.training.data_loader import _arrow_column_to_numpy


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str
    params: str
    output_dir: str = "outputs/patch_mean_force_encoder_eval"
    config_name: str = "xhand_patch_mean_force_encoder_pretrain"
    filter_path: str | None = None
    batch_size: int = 256
    max_frames: int | None = 20000
    frame_stride: int = 1
    seed: int = 42


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _resolve_params(path: str) -> pathlib.Path:
    params = pathlib.Path(path).expanduser()
    if params.name != "params" and (params / "params").exists():
        params = params / "params"
    return params.resolve()


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
    return np.stack(chunks, axis=1).astype(np.float32)


def _iter_tactile_batches(args: Args):
    repo = pathlib.Path(args.repo_id).expanduser().resolve()
    episode_indices = _episode_filter(args.filter_path)
    files = _episode_files(repo, episode_indices)
    logging.info("Reading %d parquet episodes from %s", len(files), repo)

    rng = np.random.default_rng(args.seed)
    frame_count = 0
    buffer = []
    for parquet_file in files:
        table = pq.read_table(parquet_file, columns=["observation.state"])
        states = _arrow_column_to_numpy(table["observation.state"]).astype(np.float32)
        if args.frame_stride > 1:
            states = states[:: args.frame_stride]
        tactile = _extract_raw_tactile_from_state(states)
        if args.max_frames is not None and frame_count + tactile.shape[0] > args.max_frames:
            remaining = args.max_frames - frame_count
            if remaining <= 0:
                break
            indices = np.sort(rng.choice(tactile.shape[0], size=remaining, replace=False))
            tactile = tactile[indices]
        frame_count += tactile.shape[0]

        for frame in tactile:
            buffer.append(frame)
            if len(buffer) == args.batch_size:
                yield np.stack(buffer, axis=0)
                buffer = []
        if args.max_frames is not None and frame_count >= args.max_frames:
            break
    if buffer:
        yield np.stack(buffer, axis=0)


def _load_model(config_name: str, params_path: pathlib.Path):
    train_config = _config.get_config(config_name)
    params = _model.restore_params(params_path, restore_type=np.ndarray)
    return train_config.model.load(params)


def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
    error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
    return jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)


def _pearson(x: jax.Array, y: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32).reshape(-1)
    y = y.astype(jnp.float32).reshape(-1)
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    return jnp.sum(x * y) / jnp.maximum(jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y)), 1e-6)


def _contact_stats(pred_contact: jax.Array, target_contact: jax.Array) -> dict[str, jax.Array]:
    tp = jnp.sum(jnp.logical_and(pred_contact, target_contact))
    fp = jnp.sum(jnp.logical_and(pred_contact, jnp.logical_not(target_contact)))
    fn = jnp.sum(jnp.logical_and(jnp.logical_not(pred_contact), target_contact))
    tn = jnp.sum(jnp.logical_and(jnp.logical_not(pred_contact), jnp.logical_not(target_contact)))
    precision = tp / jnp.maximum(tp + fp, 1.0)
    recall = tp / jnp.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / jnp.maximum(precision + recall, 1e-6)
    accuracy = (tp + tn) / jnp.maximum(tp + fp + fn + tn, 1.0)
    return {
        "contact_accuracy_from_mean_force": accuracy,
        "contact_precision_from_mean_force": precision,
        "contact_recall_from_mean_force": recall,
        "contact_f1_from_mean_force": f1,
    }


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(jnp.float32)
    return jnp.sum(values.astype(jnp.float32) * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _eval_batch(model, tactile: jax.Array) -> dict[str, jax.Array]:
    effort = tactile[:, None, ...].astype(jnp.float32)
    times = jnp.zeros((1,), dtype=jnp.float32)
    tokens = model.patch_encoder._encode_steps(effort, times, future=False, include_temporal=False)

    pred_patch_force = model.patch_force_mean_head(tokens)
    pred_patch_force = einops.rearrange(
        pred_patch_force,
        "b t f (r c) -> b t f r c",
        r=model.num_patches,
        c=model.dim_per_point,
    ).astype(jnp.float32)
    target_patch_force = model.patch_encoder.patch_mean_force_targets(effort).astype(jnp.float32)

    pred_mag = jnp.linalg.norm(pred_patch_force, axis=-1)
    target_mag = jnp.linalg.norm(target_patch_force, axis=-1)
    threshold = jnp.asarray(model.contact_threshold, dtype=jnp.float32)
    pred_contact = pred_mag > threshold
    target_contact = target_mag > threshold
    active = target_contact.astype(jnp.float32)

    target_active_mag = target_mag * active
    target_distribution = target_active_mag / jnp.maximum(
        jnp.sum(target_active_mag, axis=-1, keepdims=True),
        1e-6,
    )
    pred_distribution = pred_mag / jnp.maximum(
        jnp.sum(pred_mag, axis=-1, keepdims=True),
        1e-6,
    )
    has_distribution = jnp.sum(target_active_mag, axis=-1) > 1e-6
    distribution_l1 = jnp.sum(jnp.abs(pred_distribution - target_distribution), axis=-1)
    distribution_cross_entropy = -jnp.sum(
        target_distribution * jnp.log(jnp.maximum(pred_distribution, 1e-6)),
        axis=-1,
    )

    vector_error = jnp.linalg.norm(pred_patch_force - target_patch_force, axis=-1)
    cosine = jnp.sum(pred_patch_force * target_patch_force, axis=-1) / jnp.maximum(pred_mag * target_mag, 1e-6)

    metrics = {
        "force_mae": jnp.mean(jnp.abs(pred_patch_force - target_patch_force)),
        "force_smooth_l1": jnp.mean(_smooth_l1(pred_patch_force, target_patch_force)),
        "force_vector_l2": jnp.mean(vector_error),
        "magnitude_mae": jnp.mean(jnp.abs(pred_mag - target_mag)),
        "magnitude_pearson": _pearson(pred_mag, target_mag),
        "active_force_vector_l2": _masked_mean(vector_error, active),
        "active_magnitude_mae": _masked_mean(jnp.abs(pred_mag - target_mag), active),
        "active_vector_cosine": _masked_mean(cosine, active),
        "active_distribution_l1": _masked_mean(distribution_l1, has_distribution),
        "active_distribution_cross_entropy": _masked_mean(distribution_cross_entropy, has_distribution),
        "pred_contact_ratio": jnp.mean(pred_contact.astype(jnp.float32)),
        "target_contact_ratio": jnp.mean(target_contact.astype(jnp.float32)),
        "pred_magnitude_mean": jnp.mean(pred_mag),
        "target_magnitude_mean": jnp.mean(target_mag),
        **_contact_stats(pred_contact, target_contact),
    }

    for finger_idx, finger_name in enumerate(FINGER_NAMES):
        finger_pred_mag = pred_mag[:, :, finger_idx]
        finger_target_mag = target_mag[:, :, finger_idx]
        finger_active = target_contact[:, :, finger_idx].astype(jnp.float32)
        finger_pred_contact = pred_contact[:, :, finger_idx]
        finger_target_contact = target_contact[:, :, finger_idx]
        finger_contact = _contact_stats(finger_pred_contact, finger_target_contact)
        metrics[f"finger/{finger_name}/magnitude_mae"] = jnp.mean(jnp.abs(finger_pred_mag - finger_target_mag))
        metrics[f"finger/{finger_name}/magnitude_pearson"] = _pearson(finger_pred_mag, finger_target_mag)
        metrics[f"finger/{finger_name}/active_magnitude_mae"] = _masked_mean(
            jnp.abs(finger_pred_mag - finger_target_mag),
            finger_active,
        )
        metrics[f"finger/{finger_name}/contact_f1"] = finger_contact["contact_f1_from_mean_force"]
        metrics[f"finger/{finger_name}/contact_accuracy"] = finger_contact["contact_accuracy_from_mean_force"]
    return metrics


def _write_outputs(output_dir: pathlib.Path, args: Args, params_path: pathlib.Path, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    args_dict = dataclasses.asdict(args)
    args_dict["resolved_params"] = str(params_path)
    (output_dir / "args.json").write_text(json.dumps(args_dict, indent=2, ensure_ascii=False))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (output_dir / "metrics_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(metrics):
            writer.writerow({"metric": key, "value": metrics[key]})


def _plot_outputs(output_dir: pathlib.Path, metrics: dict[str, float]) -> None:
    mae = [metrics.get(f"finger/{name}/magnitude_mae", np.nan) for name in FINGER_NAMES]
    corr = [metrics.get(f"finger/{name}/magnitude_pearson", np.nan) for name in FINGER_NAMES]
    f1 = [metrics.get(f"finger/{name}/contact_f1", np.nan) for name in FINGER_NAMES]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), dpi=180)
    axes[0].bar(FINGER_NAMES, mae, color="#d07c2c")
    axes[0].set_ylabel("magnitude MAE")
    axes[0].set_title("Patch Force Error")

    axes[1].bar(FINGER_NAMES, corr, color="#2f6f9f")
    axes[1].set_ylim(-1, 1)
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Magnitude Correlation")

    axes[2].bar(FINGER_NAMES, f1, color="#5f8f3a")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("contact F1")
    axes[2].set_title("Contact from Force")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_finger_patch_mean_force_metrics.png")
    plt.close(fig)


def main(args: Args) -> None:
    init_logging()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")

    params_path = _resolve_params(args.params)
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    logging.info("Loading patch mean-force encoder checkpoint: %s", params_path)
    model = _load_model(args.config_name, params_path)
    model.eval()

    peval = jax.jit(lambda batch: _eval_batch(model, batch))
    sums: dict[str, float] = {}
    total_frames = 0
    batches = 0
    for tactile_np in _iter_tactile_batches(args):
        metrics = peval(jnp.asarray(tactile_np, dtype=jnp.float32))
        metrics_np = jax.device_get(metrics)
        batch_size = tactile_np.shape[0]
        for key, value in metrics_np.items():
            sums[key] = sums.get(key, 0.0) + float(np.asarray(value)) * batch_size
        total_frames += batch_size
        batches += 1
        if batches == 1 or batches % 20 == 0:
            logging.info("Processed %d frames.", total_frames)

    if total_frames == 0:
        raise RuntimeError("No frames were evaluated.")
    metrics = {key: value / total_frames for key, value in sorted(sums.items())}
    metrics["eval_frames"] = float(total_frames)
    metrics["eval_batches"] = float(batches)
    _write_outputs(output_dir, args, params_path, metrics)
    _plot_outputs(output_dir, metrics)
    logging.info("Saved patch mean-force encoder metrics to %s", output_dir)
    logging.info(
        "Summary: force_mae=%.4f magnitude_mae=%.4f magnitude_pearson=%.4f active_vector_cosine=%.4f",
        metrics["force_mae"],
        metrics["magnitude_mae"],
        metrics["magnitude_pearson"],
        metrics["active_vector_cosine"],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
