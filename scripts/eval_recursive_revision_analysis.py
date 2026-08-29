#!/usr/bin/env python3
"""Offline recursive revision analysis for cached-VLM async tactile foresight.

This script evaluates whether asynchronous tactile updates improve future
tactile latents and action chunks. It compares:

  one_shot:      run at offset 0 once and reuse the same result.
  fresh_reinfer: rerun with fresh tactile history, but learned future queries.
  retouch:       rerun with fresh tactile history and previous-prefix warm-start.

The Hindsight/Teacher AE is used only as an offline privileged reference for
latent cosine metrics. It is not used as an input to the Student branch.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import pathlib
from typing import Literal

import einops
from flax import traverse_util
import flax.nnx as nnx
import jax
from jax import ShapeDtypeStruct
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from openpi.models import model_tavla as _model
from openpi.models.pi0_tavla import make_attn_mask
from openpi.policies import xhand_policy
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders

EvalMode = Literal["one_shot", "action_only", "fresh_reinfer", "retouch"]
MODES: tuple[EvalMode, ...] = ("one_shot", "action_only", "fresh_reinfer", "retouch")
PHASES = ("overall", "pre_contact", "post_contact", "no_contact")


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str
    pretrained_params: str
    model_label: str | None = None
    modes: tuple[EvalMode, ...] = MODES

    repo_id: str | None = None
    asset_id: str | None = None
    assets_dir: str | None = None
    filter_path: str | None = None

    output_dir: str = "outputs/recursive_revision_analysis"
    batch_size: int = 4
    fsdp_devices: int = 1
    num_workers: int = 0
    # 0 evaluates one complete dataloader pass. Positive values cap the pass.
    max_batches: int = 0
    seed: int = 42
    num_steps: int = 10
    offsets: tuple[int, ...] = (0, 4, 8, 12)

    latent_action_condition: Literal["zero", "gt"] = "zero"

    # Contact onset is computed from raw (pre-normalization) XHand taxels over
    # each complete episode. A frame is contact-active when at least
    # contact_min_taxels exceed contact_threshold.
    split_by_contact: bool = True
    contact_threshold: float = 1.0
    contact_min_taxels: int = 1
    contact_min_consecutive_frames: int = 1


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _resolve_params_path(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text).expanduser().resolve()
    if (path / "params").is_dir():
        path = path / "params"
    if not path.exists():
        raise FileNotFoundError(f"Policy parameter path does not exist: {path}")
    return path


def _selected_episode_indices(filter_path: str | None) -> set[int] | None:
    if filter_path is None:
        return None
    with pathlib.Path(filter_path).expanduser().open() as f:
        value = json.load(f)
    if isinstance(value, list):
        episodes = value
    elif isinstance(value, dict):
        episodes = next(
            (
                value[key]
                for key in ("episodes", "episode_indices", "episode_index", "train", "val")
                if value.get(key) is not None
            ),
            None,
        )
    else:
        episodes = None
    if episodes is None:
        raise ValueError(f"Cannot read episode indices from filter file: {filter_path}")
    return {int(episode) for episode in episodes}


def _extract_raw_tactile(states: np.ndarray) -> np.ndarray:
    required_dim = (
        xhand_policy.TACTILE_BLOCK_START
        + xhand_policy.TACTILE_SENSOR_COUNT * xhand_policy.TACTILE_BLOCK_SIZE
    )
    if states.ndim != 2 or states.shape[-1] < required_dim:
        raise ValueError(
            "Contact-onset analysis requires full XHand observation.state; "
            f"expected [T,D] with D >= {required_dim}, got {states.shape}."
        )
    fingers = []
    for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + finger * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        fingers.append(states[:, start:end].reshape(states.shape[0], xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    return np.stack(fingers, axis=1).astype(np.float32)


def _first_consecutive_true(mask: np.ndarray, length: int) -> int | None:
    length = max(int(length), 1)
    if mask.size < length:
        return None
    if length == 1:
        hits = np.flatnonzero(mask)
    else:
        hits = np.flatnonzero(np.convolve(mask.astype(np.int32), np.ones(length, dtype=np.int32), mode="valid") >= length)
    return int(hits[0]) if hits.size else None


def _compute_contact_onsets(args: Args) -> tuple[dict[int, int | None], dict[str, object]]:
    if args.repo_id is None:
        raise ValueError("--repo-id is required when --split-by-contact is enabled.")
    repo = pathlib.Path(args.repo_id).expanduser().resolve()
    parquet_files = sorted(repo.glob("data/**/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {repo / 'data'}")
    info_path = repo / "meta" / "info.json"
    fps = 15.0
    if info_path.exists():
        with info_path.open() as f:
            fps = float(json.load(f).get("fps", fps))

    selected = _selected_episode_indices(args.filter_path)
    onsets: dict[int, int | None] = {}
    episode_rows: list[dict[str, object]] = []
    for parquet_path in parquet_files:
        frame = pd.read_parquet(
            parquet_path,
            columns=["observation.state", "episode_index", "frame_index"],
        )
        states = np.asarray(frame["observation.state"].to_list(), dtype=np.float32)
        episode_indices = np.asarray(frame["episode_index"].to_list(), dtype=np.int64)
        frame_indices = np.asarray(frame["frame_index"].to_list(), dtype=np.int64)
        for episode_index in np.unique(episode_indices):
            episode = int(episode_index)
            if selected is not None and episode not in selected:
                continue
            rows = episode_indices == episode_index
            order = np.argsort(frame_indices[rows])
            episode_frames = frame_indices[rows][order]
            tactile = _extract_raw_tactile(states[rows][order])
            magnitude = np.linalg.norm(tactile, axis=-1)
            active_taxels = np.sum(magnitude > float(args.contact_threshold), axis=(1, 2))
            onset_row = _first_consecutive_true(
                active_taxels >= int(args.contact_min_taxels),
                args.contact_min_consecutive_frames,
            )
            onset_frame = None if onset_row is None else int(episode_frames[onset_row])
            onsets[episode] = onset_frame
            episode_rows.append(
                {
                    "episode_index": episode,
                    "first_contact_frame": onset_frame,
                    "first_contact_time_sec": None if onset_frame is None else onset_frame / fps,
                    "num_frames": int(episode_frames.size),
                    "max_active_taxels": int(np.max(active_taxels, initial=0)),
                }
            )

    summary: dict[str, object] = {
        "repo_id": str(repo),
        "fps": fps,
        "contact_threshold": float(args.contact_threshold),
        "contact_min_taxels": int(args.contact_min_taxels),
        "contact_min_consecutive_frames": int(args.contact_min_consecutive_frames),
        "num_episodes": len(onsets),
        "num_episodes_with_contact": sum(onset is not None for onset in onsets.values()),
        "episodes": sorted(episode_rows, key=lambda row: int(row["episode_index"])),
    }
    return onsets, summary


def _path_key(key: tuple[object, ...]) -> str:
    return "/".join(map(str, key))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    flat_shape = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)

    filtered = {}
    failed = []
    for key, initialized_value in flat_shape.items():
        loaded_value = flat_loaded.get(key)
        reason = None
        if loaded_value is None:
            reason = "missing in checkpoint"
        elif isinstance(loaded_value, ShapeDtypeStruct):
            reason = "is ShapeDtypeStruct"
        elif not hasattr(loaded_value, "shape"):
            reason = "has no shape"
        elif loaded_value.shape != initialized_value.shape:
            reason = f"shape mismatch ckpt={loaded_value.shape}, model={initialized_value.shape}"

        if reason is None:
            filtered[key] = loaded_value
        else:
            filtered[key] = initialized_value
            failed.append((key, reason))

    logging.info("Loaded %d/%d policy params.", len(flat_shape) - len(failed), len(flat_shape))
    if failed:
        logging.warning("Policy checkpoint had %d missing/mismatched params; using initialization.", len(failed))
        for key, reason in failed[:30]:
            logging.warning("  %s -> %s", _path_key(key), reason)
        if len(failed) > 30:
            logging.warning("  ... %d more", len(failed) - 30)
    return traverse_util.unflatten_dict(filtered)


def _override_config(args: Args) -> _config.TrainConfig:
    base = _config.get_config(args.config_name)
    data = base.data
    if args.repo_id is not None:
        data = dataclasses.replace(data, repo_id=args.repo_id)
    if args.filter_path is not None:
        base_data_config = data.base_config or _config.DataConfig()
        data = dataclasses.replace(
            data,
            base_config=dataclasses.replace(base_data_config, filter_dict_path=args.filter_path),
        )
    if args.asset_id is not None or args.assets_dir is not None:
        assets = dataclasses.replace(
            data.assets,
            asset_id=args.asset_id if args.asset_id is not None else data.assets.asset_id,
            assets_dir=args.assets_dir if args.assets_dir is not None else data.assets.assets_dir,
        )
        data = dataclasses.replace(data, assets=assets)
    return dataclasses.replace(
        base,
        data=data,
        weight_loader=_weight_loaders.CheckpointWeightLoader(str(_resolve_params_path(args.pretrained_params))),
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        wandb_enabled=False,
    )


def _validate_config(config: _config.TrainConfig, offsets: tuple[int, ...]) -> None:
    model_config = config.model
    if not getattr(model_config, "structured_tactile", False):
        raise ValueError("Recursive revision analysis requires structured tactile effort.")
    if not getattr(model_config, "cached_vlm_async_ae_enabled", False):
        raise ValueError(
            f"{config.name} does not have cached_vlm_async_ae_enabled=True. "
            "Use an async ReTouch config."
        )
    if int(model_config.action_horizon) <= max(offsets):
        raise ValueError(f"Max offset {max(offsets)} must be smaller than action_horizon={model_config.action_horizon}.")
    if int(model_config.future_steps_per_segment) <= 0:
        raise ValueError("future_steps_per_segment must be positive.")


def config_mode_values(args_modes: tuple[EvalMode, ...]) -> tuple[EvalMode, ...]:
    """Validate modes while keeping Tyro's tuple value available as a static JIT argument."""
    invalid = sorted(set(args_modes) - set(MODES))
    if invalid:
        raise ValueError(f"Unsupported --modes values: {invalid}; expected a subset of {MODES}.")
    if not args_modes:
        raise ValueError("--modes must contain at least one evaluation behavior.")
    return args_modes


def _init_frozen_model(config: _config.TrainConfig, rng: at.KeyArrayLike) -> tuple[nnx.GraphDef, nnx.State]:
    model = config.model.create(rng)
    graphdef, params = nnx.split(model)
    partial_params = _load_weights_and_validate(config.weight_loader, params.to_pure_dict())
    params.replace_by_pure_dict(partial_params)
    return graphdef, params


def _preprocess(model, observation: _model.Observation, rng: at.KeyArrayLike) -> _model.Observation:
    original_flow_img = observation.flow_img
    original_wrist_flow_img = observation.wrist_flow_img
    original_future_rgb_img = observation.future_rgb_img
    original_future_wrist_rgb_img = observation.future_wrist_rgb_img
    original_scene_flow = observation.scene_flow
    processed = _model.preprocess_observation(rng, observation, train=False, effort_type=model.effort_type)
    return model._restore_aux_images(
        processed,
        original_flow_img,
        original_wrist_flow_img,
        original_future_rgb_img,
        original_future_wrist_rgb_img,
        original_scene_flow,
    )


def _build_branch_attn(
    prefix_tokens: at.Array,
    prefix_mask: at.Array,
    prefix_ar_mask: at.Array,
    branch_tokens: at.Array,
    branch_mask: at.Array,
    branch_ar_mask: at.Array,
) -> tuple[at.Array, at.Array]:
    batch_size = prefix_tokens.shape[0]
    prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
    branch_attn = make_attn_mask(branch_mask, branch_ar_mask)
    branch_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=branch_tokens.shape[1])
    branch_to_prefix = jnp.logical_and(branch_to_prefix, branch_mask[:, :, None])
    prefix_row = jnp.concatenate(
        [prefix_attn, jnp.zeros((batch_size, prefix_tokens.shape[1], branch_tokens.shape[1]), dtype=jnp.bool_)],
        axis=-1,
    )
    branch_row = jnp.concatenate([branch_to_prefix, branch_attn], axis=-1)
    full_attn = jnp.concatenate([prefix_row, branch_row], axis=1)

    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
    branch_positions = prefix_len + jnp.cumsum(branch_mask, axis=-1) - 1
    return full_attn, jnp.concatenate([prefix_positions, branch_positions], axis=1)


def _teacher_future_hidden(
    model,
    processed: _model.Observation,
    prefix_tokens: at.Array,
    prefix_mask: at.Array,
    prefix_ar_mask: at.Array,
    history_effort: at.Array,
    future_effort: at.Array,
    token_actions: at.Array,
    token_time: at.Array,
) -> at.Array:
    branch_tokens, branch_mask, branch_ar_mask, branch_adarms = model.embed_teacher_suffix(
        processed,
        history_effort,
        future_effort,
        token_actions,
        token_time,
    )
    full_attn, positions = _build_branch_attn(
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        branch_tokens,
        branch_mask,
        branch_ar_mask,
    )
    (_, selected_layers), _ = model.PaliGemma.llm(
        [prefix_tokens, None, branch_tokens],
        mask=full_attn,
        positions=positions,
        adarms_cond=[None, None, branch_adarms],
        return_layer_indices=(int(model.distill_layer_indices[-1]),),
    )
    future_force_slice = model._suffix_slices(model.action_horizon)["future_force"]
    return selected_layers[0][2][:, future_force_slice, :].astype(jnp.float32)


def _student_future_hidden(
    model,
    processed: _model.Observation,
    prefix_tokens: at.Array,
    prefix_mask: at.Array,
    prefix_ar_mask: at.Array,
    history_effort: at.Array,
    token_actions: at.Array,
    token_time: at.Array,
    *,
    async_offset: at.Array,
    previous_future_hidden: at.Array | None,
) -> at.Array:
    future_override = (
        model._cached_async_future_query_override(previous_future_hidden, async_offset, token_actions.dtype)
        if previous_future_hidden is not None
        else None
    )
    branch_tokens, branch_mask, branch_ar_mask, branch_adarms, *_ = model.embed_student_suffix(
        processed,
        history_effort,
        token_actions,
        token_time,
        train=False,
        noise_rng=None,
        future_force_query_override=future_override,
        async_offset=async_offset,
    )
    student_out, student_layers = model._forward_student_multilayer(
        prefix_tokens=prefix_tokens,
        prefix_mask=prefix_mask,
        prefix_ar_mask=prefix_ar_mask,
        student_suffix_tokens=branch_tokens,
        student_suffix_mask=branch_mask,
        student_suffix_ar_mask=branch_ar_mask,
        student_adarms=branch_adarms,
    )
    del student_out
    future_force_slice = model._suffix_slices(model.action_horizon)["future_force"]
    return student_layers[-1][:, future_force_slice, :].astype(jnp.float32)


def _project_student_for_cosine(model, student_hidden: at.Array) -> at.Array:
    return model._project_prompt_distill(student_hidden).astype(jnp.float32)


def _latent_cosine(student_hidden: at.Array, teacher_hidden: at.Array, token_mask: at.Array) -> at.Array:
    student = student_hidden / jnp.maximum(jnp.linalg.norm(student_hidden, axis=-1, keepdims=True), 1e-6)
    teacher = teacher_hidden / jnp.maximum(jnp.linalg.norm(teacher_hidden, axis=-1, keepdims=True), 1e-6)
    per_token = jnp.sum(student * teacher, axis=-1)
    mask = token_mask.astype(per_token.dtype)
    return jnp.sum(per_token * mask[None, :], axis=-1) / jnp.maximum(jnp.sum(mask), 1.0)


def _future_token_mask(model, offset: at.Array, *, suffix_only: bool) -> at.Array:
    if not suffix_only:
        return jnp.ones((model.future_force_token_count,), dtype=jnp.bool_)
    segment_ids = jnp.repeat(
        jnp.arange(model.future_tactile_segments, dtype=jnp.int32),
        model.tactile_tokens_per_step,
    )
    segment_start = segment_ids * int(model.future_steps_per_segment)
    return segment_start >= jnp.asarray(offset, dtype=jnp.int32)


def _physical_actions(model, actions: at.Array) -> at.Array:
    hand_slice = slice(model.hand_action_start, model.hand_action_start + model.hand_action_dim)
    return jnp.concatenate([actions[..., : model.arm_action_dim], actions[..., hand_slice]], axis=-1)


def _action_group(model, actions: at.Array, group: Literal["all", "arm", "hand"]) -> at.Array:
    if group == "arm":
        return actions[..., : model.arm_action_dim]
    hand_slice = slice(model.hand_action_start, model.hand_action_start + model.hand_action_dim)
    if group == "hand":
        return actions[..., hand_slice]
    if group == "all":
        return _physical_actions(model, actions)
    raise ValueError(f"Unsupported action group: {group}")


def _action_stats_group(
    model,
    values: at.Array,
    group: Literal["all", "arm", "hand"],
) -> at.Array:
    """Select stats packed as contiguous [arm, hand] physical dimensions."""
    if group == "arm":
        return values[..., : model.arm_action_dim]
    if group == "hand":
        start = model.arm_action_dim
        return values[..., start : start + model.hand_action_dim]
    if group == "all":
        return values
    raise ValueError(f"Unsupported action group: {group}")


def _masked_action_mse(
    prediction: at.Array,
    target: at.Array,
    offset: at.Array,
    action_horizon: int,
    *,
    suffix_only: bool,
) -> at.Array:
    """Compute per-sample MSE on already-selected action dimensions."""
    sq = jnp.square(prediction - target)
    if not suffix_only:
        return jnp.mean(sq, axis=(1, 2))
    mask = (
        jnp.arange(action_horizon, dtype=jnp.int32)[None, :, None]
        >= jnp.asarray(offset, dtype=jnp.int32)
    ).astype(sq.dtype)
    return jnp.sum(sq * mask, axis=(1, 2)) / jnp.maximum(jnp.sum(mask) * sq.shape[-1], 1.0)


def _normalized_action_mse(
    model,
    prediction: at.Array,
    target: at.Array,
    offset: at.Array,
    *,
    group: Literal["all", "arm", "hand"] = "all",
    suffix_only: bool,
) -> at.Array:
    prediction_normalized = _action_group(model, prediction, group)
    target_normalized = _action_group(model, target, group)
    return _masked_action_mse(
        prediction_normalized,
        target_normalized,
        offset,
        model.action_horizon,
        suffix_only=suffix_only,
    )


def _physical_action_mse(
    model,
    prediction: at.Array,
    target: at.Array,
    offset: at.Array,
    action_center: at.Array,
    action_scale: at.Array,
    *,
    group: Literal["all", "arm", "hand"] = "all",
    suffix_only: bool,
) -> at.Array:
    prediction_normalized = _action_group(model, prediction, group)
    target_normalized = _action_group(model, target, group)
    group_center = _action_stats_group(model, action_center, group)
    group_scale = _action_stats_group(model, action_scale, group)
    prediction_physical = prediction_normalized * group_scale + group_center
    target_physical = target_normalized * group_scale + group_center
    return _masked_action_mse(
        prediction_physical,
        target_physical,
        offset,
        model.action_horizon,
        suffix_only=suffix_only,
    )


def _eval_batch(
    args: Args,
    offsets: tuple[int, ...],
    model_def: nnx.GraphDef,
    model_params: nnx.State,
    batch,
    rng: at.KeyArrayLike,
    action_center: at.Array,
    action_scale: at.Array,
) -> dict[str, at.Array]:
    model = nnx.merge(model_def, model_params)
    model.eval()
    observation, target_actions = batch
    preprocess_rng, noise_rng = jax.random.split(rng)
    processed = _preprocess(model, observation, preprocess_rng)
    history_effort, future_effort = model._split_effort(processed, require_future=True, dtype=target_actions.dtype)
    if future_effort is None:
        raise ValueError("Recursive revision analysis requires future tactile in observation.effort.")

    batch_size = processed.state.shape[0]
    token_actions = (
        jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=target_actions.dtype)
        if args.latent_action_condition == "zero"
        else target_actions
    )
    token_time = jnp.ones((batch_size,), dtype=target_actions.dtype)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(processed)
    prefix_kv_cache, _ = model._prefix_kv_cache(prefix_tokens, prefix_mask, prefix_ar_mask)
    action_noise = jax.random.normal(noise_rng, (batch_size, model.action_horizon, model.action_dim), dtype=target_actions.dtype)

    offset0 = jnp.asarray(offsets[0], dtype=jnp.int32)
    if int(offsets[0]) != 0:
        raise ValueError("--offsets must start with 0 for one-shot and recursive baselines.")

    one_hidden0 = _student_future_hidden(
        model,
        processed,
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        history_effort,
        token_actions,
        token_time,
        async_offset=offset0,
        previous_future_hidden=None,
    )
    one_actions0, sample_hidden0 = model._cached_vlm_async_denoise_with_prefix_cache(
        rng,
        processed,
        history_effort=history_effort,
        prefix_mask=prefix_mask,
        prefix_kv_cache=prefix_kv_cache,
        async_chunk_offset=offset0,
        prefix_future_hidden=None,
        noise=action_noise,
        num_steps=args.num_steps,
    )

    results: dict[str, at.Array] = {}
    previous_latent_hidden = one_hidden0
    previous_sample_hidden = sample_hidden0
    requested_modes = set(args.modes)

    for offset_index, offset_int in enumerate(offsets):
        offset = jnp.asarray(offset_int, dtype=jnp.int32)
        offset_history = model._history_effort_for_offset(history_effort, future_effort, offset)
        teacher_hidden = _teacher_future_hidden(
            model,
            processed,
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            offset_history,
            future_effort,
            token_actions,
            token_time,
        )
        teacher_hidden = jax.lax.stop_gradient(teacher_hidden)

        hidden_by_mode: dict[str, at.Array] = {}
        action_by_mode: dict[str, at.Array] = {}
        if "one_shot" in requested_modes:
            hidden_by_mode["one_shot"] = one_hidden0
            action_by_mode["one_shot"] = one_actions0

        if offset_index == 0:
            if "action_only" in requested_modes:
                hidden_by_mode["action_only"] = one_hidden0
                action_by_mode["action_only"] = one_actions0
            if "fresh_reinfer" in requested_modes:
                hidden_by_mode["fresh_reinfer"] = one_hidden0
                action_by_mode["fresh_reinfer"] = one_actions0
            if "retouch" in requested_modes:
                hidden_by_mode["retouch"] = one_hidden0
                action_by_mode["retouch"] = one_actions0
        else:
            if requested_modes.intersection(("action_only", "fresh_reinfer")):
                if "fresh_reinfer" in requested_modes:
                    fresh_hidden = _student_future_hidden(
                        model,
                        processed,
                        prefix_tokens,
                        prefix_mask,
                        prefix_ar_mask,
                        offset_history,
                        token_actions,
                        token_time,
                        async_offset=offset,
                        previous_future_hidden=None,
                    )
                fresh_actions, _ = model._cached_vlm_async_denoise_with_prefix_cache(
                    rng,
                    processed,
                    history_effort=offset_history,
                    prefix_mask=prefix_mask,
                    prefix_kv_cache=prefix_kv_cache,
                    async_chunk_offset=offset,
                    prefix_future_hidden=None,
                    noise=action_noise,
                    num_steps=args.num_steps,
                )
                if "action_only" in requested_modes:
                    # Action is refreshed from the latest tactile window, while the
                    # future-contact prediction remains the one made at t0.
                    hidden_by_mode["action_only"] = one_hidden0
                    action_by_mode["action_only"] = fresh_actions
                if "fresh_reinfer" in requested_modes:
                    hidden_by_mode["fresh_reinfer"] = fresh_hidden
                    action_by_mode["fresh_reinfer"] = fresh_actions

            if "retouch" in requested_modes:
                retouch_hidden = _student_future_hidden(
                    model,
                    processed,
                    prefix_tokens,
                    prefix_mask,
                    prefix_ar_mask,
                    offset_history,
                    token_actions,
                    token_time,
                    async_offset=offset,
                    previous_future_hidden=previous_latent_hidden,
                )
                retouch_actions, previous_sample_hidden = model._cached_vlm_async_denoise_with_prefix_cache(
                    rng,
                    processed,
                    history_effort=offset_history,
                    prefix_mask=prefix_mask,
                    prefix_kv_cache=prefix_kv_cache,
                    async_chunk_offset=offset,
                    prefix_future_hidden=previous_sample_hidden,
                    noise=action_noise,
                    num_steps=args.num_steps,
                )
                previous_latent_hidden = retouch_hidden
                hidden_by_mode["retouch"] = retouch_hidden
                action_by_mode["retouch"] = retouch_actions

        full_token_mask = _future_token_mask(model, offset, suffix_only=False)
        suffix_token_mask = _future_token_mask(model, offset, suffix_only=True)
        for mode in args.modes:
            projected_student = _project_student_for_cosine(model, hidden_by_mode[mode])
            results[f"latent_cosine/full/{mode}/{offset_int}"] = _latent_cosine(
                projected_student,
                teacher_hidden,
                full_token_mask,
            )
            results[f"latent_cosine/suffix/{mode}/{offset_int}"] = _latent_cosine(
                projected_student,
                teacher_hidden,
                suffix_token_mask,
            )
            # Keep the historical ``action_mse`` key for backward
            # compatibility. It denotes MSE after inverse normalization.
            results[f"action_mse/full/{mode}/{offset_int}"] = _physical_action_mse(
                model,
                action_by_mode[mode],
                target_actions,
                offset,
                action_center,
                action_scale,
                suffix_only=False,
            )
            results[f"action_mse/suffix/{mode}/{offset_int}"] = _physical_action_mse(
                model,
                action_by_mode[mode],
                target_actions,
                offset,
                action_center,
                action_scale,
                suffix_only=True,
            )
            results[f"action_mse_normalized/full/{mode}/{offset_int}"] = _normalized_action_mse(
                model,
                action_by_mode[mode],
                target_actions,
                offset,
                suffix_only=False,
            )
            results[f"action_mse_normalized/suffix/{mode}/{offset_int}"] = _normalized_action_mse(
                model,
                action_by_mode[mode],
                target_actions,
                offset,
                suffix_only=True,
            )
            for group in ("arm", "hand"):
                results[f"action_mse_{group}/full/{mode}/{offset_int}"] = _physical_action_mse(
                    model,
                    action_by_mode[mode],
                    target_actions,
                    offset,
                    action_center,
                    action_scale,
                    group=group,
                    suffix_only=False,
                )
                results[f"action_mse_{group}/suffix/{mode}/{offset_int}"] = _physical_action_mse(
                    model,
                    action_by_mode[mode],
                    target_actions,
                    offset,
                    action_center,
                    action_scale,
                    group=group,
                    suffix_only=True,
                )
                results[f"action_mse_normalized_{group}/full/{mode}/{offset_int}"] = _normalized_action_mse(
                    model,
                    action_by_mode[mode],
                    target_actions,
                    offset,
                    group=group,
                    suffix_only=False,
                )
                results[f"action_mse_normalized_{group}/suffix/{mode}/{offset_int}"] = _normalized_action_mse(
                    model,
                    action_by_mode[mode],
                    target_actions,
                    offset,
                    group=group,
                    suffix_only=True,
                )

    return results


def _phase_masks(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    offset: int,
    contact_onsets: dict[int, int | None],
) -> dict[str, np.ndarray]:
    eval_frames = frame_indices.astype(np.int64) + int(offset)
    pre_contact = np.zeros_like(eval_frames, dtype=bool)
    post_contact = np.zeros_like(eval_frames, dtype=bool)
    no_contact = np.zeros_like(eval_frames, dtype=bool)
    for index, (episode, frame) in enumerate(zip(episode_indices, eval_frames, strict=True)):
        onset = contact_onsets.get(int(episode))
        if onset is None:
            no_contact[index] = True
        elif int(frame) < onset:
            pre_contact[index] = True
        else:
            post_contact[index] = True
    return {
        "overall": np.ones_like(eval_frames, dtype=bool),
        "pre_contact": pre_contact,
        "post_contact": post_contact,
        "no_contact": no_contact,
    }


def _write_metrics(
    output_dir: pathlib.Path,
    rows: list[dict[str, object]],
    metrics: dict[str, float],
    args: Args,
    contact_summary: dict[str, object] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, ensure_ascii=False))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    if contact_summary is not None:
        (output_dir / "contact_onsets.json").write_text(
            json.dumps(contact_summary, indent=2, ensure_ascii=False)
        )
    with (output_dir / "metrics_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "metric", "scope", "mode", "offset", "phase", "value", "samples"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_offset_zero_consistency(
    metrics: dict[str, float],
    modes: tuple[EvalMode, ...],
    *,
    atol: float = 1e-8,
) -> None:
    """Verify that inference behaviors share the same initial prediction.

    At offset zero, one-shot, fresh re-inference, and recursive ReTouch all use
    ``one_hidden0`` and ``one_actions0``. Any difference therefore indicates
    that the comparison did not come from one shared model/evaluation pass.
    """
    if len(modes) < 2:
        return
    reference_mode = modes[0]
    checked = 0
    mismatches: list[str] = []
    for metric in (
        "latent_cosine",
        "action_mse",
        "action_mse_arm",
        "action_mse_hand",
        "action_mse_normalized",
        "action_mse_normalized_arm",
        "action_mse_normalized_hand",
    ):
        for scope in ("full", "suffix"):
            prefix = f"{metric}/{scope}/{reference_mode}/0/"
            phases = sorted(key.removeprefix(prefix) for key in metrics if key.startswith(prefix))
            for phase in phases:
                reference = metrics[f"{metric}/{scope}/{reference_mode}/0/{phase}"]
                if not np.isfinite(reference):
                    continue
                for mode in modes[1:]:
                    key = f"{metric}/{scope}/{mode}/0/{phase}"
                    value = metrics.get(key)
                    if value is None or not np.isfinite(value):
                        continue
                    checked += 1
                    if not np.isclose(reference, value, rtol=0.0, atol=atol):
                        mismatches.append(
                            f"{metric}/{scope}/0/{phase}: "
                            f"{reference_mode}={reference:.10g}, {mode}={value:.10g}"
                        )
    if mismatches:
        details = "\n  ".join(mismatches)
        raise RuntimeError(
            "Offset-0 consistency check failed. A fair fixed-checkpoint comparison "
            f"must produce identical initial predictions:\n  {details}"
        )
    logging.info(
        "Offset-0 consistency check passed for modes=%s across %d comparisons.",
        modes,
        checked,
    )


def _plot_metric(
    output_dir: pathlib.Path,
    rows: list[dict[str, object]],
    *,
    metric: str,
    scope: str,
    phase: str,
    modes: tuple[EvalMode, ...],
    ylabel: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=180)
    plotted = False
    for mode in modes:
        selected = [
            row
            for row in rows
            if row["metric"] == metric
            and row["scope"] == scope
            and row["mode"] == mode
            and row["phase"] == phase
            and int(row["samples"]) > 0
        ]
        selected = sorted(selected, key=lambda row: int(row["offset"]))
        if not selected:
            continue
        ax.plot(
            [int(row["offset"]) for row in selected],
            [float(row["value"]) for row in selected],
            marker="o",
            linewidth=2,
            label=mode,
        )
        plotted = True
    ax.set_xlabel("async offset")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def _action_unnormalization_affine(data_config: _config.DataConfig) -> tuple[np.ndarray, np.ndarray]:
    if data_config.norm_stats is None or "actions" not in data_config.norm_stats:
        raise ValueError("Action normalization stats are required for physical-space action MSE.")
    stats = data_config.norm_stats["actions"]
    if data_config.use_quantile_norm:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile action normalization requires q01/q99 stats.")
        q01 = np.asarray(stats.q01, dtype=np.float32)
        q99 = np.asarray(stats.q99, dtype=np.float32)
        return (q01 + q99) * 0.5, (q99 - q01 + 1e-6) * 0.5
    return np.asarray(stats.mean, dtype=np.float32), np.asarray(stats.std, dtype=np.float32) + 1e-6


def main(args: Args) -> None:
    init_logging()
    config_mode_values(args.modes)
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    config = _override_config(args)
    _validate_config(config, args.offsets)
    if args.batch_size % jax.process_count() != 0:
        raise ValueError(f"--batch-size must be divisible by jax.process_count()={jax.process_count()}.")

    logging.info("Config: %s", args.config_name)
    logging.info("Policy params: %s", args.pretrained_params)
    logging.info("Dataset: repo=%s asset=%s filter=%s", args.repo_id, args.asset_id, args.filter_path)
    logging.info("Offsets: %s", args.offsets)
    logging.info("Evaluation behavior(s): %s", args.modes)

    contact_onsets: dict[int, int | None] = {}
    contact_summary: dict[str, object] | None = None
    if args.split_by_contact:
        contact_onsets, contact_summary = _compute_contact_onsets(args)
        logging.info(
            "Contact split: %d/%d episodes contain contact (threshold=%.3f, min_taxels=%d, consecutive=%d).",
            contact_summary["num_episodes_with_contact"],
            contact_summary["num_episodes"],
            args.contact_threshold,
            args.contact_min_taxels,
            args.contact_min_consecutive_frames,
        )

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False)
    action_center_np, action_scale_np = _action_unnormalization_affine(data_loader.data_config())
    logging.info("Action MSE will be computed after unnormalizing %d physical action dimensions.", action_center_np.size)

    rng = jax.random.PRNGKey(args.seed)
    model_rng, eval_rng = jax.random.split(rng)
    with sharding.set_mesh(mesh):
        model_def, model_params = _init_frozen_model(config, model_rng)
        model_params = jax.device_put(model_params, replicated)

        action_center = jax.device_put(jnp.asarray(action_center_np), replicated)
        action_scale = jax.device_put(jnp.asarray(action_scale_np), replicated)

        def peval(model_params_arg, batch_arg, rng_arg):
            return _eval_batch(
                args,
                args.offsets,
                model_def,
                model_params_arg,
                batch_arg,
                rng_arg,
                action_center,
                action_scale,
            )

        peval = jax.jit(peval)
        sums: dict[tuple[str, str], float] = {}
        sample_counts: dict[tuple[str, str], int] = {}
        batch_count = 0

        # Use the transformed dictionary iterator directly so episode/frame IDs
        # remain available for contact-phase grouping. Observation.from_dict
        # ignores these metadata fields when constructing model inputs.
        raw_loader = data_loader._data_loader  # noqa: SLF001
        available_batches = len(raw_loader.torch_loader)
        target_batches = available_batches if args.max_batches <= 0 else min(args.max_batches, available_batches)
        for batch_index, raw_batch in enumerate(raw_loader):
            if batch_index >= target_batches:
                break
            if "episode_index" not in raw_batch or "frame_index" not in raw_batch:
                raise KeyError(
                    "Transformed eval batch is missing episode_index/frame_index. "
                    "Use XHandTactileFlowInputs with metadata preservation enabled."
                )
            episode_indices = np.asarray(jax.device_get(raw_batch["episode_index"]), dtype=np.int64).reshape(-1)
            frame_indices = np.asarray(jax.device_get(raw_batch["frame_index"]), dtype=np.int64).reshape(-1)
            batch = (_model.Observation.from_dict(raw_batch), raw_batch["actions"])
            batch_rng = jax.random.fold_in(eval_rng, batch_index)
            batch_metrics = peval(model_params, batch, batch_rng)
            batch_metrics = jax.device_get(batch_metrics)
            for key, value in batch_metrics.items():
                values = np.asarray(value, dtype=np.float64).reshape(-1)
                if values.shape[0] != episode_indices.shape[0]:
                    raise ValueError(
                        f"Metric {key} returned {values.shape[0]} samples for a batch of "
                        f"{episode_indices.shape[0]}."
                    )
                _, _, _, offset_text = key.split("/")
                phase_masks = (
                    _phase_masks(episode_indices, frame_indices, int(offset_text), contact_onsets)
                    if args.split_by_contact
                    else {"overall": np.ones_like(episode_indices, dtype=bool)}
                )
                for phase, phase_mask in phase_masks.items():
                    phase_count = int(np.sum(phase_mask))
                    aggregate_key = (key, phase)
                    if phase_count > 0:
                        sums[aggregate_key] = sums.get(aggregate_key, 0.0) + float(np.sum(values[phase_mask]))
                    sample_counts[aggregate_key] = sample_counts.get(aggregate_key, 0) + phase_count
            batch_count += 1
            if batch_count == 1 or batch_count % 10 == 0:
                logging.info("Processed %d/%d batches.", batch_count, target_batches)

    if batch_count == 0:
        raise RuntimeError("No batches were evaluated.")

    metrics: dict[str, float] = {}
    rows = []
    model_label = args.model_label or args.config_name
    for (key, phase), count in sorted(sample_counts.items()):
        value = float("nan") if count == 0 else sums[(key, phase)] / count
        metric, scope, mode, offset = key.split("/")
        output_key = f"{metric}/{scope}/{mode}/{offset}/{phase}"
        metrics[output_key] = value
        rows.append(
            {
                "model": model_label,
                "metric": metric,
                "scope": scope,
                "mode": mode,
                "offset": int(offset),
                "phase": phase,
                "value": value,
                "samples": count,
            }
        )
    _validate_offset_zero_consistency(metrics, args.modes)
    _write_metrics(output_dir, rows, metrics, args, contact_summary)
    phases = PHASES if args.split_by_contact else ("overall",)
    for scope in ("full", "suffix"):
        for phase in phases:
            phase_suffix = "" if phase == "overall" else f"_{phase}"
            _plot_metric(
                output_dir,
                rows,
                metric="latent_cosine",
                scope=scope,
                phase=phase,
                modes=args.modes,
                ylabel=f"latent cosine ({scope}, {phase}, higher is better)",
                filename=f"latent_cosine_{scope}{phase_suffix}.png",
            )
            action_metric_specs = (
                ("action_mse", "physical action"),
                ("action_mse_arm", "physical arm action"),
                ("action_mse_hand", "physical hand action"),
                ("action_mse_normalized", "normalized action"),
                ("action_mse_normalized_arm", "normalized arm action"),
                ("action_mse_normalized_hand", "normalized hand action"),
            )
            for metric, label in action_metric_specs:
                _plot_metric(
                    output_dir,
                    rows,
                    metric=metric,
                    scope=scope,
                    phase=phase,
                    modes=args.modes,
                    ylabel=f"{label} {scope} MSE ({phase}, lower is better)",
                    filename=f"{metric}_{scope}{phase_suffix}.png",
                )
    logging.info("Saved recursive revision metrics to %s", output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
