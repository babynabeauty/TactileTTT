#!/usr/bin/env python3
"""Probe whether dual-AE future tactile tokens encode future 5x3 force.

This script freezes a trained Pi0LatentFlow policy, extracts the hidden states
of the future tactile tokens at a selected transformer layer, and trains a small
decoder to reconstruct the normalized future five-finger force sequence.

export HF_LEROBOT_HOME=$PWD
export HF_HUB_OFFLINE=1

DATA_REPO=data/task1_2_3_315ep
ASSET_ID=$(basename "$DATA_REPO")
POLICY_PARAMS=checkpoints/pi0_xhand_tactile_structured_dual_ae_history_future_pool/B_calc_dual_ae_history_future_pool_task1_2_3_315ep_pool_30k_0628/29999/params

setsid nohup env \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env/.venv/bin/python scripts/train_future_tactile_token_probe.py \
    --exp-name probe_teacher_task123_valsplit \
    --pretrained-params "$POLICY_PARAMS" \
    --repo-id "$DATA_REPO" \
    --asset-id "$ASSET_ID" \
    --assets-dir assets/pi0_xhand_tactile_structured_dual_ae \
    --train-filter-path outputs/episode_splits/task1_2_3_315ep/train_episodes.json \
    --eval-filter-path outputs/episode_splits/task1_2_3_315ep/val_episodes.json \
    --token-source teacher \
    --num-train-steps 5000 \
    --batch-size 8 \
    --fsdp-devices 1 \
    --num-workers 2 \
    --save-interval 1000 \
    --overwrite \
  > logs/probe_teacher_task123_valsplit.log 2>&1 &


"""

import dataclasses
import json
import logging
import pathlib
import shutil
from typing import Literal

import einops
import flax.nnx as nnx
from flax import struct
from flax import traverse_util
import jax
from jax import ShapeDtypeStruct
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import tyro

from openpi.models import model_tavla as _model
from openpi.models import gemma as _gemma
from openpi.models.pi0_tavla import make_attn_mask
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_xhand_tactile_structured_dual_ae_history_future_pool"
    exp_name: str = tyro.MISSING
    pretrained_params: str = tyro.MISSING

    repo_id: str | None = None
    asset_id: str | None = None
    assets_dir: str | None = None
    train_filter_path: str | None = None
    eval_filter_path: str | None = None

    token_source: Literal["student", "teacher"] = "student"
    probe_layer: int | None = None

    num_train_steps: int = 5000
    batch_size: int = 8
    fsdp_devices: int = 1
    num_workers: int = 0
    learning_rate: float = 1e-4
    seed: int = 42

    decoder_dim: int = 512
    decoder_depth: int = 2

    force_loss_weight: float = 1.0
    delta_loss_weight: float = 0.2
    magnitude_loss_weight: float = 0.2
    contact_loss_weight: float = 0.1
    contact_threshold: float = 1.0

    log_interval: int = 50
    save_interval: int = 1000
    checkpoint_base_dir: str = "checkpoints/future_tactile_token_probe"
    overwrite: bool = False
    resume: bool = False


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


class ProbeFFNBlock(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.ff_in = nnx.Linear(hidden_dim, hidden_dim * 4, rngs=rngs)
        self.ff_out = nnx.Linear(hidden_dim * 4, hidden_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b t f d"]) -> at.Float[at.Array, "b t f d"]:
        residual = x
        x = self.norm(x)
        x = self.ff_in(x)
        x = nnx.swish(x)
        x = self.ff_out(x)
        return residual + x


class FutureTactileForceProbe(nnx.Module):
    """Decode pooled future finger tokens [B, 5, D] into [B, 32, 5, 3]."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_horizon: int,
        num_fingers: int,
        force_dim: int,
        depth: int,
        rngs: nnx.Rngs,
    ):
        self.action_horizon = int(action_horizon)
        self.num_fingers = int(num_fingers)
        self.force_dim = int(force_dim)
        self.token_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.time_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.action_horizon, hidden_dim), dtype=jnp.float32)
        )
        self.finger_embedding = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.num_fingers, hidden_dim), dtype=jnp.float32)
        )
        self.blocks = [ProbeFFNBlock(hidden_dim, rngs=rngs) for _ in range(int(depth))]
        self.out_norm = nnx.LayerNorm(num_features=hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, self.force_dim, rngs=rngs)

    def __call__(self, future_hidden: at.Float[at.Array, "b f d"]) -> at.Float[at.Array, "b t f c"]:
        if future_hidden.shape[1] != self.num_fingers:
            raise ValueError(f"Expected {self.num_fingers} future finger tokens, got {future_hidden.shape}.")
        x = self.token_proj(future_hidden)
        x = (
            x[:, None, :, :]
            + jnp.asarray(self.time_embedding.value, dtype=x.dtype)[None, :, None, :]
            + jnp.asarray(self.finger_embedding.value, dtype=x.dtype)[None, None, :, :]
        )
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.out_norm(x))


@struct.dataclass
class ProbeTrainState:
    step: int
    params: nnx.State
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)


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

    logging.info("Loaded %d/%d frozen policy params.", len(flat_shape) - len(failed), len(flat_shape))
    if failed:
        logging.warning("Policy checkpoint had %d missing/mismatched params; using initialization for them.", len(failed))
        for key, reason in failed[:30]:
            logging.warning("  %s -> %s", _path_key(key), reason)
        if len(failed) > 30:
            logging.warning("  ... %d more", len(failed) - 30)
    return traverse_util.unflatten_dict(filtered)


def _override_config(args: Args, *, filter_path: str | None = None) -> _config.TrainConfig:
    base = _config.get_config(args.config_name)
    data = base.data
    if args.repo_id is not None:
        data = dataclasses.replace(data, repo_id=args.repo_id)
    if filter_path is not None:
        base_data_config = data.base_config or _config.DataConfig()
        data = dataclasses.replace(
            data,
            base_config=dataclasses.replace(base_data_config, filter_dict_path=filter_path),
        )
    if args.asset_id is not None or args.assets_dir is not None:
        assets = data.assets
        assets = dataclasses.replace(
            assets,
            asset_id=args.asset_id if args.asset_id is not None else assets.asset_id,
            assets_dir=args.assets_dir if args.assets_dir is not None else assets.assets_dir,
        )
        data = dataclasses.replace(data, assets=assets)
    return dataclasses.replace(
        base,
        exp_name=args.exp_name,
        data=data,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.pretrained_params),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        num_train_steps=args.num_train_steps,
        fsdp_devices=args.fsdp_devices,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=False,
    )


def _validate_model_config(config: _config.TrainConfig) -> None:
    model_config = config.model
    required = {
        "structured_tactile": True,
        "pool_tactile_history": True,
        "future_tactile_segments": 1,
        "future_steps_per_segment": model_config.action_horizon,
        "tactile_points_per_finger": 1,
        "tactile_num_fingers": 5,
        "tactile_dim_per_finger": 3,
    }
    for name, expected in required.items():
        actual = getattr(model_config, name)
        if actual != expected:
            raise ValueError(
                f"{config.name} is not the expected pooled 5x3 tactile config: {name}={actual}, expected {expected}."
            )


def _init_frozen_model(config: _config.TrainConfig, rng: at.KeyArrayLike) -> tuple[nnx.GraphDef, nnx.State]:
    model = config.model.create(rng)
    graphdef, params = nnx.split(model)
    partial_params = _load_weights_and_validate(config.weight_loader, params.to_pure_dict())
    params.replace_by_pure_dict(partial_params)
    return graphdef, params


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
    positions = jnp.concatenate([prefix_positions, branch_positions], axis=1)
    return full_attn, positions


def _extract_future_token_hiddens(
    model,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    probe_layer: int,
    token_source: Literal["student", "teacher"],
) -> tuple[at.Array, at.Array]:
    original_flow_img = observation.flow_img
    original_wrist_flow_img = observation.wrist_flow_img
    original_future_rgb_img = observation.future_rgb_img
    original_future_wrist_rgb_img = observation.future_wrist_rgb_img
    original_scene_flow = observation.scene_flow
    processed = _model.preprocess_observation(rng, observation, train=False, effort_type=model.effort_type)
    processed = model._restore_aux_images(
        processed,
        original_flow_img,
        original_wrist_flow_img,
        original_future_rgb_img,
        original_future_wrist_rgb_img,
        original_scene_flow,
    )

    history_effort, future_effort = model._split_effort(processed, require_future=True, dtype=actions.dtype)
    if future_effort is None:
        raise ValueError("Future tactile token probe requires future effort in observation.effort.")

    batch_size = processed.state.shape[0]
    zero_actions = jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=actions.dtype)
    time = jnp.ones((batch_size,), dtype=actions.dtype)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(processed)
    if token_source == "student":
        branch_tokens, branch_mask, branch_ar_mask, branch_adarms, *_ = model.embed_student_suffix(
            processed,
            history_effort,
            zero_actions,
            time,
            train=False,
            noise_rng=None,
        )
        llm_inputs = [prefix_tokens, branch_tokens, None]
        adarms_cond = [None, branch_adarms, None]
        branch_index = 1
    else:
        branch_tokens, branch_mask, branch_ar_mask, branch_adarms = model.embed_teacher_suffix(
            processed,
            history_effort,
            future_effort,
            zero_actions,
            time,
        )
        llm_inputs = [prefix_tokens, None, branch_tokens]
        adarms_cond = [None, None, branch_adarms]
        branch_index = 2

    full_attn, positions = _build_branch_attn(
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        branch_tokens,
        branch_mask,
        branch_ar_mask,
    )
    (outputs, selected_layers), _ = model.PaliGemma.llm(
        llm_inputs,
        mask=full_attn,
        positions=positions,
        adarms_cond=adarms_cond,
        return_layer_indices=(int(probe_layer),),
    )
    del outputs
    branch_hidden = selected_layers[0][branch_index]
    future_force_slice = model._suffix_slices(model.action_horizon)["future_force"]
    future_hidden = branch_hidden[:, future_force_slice, :].astype(jnp.float32)
    return jax.lax.stop_gradient(future_hidden), future_effort.astype(jnp.float32), history_effort.astype(jnp.float32)


def _smooth_l1(x: at.Array) -> at.Array:
    abs_x = jnp.abs(x)
    return jnp.where(abs_x < 1.0, 0.5 * jnp.square(abs_x), abs_x - 0.5)


def _contact_stats(args: Args, pred_mag: at.Array, target_mag: at.Array) -> dict[str, at.Array]:
    pred_contact = pred_mag > args.contact_threshold
    true_contact = target_mag > args.contact_threshold
    tp = jnp.sum(jnp.logical_and(pred_contact, true_contact))
    fp = jnp.sum(jnp.logical_and(pred_contact, jnp.logical_not(true_contact)))
    fn = jnp.sum(jnp.logical_and(jnp.logical_not(pred_contact), true_contact))
    tn = jnp.sum(jnp.logical_and(jnp.logical_not(pred_contact), jnp.logical_not(true_contact)))
    precision = tp / jnp.maximum(tp + fp, 1.0)
    recall = tp / jnp.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / jnp.maximum(precision + recall, 1e-6)
    accuracy = (tp + tn) / jnp.maximum(tp + fp + fn + tn, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _loss_and_stats(
    args: Args,
    pred_force: at.Array,
    target_force: at.Array,
    history_force: at.Array | None = None,
) -> tuple[at.Array, dict[str, at.Array]]:
    loss_force = jnp.mean(_smooth_l1(pred_force - target_force))
    loss_delta = jnp.mean(_smooth_l1((pred_force[:, 1:] - pred_force[:, :-1]) - (target_force[:, 1:] - target_force[:, :-1])))

    pred_mag = jnp.linalg.norm(pred_force, axis=-1)
    target_mag = jnp.linalg.norm(target_force, axis=-1)
    loss_magnitude = jnp.mean(_smooth_l1(pred_mag - target_mag))

    target_contact = (target_mag > args.contact_threshold).astype(jnp.float32)
    contact_logits = pred_mag - args.contact_threshold
    loss_contact = jnp.mean(optax.sigmoid_binary_cross_entropy(contact_logits, target_contact))

    contact = _contact_stats(args, pred_mag, target_mag)

    total = (
        args.force_loss_weight * loss_force
        + args.delta_loss_weight * loss_delta
        + args.magnitude_loss_weight * loss_magnitude
        + args.contact_loss_weight * loss_contact
    )
    stats = {
        "loss/total": total,
        "loss/force_smooth_l1": loss_force,
        "loss/delta_smooth_l1": loss_delta,
        "loss/magnitude_smooth_l1": loss_magnitude,
        "loss/contact_bce": loss_contact,
        "metric/contact_precision": contact["precision"],
        "metric/contact_recall": contact["recall"],
        "metric/contact_f1": contact["f1"],
        "metric/contact_accuracy": contact["accuracy"],
        "metric/pred_mag_mean": jnp.mean(pred_mag),
        "metric/target_mag_mean": jnp.mean(target_mag),
        "metric/target_contact_ratio": jnp.mean(target_contact),
    }
    if history_force is not None:
        last_history = jnp.repeat(history_force[:, -1:, :, :], target_force.shape[1], axis=1)
        baseline_mag = jnp.linalg.norm(last_history, axis=-1)
        baseline_contact = _contact_stats(args, baseline_mag, target_mag)
        stats.update(
            {
                "baseline_last_history/force_smooth_l1": jnp.mean(_smooth_l1(last_history - target_force)),
                "baseline_last_history/magnitude_smooth_l1": jnp.mean(_smooth_l1(baseline_mag - target_mag)),
                "baseline_last_history/contact_f1": baseline_contact["f1"],
                "baseline_last_history/contact_accuracy": baseline_contact["accuracy"],
            }
        )
    return total, stats


def train_step(
    args: Args,
    probe_layer: int,
    model_def: nnx.GraphDef,
    model_params: nnx.State,
    probe_def: nnx.GraphDef,
    state: ProbeTrainState,
    batch,
    rng: at.KeyArrayLike,
) -> tuple[ProbeTrainState, dict[str, at.Array]]:
    model = nnx.merge(model_def, model_params)
    model.eval()
    probe = nnx.merge(probe_def, state.params)

    def loss_fn(probe, step_rng, observation, actions):
        future_hidden, target_force, history_force = _extract_future_token_hiddens(
            model,
            step_rng,
            observation,
            actions,
            probe_layer=probe_layer,
            token_source=args.token_source,
        )
        pred_force = probe(future_hidden)
        if pred_force.shape != target_force.shape:
            raise ValueError(f"Probe output {pred_force.shape} does not match target {target_force.shape}.")
        return _loss_and_stats(args, pred_force, target_force, history_force)

    step_rng = jax.random.fold_in(rng, state.step)
    (loss, stats), grads = nnx.value_and_grad(loss_fn, has_aux=True)(probe, step_rng, batch[0], batch[1])
    del loss
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    nnx.update(probe, new_params)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(probe),
        opt_state=new_opt_state,
    )
    stats = {
        **stats,
        "grad_norm": optax.global_norm(grads),
        "probe_param_norm": optax.global_norm(new_state.params),
    }
    return new_state, stats


def eval_step(
    args: Args,
    probe_layer: int,
    model_def: nnx.GraphDef,
    model_params: nnx.State,
    probe_def: nnx.GraphDef,
    state: ProbeTrainState,
    batch,
    rng: at.KeyArrayLike,
) -> dict[str, at.Array]:
    model = nnx.merge(model_def, model_params)
    model.eval()
    probe = nnx.merge(probe_def, state.params)
    future_hidden, target_force, history_force = _extract_future_token_hiddens(
        model,
        rng,
        batch[0],
        batch[1],
        probe_layer=probe_layer,
        token_source=args.token_source,
    )
    pred_force = probe(future_hidden)
    _, stats = _loss_and_stats(args, pred_force, target_force, history_force)
    return stats


def _format_stats(stats: dict[str, at.Array]) -> str:
    parts = []
    for key in sorted(stats):
        value = jax.device_get(stats[key])
        try:
            scalar = float(value)
        except TypeError:
            continue
        parts.append(f"{key}={scalar:.5f}")
    return " ".join(parts)


def _checkpoint_dir(args: Args) -> pathlib.Path:
    return (
        pathlib.Path(args.checkpoint_base_dir)
        .expanduser()
        .resolve()
        / args.config_name
        / args.exp_name
        / args.token_source
    )


def main(args: Args) -> None:
    init_logging()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot both be set.")

    train_config = _override_config(args, filter_path=args.train_filter_path)
    eval_config = _override_config(args, filter_path=args.eval_filter_path)
    _validate_model_config(train_config)
    _validate_model_config(eval_config)
    probe_layer = int(args.probe_layer if args.probe_layer is not None else train_config.model.future_tactile_align_layer)
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"--batch-size must be divisible by jax.device_count()={jax.device_count()}.")

    checkpoint_dir = _checkpoint_dir(args)
    if checkpoint_dir.exists() and args.overwrite:
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, ensure_ascii=False))

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_loader = _data_loader.create_data_loader(train_config, sharding=data_sharding, shuffle=True)
    eval_loader = _data_loader.create_data_loader(eval_config, sharding=data_sharding, shuffle=False)

    init_rng = jax.random.PRNGKey(args.seed)
    model_rng, probe_rng, train_rng = jax.random.split(init_rng, 3)
    with sharding.set_mesh(mesh):
        model_def, model_params = _init_frozen_model(train_config, model_rng)
        model_params = jax.device_put(model_params, replicated)

        student_width = int(_gemma.get_config(train_config.model.action_expert_variant).width)
        teacher_variant = getattr(train_config.model, "force_expert_variant", train_config.model.action_expert_variant)
        teacher_width = int(_gemma.get_config(teacher_variant).width)
        probe_input_width = teacher_width if args.token_source == "teacher" else student_width
        probe = FutureTactileForceProbe(
            input_dim=probe_input_width,
            hidden_dim=args.decoder_dim,
            action_horizon=train_config.model.action_horizon,
            num_fingers=train_config.model.tactile_num_fingers,
            force_dim=train_config.model.tactile_dim_per_finger,
            depth=args.decoder_depth,
            rngs=nnx.Rngs(probe_rng),
        )
        probe_def, probe_params = nnx.split(probe)
        tx = optax.adamw(args.learning_rate)
        opt_state = tx.init(probe_params)
        state = ProbeTrainState(step=0, params=probe_params, opt_state=opt_state, tx=tx)
        state = jax.device_put(state, replicated)

        checkpointer = ocp.CheckpointManager(
            checkpoint_dir,
            item_handlers={"probe_state": ocp.PyTreeCheckpointHandler()},
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=args.save_interval,
                max_to_keep=3,
                keep_period=args.save_interval,
                create=True,
            ),
        )
        if args.resume:
            latest_step = checkpointer.latest_step()
            if latest_step is not None:
                state = checkpointer.restore(latest_step, args=ocp.args.Composite(probe_state=ocp.args.PyTreeRestore(state)))[
                    "probe_state"
                ]
                logging.info("Resumed probe checkpoint step %d from %s.", latest_step, checkpoint_dir)

        def _ptrain_step(model_params_arg, state_arg, batch_arg, rng_arg):
            return train_step(
                args,
                probe_layer,
                model_def,
                model_params_arg,
                probe_def,
                state_arg,
                batch_arg,
                rng_arg,
            )

        def _peval_step(model_params_arg, state_arg, batch_arg, rng_arg):
            return eval_step(
                args,
                probe_layer,
                model_def,
                model_params_arg,
                probe_def,
                state_arg,
                batch_arg,
                rng_arg,
            )

        ptrain_step = jax.jit(_ptrain_step, donate_argnums=(1,))
        peval_step = jax.jit(_peval_step)

        train_iter = iter(train_loader)
        eval_iter = iter(eval_loader)
        logging.info("Frozen policy config: %s", args.config_name)
        logging.info("Frozen policy params: %s", args.pretrained_params)
        logging.info("Token source: %s, layer: %d", args.token_source, probe_layer)
        logging.info(
            "Future token count expected by model: %d",
            train_config.model.future_tactile_segments * train_config.model.tactile_num_fingers,
        )
        logging.info("Train filter: %s", args.train_filter_path)
        logging.info("Eval filter: %s", args.eval_filter_path)

        progress = tqdm.tqdm(range(int(state.step), args.num_train_steps), total=args.num_train_steps, initial=int(state.step))
        for _ in progress:
            batch = next(train_iter)
            state, stats = ptrain_step(model_params, state, batch, train_rng)
            step = int(jax.device_get(state.step))
            if step == 1 or step % args.log_interval == 0:
                logging.info("step=%d train %s", step, _format_stats(stats))
                try:
                    eval_batch = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(eval_loader)
                    eval_batch = next(eval_iter)
                eval_stats = peval_step(model_params, state, eval_batch, train_rng)
                logging.info("step=%d eval  %s", step, _format_stats(eval_stats))
            if step % args.save_interval == 0:
                checkpointer.save(step, args=ocp.args.Composite(probe_state=ocp.args.PyTreeSave(state)))
        checkpointer.save(int(jax.device_get(state.step)), args=ocp.args.Composite(probe_state=ocp.args.PyTreeSave(state)))
        checkpointer.wait_until_finished()
        logging.info("Done. Probe checkpoints saved under %s", checkpoint_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
