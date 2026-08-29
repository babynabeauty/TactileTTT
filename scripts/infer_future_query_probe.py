import dataclasses
import json
import logging
import pathlib

import einops
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tyro
from flax import nnx

from openpi.models import gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.pi0_tavla import make_attn_mask
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding

import train_future_query_probe as _probe_train


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_latent_flow_noise"
    repo_id: str | None = "llly/all_0409_stage_flow"
    probe_checkpoint_dir: str = "checkpoints/future_query_probe/pi0_latent_flow_noise/probe_from_30k_2gpu"
    pretrained_params: str | None = None
    image: str | None = None
    wrist_image: str | None = None
    state_npy: str | None = None
    effort_npy: str | None = None
    prompt: str = ""
    output_dir: str = "outputs/future_query_probe_infer"
    batch_size: int = 8
    max_batches: int = 16
    num_workers: int = 0
    shuffle: bool = False
    probe_layer: int | None = None
    patch_size: int = 16
    decoder_dim: int | None = None
    decoder_depth: int = 2
    decoder_heads: int = 8
    seed: int = 0
    skip_norm: bool = False


def _latest_checkpoint_step(path: pathlib.Path) -> int:
    steps = [int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"No numeric checkpoint directories found under {path}.")
    return max(steps)


def _load_probe_checkpoint_args(ckpt_dir: pathlib.Path, step: int) -> dict:
    args_path = ckpt_dir / f"{step:08d}" / "args.json"
    if not args_path.exists():
        logging.warning("Probe checkpoint args not found at %s; using CLI/default inference args.", args_path)
        return {}
    return json.loads(args_path.read_text(encoding="utf-8"))


def _read_rgb(path: str) -> np.ndarray:
    image = iio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA image at {path}, got shape {image.shape}.")
    return image.astype(np.uint8)


def _as_batched(x: np.ndarray) -> np.ndarray:
    return x[None, ...]


def _build_observation(args: Args, config: _config.TrainConfig) -> _model.Observation:
    if args.image is None or args.state_npy is None or args.effort_npy is None:
        raise ValueError("Manual mode requires --image, --state-npy, and --effort-npy.")
    state = np.asarray(np.load(args.state_npy), dtype=np.float32)
    effort = np.asarray(np.load(args.effort_npy), dtype=np.float32)
    if state.ndim != 1:
        raise ValueError(f"state_npy must have shape [state_dim], got {state.shape}.")
    if effort.ndim != 2:
        raise ValueError(f"effort_npy must have shape [history_frames, effort_dim], got {effort.shape}.")

    data = {
        "image": {"base_0_rgb": _read_rgb(args.image)},
        "image_mask": {"base_0_rgb": np.asarray(True)},
        "state": state,
        "effort": effort,
        "prompt": args.prompt,
    }
    if args.wrist_image is not None:
        data["image"]["left_wrist_0_rgb"] = _read_rgb(args.wrist_image)
        data["image_mask"]["left_wrist_0_rgb"] = np.asarray(True)

    data_config = config.data.create(config.assets_dirs, config.model)
    transforms = []
    if not args.skip_norm:
        transforms.append(_transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.extend(data_config.model_transforms.inputs)
    data = _transforms.compose(transforms)(data)

    batched = {}
    for key, value in data.items():
        if key == "image":
            batched[key] = {image_key: _as_batched(image_value) for image_key, image_value in value.items()}
        elif key == "image_mask":
            batched[key] = {image_key: _as_batched(mask_value) for image_key, mask_value in value.items()}
        elif key in ("state", "effort", "tokenized_prompt", "tokenized_prompt_mask"):
            batched[key] = _as_batched(value)
        else:
            batched[key] = value
    return _model.Observation.from_dict(batched)


def _extract_probe_hiddens_for_inference(
    model,
    rng,
    observation: _model.Observation,
    *,
    probe_layer: int,
) -> tuple[jax.Array, jax.Array]:
    processed = _model.preprocess_observation(rng, observation, train=False, effort_type=model.effort_type)
    history_effort, _ = model._split_effort(processed, require_future=False, dtype=jnp.float32)

    batch_size = processed.state.shape[0]
    zero_actions = jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=jnp.float32)
    time = jnp.ones((batch_size,), dtype=jnp.float32)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(processed)
    student_tokens, student_mask, student_ar_mask, student_adarms, *_ = model.embed_student_suffix(
        processed,
        history_effort,
        zero_actions,
        time,
        train=False,
        noise_rng=None,
    )

    prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
    student_attn = make_attn_mask(student_mask, student_ar_mask)
    student_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
    student_to_prefix = jnp.logical_and(student_to_prefix, student_mask[:, :, None])
    prefix_row = jnp.concatenate(
        [prefix_attn, jnp.zeros((batch_size, prefix_tokens.shape[1], student_tokens.shape[1]), dtype=jnp.bool_)],
        axis=-1,
    )
    student_row = jnp.concatenate([student_to_prefix, student_attn], axis=-1)
    full_attn = jnp.concatenate([prefix_row, student_row], axis=1)

    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
    student_positions = prefix_len + jnp.cumsum(student_mask, axis=-1) - 1
    positions = jnp.concatenate([prefix_positions, student_positions], axis=1)

    (_, selected_layers), _ = model.PaliGemma.llm(
        [prefix_tokens, student_tokens, None],
        mask=full_attn,
        positions=positions,
        adarms_cond=[None, student_adarms, None],
        return_layer_indices=(int(probe_layer),),
    )
    student_hidden = selected_layers[0][1]
    force_hidden = student_hidden[:, 2:3, :].astype(jnp.float32)
    flow_hidden = student_hidden[:, 3 : 3 + model.flow_token_count, :].astype(jnp.float32)
    return force_hidden, flow_hidden


def _predict_and_target(
    model_def,
    model_params,
    probe_def,
    probe_state,
    observation: _model.Observation,
    rng,
    *,
    probe_layer: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    model = nnx.merge(model_def, model_params)
    model.eval()
    probe_model = nnx.merge(probe_def, probe_state.params)
    processed = _model.preprocess_observation(rng, observation, train=False, effort_type=model.effort_type)
    processed = _probe_train._restore_aux_images(observation, processed)
    if processed.flow_img is None:
        raise ValueError("Flow image metrics require observation.flow_img in the eval batch.")
    history_effort, true_force = model._split_effort(processed, require_future=True, dtype=jnp.float32)

    batch_size = processed.state.shape[0]
    zero_actions = jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=jnp.float32)
    time = jnp.ones((batch_size,), dtype=jnp.float32)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(processed)
    student_tokens, student_mask, student_ar_mask, student_adarms, *_ = model.embed_student_suffix(
        processed,
        history_effort,
        zero_actions,
        time,
        train=False,
        noise_rng=None,
    )

    prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
    student_attn = make_attn_mask(student_mask, student_ar_mask)
    student_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
    student_to_prefix = jnp.logical_and(student_to_prefix, student_mask[:, :, None])
    prefix_row = jnp.concatenate(
        [prefix_attn, jnp.zeros((batch_size, prefix_tokens.shape[1], student_tokens.shape[1]), dtype=jnp.bool_)],
        axis=-1,
    )
    student_row = jnp.concatenate([student_to_prefix, student_attn], axis=-1)
    full_attn = jnp.concatenate([prefix_row, student_row], axis=1)

    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
    student_positions = prefix_len + jnp.cumsum(student_mask, axis=-1) - 1
    positions = jnp.concatenate([prefix_positions, student_positions], axis=1)

    (_, selected_layers), _ = model.PaliGemma.llm(
        [prefix_tokens, student_tokens, None],
        mask=full_attn,
        positions=positions,
        adarms_cond=[None, student_adarms, None],
        return_layer_indices=(int(probe_layer),),
    )
    student_hidden = selected_layers[0][1]
    force_hidden = student_hidden[:, 2:3, :].astype(jnp.float32)
    flow_hidden = student_hidden[:, 3 : 3 + model.flow_token_count, :].astype(jnp.float32)
    pred_force, pred_flow = probe_model(force_hidden, flow_hidden)
    return pred_force, true_force.astype(jnp.float32), pred_flow, jnp.asarray(processed.flow_img, dtype=jnp.float32)


def _unnormalize_effort(config: _config.TrainConfig, force: np.ndarray, *, skip_norm: bool) -> np.ndarray:
    if skip_norm:
        return force
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None or "effort" not in data_config.norm_stats:
        return force
    return _transforms.Unnormalize(
        {"effort": data_config.norm_stats["effort"]},
        use_quantiles=data_config.use_quantile_norm,
    )({"effort": force})["effort"]


def _flow_to_uint8(flow: np.ndarray) -> np.ndarray:
    return np.clip((flow + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _true_future_effort(args: Args, config: _config.TrainConfig) -> np.ndarray | None:
    effort = np.asarray(np.load(args.effort_npy), dtype=np.float32)
    if effort.ndim != 2 or effort.shape[0] <= config.model.action_horizon:
        return None
    return effort[-config.model.action_horizon :, :]


def _plot_force_curves(pred_force: np.ndarray, true_force: np.ndarray | None, output_path: pathlib.Path) -> None:
    dim = pred_force.shape[-1]
    fig, axes = plt.subplots(dim, 1, figsize=(10, max(2.0 * dim, 4.0)), sharex=True)
    if dim == 1:
        axes = [axes]
    t = np.arange(pred_force.shape[0])
    for i, ax in enumerate(axes):
        ax.plot(t, pred_force[:, i], label="pred", linewidth=1.8)
        if true_force is not None and i < true_force.shape[-1]:
            ax.plot(t[: true_force.shape[0]], true_force[:, i], label="true", linewidth=1.4, alpha=0.8)
        ax.set_ylabel(f"F{i}")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("future step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _evaluate_lerobot_dataset(
    args: Args,
    config: _config.TrainConfig,
    model_def,
    model_params,
    probe_def,
    probe_state,
    infer_rng,
    *,
    probe_layer: int,
    output_dir: pathlib.Path,
) -> None:
    data_config_factory = dataclasses.replace(config.data, repo_id=args.repo_id)
    eval_config = dataclasses.replace(
        config,
        data=data_config_factory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    mesh = sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = _data_loader.create_data_loader(
        eval_config,
        sharding=data_sharding,
        shuffle=args.shuffle,
        num_batches=args.max_batches,
    )

    @jax.jit
    def predict_batch(frozen_params, state, batch_obs):
        return _predict_and_target(
            model_def,
            frozen_params,
            probe_def,
            state,
            batch_obs,
            infer_rng,
            probe_layer=probe_layer,
        )

    pred_batches = []
    true_batches = []
    flow_batches = []
    true_flow_batches = []
    for batch_idx, batch in zip(range(args.max_batches), loader, strict=False):
        logging.info("Evaluating batch %d/%d", batch_idx + 1, args.max_batches)
        with sharding.set_mesh(mesh):
            pred_force, true_force, pred_flow, true_flow = predict_batch(model_params, probe_state, batch[0])
        pred_batches.append(np.asarray(jax.device_get(pred_force)))
        true_batches.append(np.asarray(jax.device_get(true_force)))
        flow_batches.append(np.asarray(jax.device_get(pred_flow)))
        true_flow_batches.append(np.asarray(jax.device_get(true_flow)))

    pred_norm = np.concatenate(pred_batches, axis=0)
    true_norm = np.concatenate(true_batches, axis=0)
    pred_raw = _unnormalize_effort(eval_config, pred_norm, skip_norm=args.skip_norm)
    true_raw = _unnormalize_effort(eval_config, true_norm, skip_norm=args.skip_norm)
    pred_flow = np.concatenate(flow_batches, axis=0)
    true_flow = np.concatenate(true_flow_batches, axis=0)

    err = pred_raw - true_raw
    flow_err = pred_flow.astype(np.float32) - true_flow.astype(np.float32)
    mae_per_dim = np.mean(np.abs(err), axis=(0, 1))
    rmse_per_dim = np.sqrt(np.mean(np.square(err), axis=(0, 1)))
    flow_mse = float(np.mean(np.square(flow_err)))
    flow_rmse = float(np.sqrt(flow_mse))
    flow_mae = float(np.mean(np.abs(flow_err)))
    flow_psnr = float(20.0 * np.log10(2.0 / (flow_rmse + 1e-12)))
    metrics = {
        "repo_id": args.repo_id,
        "num_samples": int(pred_raw.shape[0]),
        "action_horizon": int(pred_raw.shape[1]),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "mae_per_dim": mae_per_dim.tolist(),
        "rmse_per_dim": rmse_per_dim.tolist(),
        "flow_mse": flow_mse,
        "flow_rmse": flow_rmse,
        "flow_mae": flow_mae,
        "flow_psnr_db": flow_psnr,
    }

    np.save(output_dir / "pred_force.npy", pred_raw)
    np.save(output_dir / "true_force.npy", true_raw)
    np.save(output_dir / "pred_force_normalized.npy", pred_norm)
    np.save(output_dir / "true_force_normalized.npy", true_norm)
    np.save(output_dir / "pred_flow.npy", pred_flow)
    np.save(output_dir / "true_flow.npy", true_flow)
    iio.imwrite(output_dir / "pred_flow_sample0.png", _flow_to_uint8(pred_flow[0]))
    iio.imwrite(output_dir / "true_flow_sample0.png", _flow_to_uint8(true_flow[0]))
    _plot_force_curves(pred_raw[0], true_raw[0], output_dir / "force_curve_sample0.png")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.info("MAE %.6f, RMSE %.6f", metrics["mae"], metrics["rmse"])
    logging.info("MAE per dim: %s", np.array2string(mae_per_dim, precision=6))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))

    ckpt_dir = pathlib.Path(args.probe_checkpoint_dir).resolve()
    step = _latest_checkpoint_step(ckpt_dir)
    ckpt_args = _load_probe_checkpoint_args(ckpt_dir, step)

    base_config = _config.get_config(args.config_name)
    config = dataclasses.replace(base_config, batch_size=args.batch_size, seed=args.seed, num_workers=args.num_workers)
    if not isinstance(config.model, pi0_config.Pi0LatentFlowConfig):
        raise ValueError(f"Config {args.config_name!r} is not a Pi0LatentFlow config.")
    if ckpt_args.get("config_name") is not None and ckpt_args["config_name"] != args.config_name:
        logging.warning(
            "Probe checkpoint was trained with config_name=%s but inference uses config_name=%s.",
            ckpt_args["config_name"],
            args.config_name,
        )

    probe_layer = int(
        args.probe_layer
        if args.probe_layer is not None
        else ckpt_args.get("probe_layer")
        if ckpt_args.get("probe_layer") is not None
        else config.model.distill_layer_indices[-1]
    )
    pretrained_params = args.pretrained_params
    if pretrained_params is None:
        pretrained_params = ckpt_args.get("pretrained_params")
    patch_size = int(ckpt_args.get("patch_size", args.patch_size))
    decoder_dim = ckpt_args.get("decoder_dim", args.decoder_dim)
    decoder_depth = int(ckpt_args.get("decoder_depth", args.decoder_depth))
    decoder_heads = int(ckpt_args.get("decoder_heads", args.decoder_heads))
    logging.info(
        "Using probe settings: layer=%s, pretrained_params=%s, patch_size=%s, decoder_dim=%s, "
        "decoder_depth=%s, decoder_heads=%s.",
        probe_layer,
        pretrained_params,
        patch_size,
        decoder_dim,
        decoder_depth,
        decoder_heads,
    )

    mesh = sharding.make_mesh(num_fsdp_devices=1)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    rng = jax.random.key(args.seed)
    rng, model_rng, probe_rng, infer_rng = jax.random.split(rng, 4)
    model_def, model_params = _probe_train._init_frozen_model(
        config,
        model_rng,
        pretrained_params=pretrained_params,
    )
    model_params = jax.device_put(model_params, replicated_sharding)

    student_width = int(_gemma.get_config(config.model.action_expert_variant).width)
    probe = _probe_train.FutureQueryProbe(
        student_width=student_width,
        flow_token_count=int(config.model.flow_token_count),
        action_horizon=int(config.model.action_horizon),
        effort_dim=int(config.model.effort_dim if config.model.effort_dim is not None else config.model.effort_dim_in),
        decoder_dim=int(decoder_dim or student_width),
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        image_size=224,
        patch_size=patch_size,
        rngs=nnx.Rngs(probe_rng),
    )
    probe_def, probe_params = nnx.split(probe)
    tx = optax.adamw(1e-4)
    probe_state = _probe_train.ProbeTrainState(step=0, params=probe_params, opt_state=tx.init(probe_params), tx=tx)
    probe_state = _probe_train._restore_probe_checkpoint(ckpt_dir, probe_state)
    probe_state = jax.device_put(probe_state, replicated_sharding)
    logging.info("Loaded probe checkpoint step %08d from %s", step, ckpt_dir)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.repo_id is not None:
        _evaluate_lerobot_dataset(
            args,
            config,
            model_def,
            model_params,
            probe_def,
            probe_state,
            infer_rng,
            probe_layer=probe_layer,
            output_dir=output_dir,
        )
        logging.info("Saved predictions and metrics to %s", output_dir.resolve())
        return

    observation = _build_observation(args, config)
    observation = jax.device_put(observation, replicated_sharding)

    @jax.jit
    def predict(frozen_params, state, obs):
        model = nnx.merge(model_def, frozen_params)
        model.eval()
        probe_model = nnx.merge(probe_def, state.params)
        force_hidden, flow_hidden = _extract_probe_hiddens_for_inference(
            model,
            infer_rng,
            obs,
            probe_layer=probe_layer,
        )
        return probe_model(force_hidden, flow_hidden)

    with sharding.set_mesh(mesh):
        pred_force, pred_flow = predict(model_params, probe_state, observation)
    pred_force = np.asarray(jax.device_get(pred_force[0]))
    pred_flow = np.asarray(jax.device_get(pred_flow[0]))
    pred_force_raw = _unnormalize_effort(config, pred_force, skip_norm=args.skip_norm)
    true_force_raw = _true_future_effort(args, config)

    np.save(output_dir / "pred_force_normalized.npy", pred_force)
    np.save(output_dir / "pred_force.npy", pred_force_raw)
    np.save(output_dir / "pred_flow.npy", pred_flow)
    iio.imwrite(output_dir / "pred_flow.png", _flow_to_uint8(pred_flow))
    if true_force_raw is not None:
        np.save(output_dir / "true_force.npy", true_force_raw)
    _plot_force_curves(pred_force_raw, true_force_raw, output_dir / "force_curve.png")
    logging.info("Saved predictions to %s", output_dir.resolve())
    logging.info("pred_force shape=%s, pred_flow shape=%s", pred_force_raw.shape, pred_flow.shape)


if __name__ == "__main__":
    main(tyro.cli(Args))
