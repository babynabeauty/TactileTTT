import dataclasses
import functools
import logging
import pathlib
import platform
from typing import Any
import openpi.training.checkpoints as _checkpoints
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
# import openpi.models.model as _model # bug, 添加这一行会莫名奇妙地退出，
# import openpi.models.model_tavla as _model # bug, 添加这一行会莫名奇妙地退出，
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from jax import ShapeDtypeStruct
from flax import traverse_util

def debug(host="0.0.0.0", port=5678, wait_client=True):
    import debugpy
    debugpy.listen((host, port))
    print(f"✅ debugpy is listening on {host}:{port}")
    if wait_client:
        print("⏳ Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("🟢 Debugger attached, execution will continue.")
        
def init_logging():
    """Custom logging format for better readability."""
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
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


# def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
#     """Loads and validates the weights. Returns a loaded subset of the weights."""
#     loaded_params = loader.load(params_shape)
#     at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

#     # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
#     return traverse_util.unflatten_dict(
#         {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
#     )
def _load_weights_and_validate(loader, params_shape):
    """Loads weights, ignores mismatched keys, and filters out ShapeDtypeStruct.
    Also prints which parameters failed to load and why."""

    # Stage-1 pretraining intentionally starts from random initialization. A
    # NoOp loader returns the shape tree unchanged, so there are no concrete
    # checkpoint arrays to merge into the initialized model.
    if isinstance(loader, _weight_loaders.NoOpWeightLoader):
        print(f"\n[INFO] No checkpoint loader configured; randomly initializing {len(traverse_util.flatten_dict(params_shape))} parameters.")
        return {}

    loaded_params = loader.load(params_shape)

    # Flatten both trees
    flat_shape = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)

    filtered = {}
    failed_keys = []

    for k, v in flat_shape.items():
        loaded_v = flat_loaded.get(k, None)
        reason = None

        if loaded_v is None:
            reason = "missing in checkpoint"
        elif isinstance(loaded_v, ShapeDtypeStruct):
            reason = "is ShapeDtypeStruct (not actual array)"
        elif not hasattr(loaded_v, "shape"):
            reason = "no shape attribute"
        elif loaded_v.shape != v.shape:
            reason = f"shape mismatch (ckpt={loaded_v.shape}, model={v.shape})"

        if reason is None:
            filtered[k] = loaded_v
        else:
            filtered[k] = v  # use initialized
            failed_keys.append((k, reason))

    # Remove any ShapeDtypeStruct accidentally left
    filtered = {
        k: v for k, v in filtered.items() if not isinstance(v, ShapeDtypeStruct)
    }

    total = len(flat_shape)
    loaded = total - len(failed_keys)
    print(f"\n[INFO] Loaded {loaded}/{total} parameters successfully.")
    if failed_keys:
        print("[WARN] The following parameters were not loaded:")
        for k, reason in failed_keys:
            print("   -", "/".join(map(str, k)), f"→ {reason}")

    return traverse_util.unflatten_dict(filtered)


def _preload_model_assets(config: _config.TrainConfig) -> None:
    if not getattr(config.model, "use_future_flow", True):
        return
    if getattr(config.model, "future_flow_source", "image") != "image":
        return
    flow_vae_name = getattr(config.model, "flow_vae_name", None)
    if flow_vae_name is None:
        return

    from openpi.models.pi0_latent_flow import preload_flow_vae

    logging.info("Preloading flow VAE '%s' before model initialization.", flow_vae_name)
    preload_flow_vae(flow_vae_name)


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    _preload_model_assets(config)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    has_loss_stats = hasattr(model, "compute_loss_with_stats")
    train_progress = jnp.asarray(state.step, dtype=jnp.float32) / jnp.asarray(max(config.num_train_steps - 1, 1), dtype=jnp.float32)

    @at.typecheck
    def loss_fn(
        model, rng, observation, actions
    ):
        if has_loss_stats:
            if getattr(model, "uses_train_progress", False):
                if getattr(model, "sequence_training", False):
                    chunked_loss, loss_stats = model.compute_sequence_loss_with_stats(
                        rng, observation, actions, train=True, train_progress=train_progress
                    )
                else:
                    chunked_loss, loss_stats = model.compute_loss_with_stats(
                        rng, observation, actions, train=True, train_progress=train_progress
                    )
            else:
                chunked_loss, loss_stats = model.compute_loss_with_stats(rng, observation, actions, train=True)
            # Keep aux stats scalar-friendly for logging.
            reduced_loss_stats = jax.tree.map(jnp.mean, loss_stats)
            return jnp.mean(chunked_loss), reduced_loss_stats
        if getattr(model, "uses_train_progress", False):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True, train_progress=train_progress)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    if has_loss_stats:
        (loss, loss_stats), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )
    else:
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
        loss_stats = {}

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    if hasattr(model, "update_target_encoder"):
        model.update_target_encoder(train_progress)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(loss_stats)
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    has_loss_stats = hasattr(model, "compute_loss_with_stats")
    eval_progress = jnp.asarray(state.step, dtype=jnp.float32) / jnp.asarray(
        max(config.num_train_steps - 1, 1), dtype=jnp.float32
    )
    eval_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    if has_loss_stats:
        if getattr(model, "uses_train_progress", False):
            if getattr(model, "sequence_training", False):
                chunked_loss, loss_stats = model.compute_sequence_loss_with_stats(
                    eval_rng, observation, actions, train=False, train_progress=eval_progress
                )
            else:
                chunked_loss, loss_stats = model.compute_loss_with_stats(
                    eval_rng, observation, actions, train=False, train_progress=eval_progress
                )
        else:
            chunked_loss, loss_stats = model.compute_loss_with_stats(eval_rng, observation, actions, train=False)
        info = {"loss": jnp.mean(chunked_loss)}
        info.update(jax.tree.map(jnp.mean, loss_stats))
        return info

    if getattr(model, "uses_train_progress", False):
        chunked_loss = model.compute_loss(eval_rng, observation, actions, train=False, train_progress=eval_progress)
    else:
        chunked_loss = model.compute_loss(eval_rng, observation, actions, train=False)
    return {"loss": jnp.mean(chunked_loss)}


@at.typecheck
def grad_stats_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch,
) -> dict[str, at.Array]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    if not hasattr(model, "compute_grad_stats"):
        raise ValueError("Model does not implement compute_grad_stats")

    return model.compute_grad_stats(train_rng, observation, actions, train=True)


def _with_filter_path(data_factory: _config.DataConfigFactory, filter_path: str | None) -> _config.DataConfigFactory:
    if filter_path is None:
        return data_factory
    base_config = data_factory.base_config or _config.DataConfig()
    return dataclasses.replace(
        data_factory,
        base_config=dataclasses.replace(base_config, filter_dict_path=str(pathlib.Path(filter_path).expanduser())),
    )


def _make_eval_config(config: _config.TrainConfig) -> _config.TrainConfig:
    data_factory = config.data
    if config.eval_repo_id is not None:
        data_factory = dataclasses.replace(data_factory, repo_id=config.eval_repo_id)
    if config.eval_asset_id is not None or config.eval_assets_dir is not None:
        data_factory = dataclasses.replace(
            data_factory,
            assets=dataclasses.replace(
                data_factory.assets,
                asset_id=config.eval_asset_id if config.eval_asset_id is not None else data_factory.assets.asset_id,
                assets_dir=(
                    config.eval_assets_dir if config.eval_assets_dir is not None else data_factory.assets.assets_dir
                ),
            ),
        )
    data_factory = _with_filter_path(data_factory, config.eval_filter_path)
    return dataclasses.replace(
        config,
        data=data_factory,
        batch_size=config.eval_batch_size or config.batch_size,
        num_workers=config.eval_num_workers if config.eval_num_workers is not None else config.num_workers,
        wandb_enabled=False,
    )


def _fmt_val(v):
    try:
        return f"{float(v):.8f}"
    except (TypeError, ValueError):
        return str(v)


def _run_validation(
    *,
    step: int,
    eval_num_batches: int,
    eval_iter,
    peval_step,
    eval_rng,
    train_state: training_utils.TrainState,
) -> dict[str, Any]:
    eval_infos = []
    for _ in range(eval_num_batches):
        eval_batch = next(eval_iter)
        eval_infos.append(peval_step(eval_rng, train_state, eval_batch))
    stacked_infos = common_utils.stack_forest(eval_infos)
    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
    prefixed = {f"eval/{key}": value for key, value in reduced_info.items()}
    logging.info("Eval step %d: %s", step, ", ".join(f"{k}={_fmt_val(v)}" for k, v in prefixed.items()))
    return prefixed


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    if config.eval_interval > 0:
        eval_batch_size = config.eval_batch_size or config.batch_size
        if eval_batch_size % jax.device_count() != 0:
            raise ValueError(
                f"Eval batch size {eval_batch_size} must be divisible by the number of devices {jax.device_count()}."
            )
        if config.eval_num_batches <= 0:
            raise ValueError("--eval-num-batches must be positive when --eval-interval is enabled.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    train_data_config = config
    if config.train_filter_path is not None:
        train_data_config = dataclasses.replace(config, data=_with_filter_path(config.data, config.train_filter_path))
        logging.info("Train episode filter enabled: %s", config.train_filter_path)

    data_loader = _data_loader.create_data_loader(
        train_data_config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check. Some lightweight pretraining
    # configs intentionally omit images.
    if batch[0].images:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
    else:
        logging.info("Skipping camera view logging: batch has no images.")

    eval_config = None
    eval_iter = None
    if config.eval_interval > 0:
        eval_config = _make_eval_config(config)
        logging.info(
            "Validation enabled: interval=%d num_batches=%d repo=%s filter=%s batch_size=%d",
            config.eval_interval,
            config.eval_num_batches,
            eval_config.data.repo_id,
            config.eval_filter_path,
            eval_config.batch_size,
        )
        eval_loader = _data_loader.create_data_loader(
            eval_config,
            sharding=data_sharding,
            shuffle=False,
        )
        eval_iter = iter(eval_loader)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    # 打印可训练参数名
    trainable_params = train_state.params.filter(config.trainable_filter)
    flat = traverse_util.flatten_dict(trainable_params.to_pure_dict())
    
    # ================= 新增开始 =================
    all_params_flat = traverse_util.flatten_dict(train_state.params.to_pure_dict())
    
    total_params_cnt = sum(np.prod(v.shape) for v in all_params_flat.values() if hasattr(v, "shape"))
    trainable_params_cnt = sum(np.prod(v.shape) for v in flat.values() if hasattr(v, "shape"))
    
    logging.info(f"Total parameters: {total_params_cnt:,}")
    logging.info(f"Trainable parameters: {trainable_params_cnt:,} ({trainable_params_cnt/total_params_cnt*100:.2f}%)")
    # ================= 新增结束 =================

    logging.info("Trainable parameters:")
    for k in flat:
        logging.info(f"  {'/'.join(map(str, k))}  shape={flat[k].shape}")
        
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = None
    if eval_config is not None:
        peval_step = jax.jit(
            functools.partial(eval_step, eval_config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
    model_for_stats = nnx.merge(train_state.model_def, train_state.params)
    has_grad_stats = hasattr(model_for_stats, "compute_grad_stats")
    del model_for_stats

    pgrad_stats = None
    if has_grad_stats:
        pgrad_stats = jax.jit(
            functools.partial(grad_stats_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )

    start_step = int(train_state.step)
    if config.eval_interval > 0 and config.eval_at_start:
        eval_step_num = int(jax.device_get(train_state.step))
        eval_info = _run_validation(
            step=eval_step_num,
            eval_num_batches=config.eval_num_batches,
            eval_iter=eval_iter,
            peval_step=peval_step,
            eval_rng=train_rng,
            train_state=train_state,
        )
        wandb.log(eval_info, step=eval_step_num)

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            if pgrad_stats is not None:
                grad_stats = pgrad_stats(train_rng, train_state, batch)
                grad_stats = jax.device_get(jax.tree.map(jnp.mean, grad_stats))
                reduced_info.update(grad_stats)

            info_str = ", ".join(f"{k}={_fmt_val(v)}" for k, v in reduced_info.items())
            logging.info("Step %d: %s", step, info_str)
            display_loss = reduced_info.get("loss/total", reduced_info.get("loss"))
            if display_loss is not None:
                pbar.set_postfix(loss=_fmt_val(display_loss))
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        completed_step = int(jax.device_get(train_state.step))
        should_eval = (
            config.eval_interval > 0
            and completed_step > start_step
            and (completed_step % config.eval_interval == 0 or completed_step == config.num_train_steps)
        )
        if should_eval:
            eval_info = _run_validation(
                step=completed_step,
                eval_num_batches=config.eval_num_batches,
                eval_iter=eval_iter,
                peval_step=peval_step,
                eval_rng=train_rng,
                train_state=train_state,
            )
            wandb.log(eval_info, step=completed_step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    # debug()
    main(_config.cli())
