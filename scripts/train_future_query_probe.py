import dataclasses
import json
import logging
import pathlib
import platform
import shutil

import einops
import flax.nnx as nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.pi0_tavla import make_attn_mask
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


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


def _posemb_1d_from_grid(embed_dim: int, pos: at.Float[at.Array, "n"]) -> at.Float[at.Array, "n d"]:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, got {embed_dim}.")
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = jnp.einsum("n,d->nd", pos.reshape(-1), omega)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


def _posemb_2d(embed_dim: int, grid_size: int) -> at.Float[at.Array, "n d"]:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, got {embed_dim}.")
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing="xy")
    emb_h = _posemb_1d_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _posemb_1d_from_grid(embed_dim // 2, grid[1].reshape(-1))
    return jnp.concatenate([emb_h, emb_w], axis=1)


class ProbeTransformerBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        if dim % num_heads != 0:
            raise ValueError(f"decoder dim ({dim}) must be divisible by decoder heads ({num_heads}).")
        self.num_heads = int(num_heads)
        self.norm1 = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.qkv = nnx.Linear(dim, dim * 3, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.ff_in = nnx.Linear(dim, dim * 4, rngs=rngs)
        self.ff_out = nnx.Linear(dim * 4, dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n d"]:
        residual = x
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        head_dim = q.shape[-1] // self.num_heads
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * (head_dim**-0.5)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        x = jnp.einsum("bhij,bhjd->bhid", attn, v)
        x = einops.rearrange(x, "b h n d -> b n (h d)")
        x = residual + self.proj(x)

        residual = x
        x_norm = self.norm2(x)
        x_ff = self.ff_in(x_norm)
        x_ff = jax.nn.gelu(x_ff)
        x_ff = self.ff_out(x_ff)
        return residual + x_ff


class ForceProbeMLP(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_horizon: int, effort_dim: int, rngs: nnx.Rngs):
        self.action_horizon = int(action_horizon)
        self.effort_dim = int(effort_dim)
        self.in_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.mid_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, self.action_horizon * self.effort_dim, rngs=rngs)

    def __call__(self, force_hidden: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b t e"]:
        x = einops.rearrange(force_hidden, "b n d -> b (n d)")
        x = jax.nn.gelu(self.in_proj(x))
        x = jax.nn.gelu(self.mid_proj(x))
        x = self.out_proj(x)
        return einops.rearrange(x, "b (t e) -> b t e", t=self.action_horizon, e=self.effort_dim)


class FlowProbeViT(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        flow_token_count: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        image_size: int,
        patch_size: int,
        rngs: nnx.Rngs,
    ):
        if image_size % patch_size != 0:
            raise ValueError(f"image size ({image_size}) must be divisible by patch size ({patch_size}).")
        grid_size = image_size // patch_size
        if decoder_dim % 2 != 0:
            raise ValueError(f"decoder dim must be even for sinusoidal position embeddings, got {decoder_dim}.")

        self.flow_token_count = int(flow_token_count)
        self.decoder_dim = int(decoder_dim)
        self.decoder_depth = int(decoder_depth)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_patch_tokens = int(grid_size * grid_size)

        self.input_proj = nnx.Linear(input_dim, decoder_dim, rngs=rngs)
        self.mask_token = nnx.Param(0.02 * jax.random.normal(rngs.params(), (decoder_dim,), dtype=jnp.float32))
        self.blocks = [
            ProbeTransformerBlock(decoder_dim, decoder_heads, rngs=rngs)
            for _ in range(self.decoder_depth)
        ]
        self.norm = nnx.LayerNorm(num_features=decoder_dim, rngs=rngs)
        self.pred = nnx.Linear(decoder_dim, patch_size * patch_size * 3, rngs=rngs)

    def _position_embedding(self) -> at.Float[at.Array, "1 n d"]:
        grid_size = self.image_size // self.patch_size
        return jnp.concatenate(
            [
                _posemb_1d_from_grid(
                    self.decoder_dim,
                    jnp.arange(self.flow_token_count, dtype=jnp.float32),
                ),
                _posemb_2d(self.decoder_dim, grid_size),
            ],
            axis=0,
        )[None, :, :]

    def __call__(self, flow_hidden: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b h w c"]:
        decoded = self.input_proj(flow_hidden)
        mask_tokens = jnp.asarray(self.mask_token.value, dtype=decoded.dtype)
        mask_tokens = jnp.broadcast_to(
            mask_tokens[None, None, :],
            (decoded.shape[0], self.num_patch_tokens, self.decoder_dim),
        )
        x = jnp.concatenate([decoded, mask_tokens], axis=1)
        x = x + jnp.asarray(self._position_embedding(), dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
        patch_hidden = self.norm(x[:, -self.num_patch_tokens :, :])
        patches = self.pred(patch_hidden)
        grid_size = self.image_size // self.patch_size
        return einops.rearrange(
            patches,
            "b (h w) (ph pw c) -> b (h ph) (w pw) c",
            h=grid_size,
            w=grid_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=3,
        )


class FutureQueryProbe(nnx.Module):
    def __init__(
        self,
        student_width: int,
        flow_token_count: int,
        action_horizon: int,
        effort_dim: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
        image_size: int,
        patch_size: int,
        rngs: nnx.Rngs,
    ):
        self.force_head = ForceProbeMLP(
            input_dim=student_width,
            hidden_dim=decoder_dim,
            action_horizon=action_horizon,
            effort_dim=effort_dim,
            rngs=rngs,
        )
        self.flow_head = FlowProbeViT(
            input_dim=student_width,
            flow_token_count=flow_token_count,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            image_size=image_size,
            patch_size=patch_size,
            rngs=rngs,
        )

    def __call__(
        self,
        force_hidden: at.Float[at.Array, "b 1 d"],
        flow_hidden: at.Float[at.Array, "b n d"],
    ) -> tuple[at.Float[at.Array, "b t e"], at.Float[at.Array, "b h w c"]]:
        return self.force_head(force_hidden), self.flow_head(flow_hidden)


@struct.dataclass
class ProbeTrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)


@dataclasses.dataclass(frozen=True)
class Args:
    config_name: str = "pi0_latent_flow_noise"
    exp_name: str = tyro.MISSING
    # Frozen Pi0LatentFlow params checkpoint. If omitted, uses the selected config's weight_loader.
    pretrained_params: str | None = None
    num_train_steps: int = 30_000
    batch_size: int = 16
    learning_rate: float = 1e-4
    seed: int = 0
    log_interval: int = 50
    save_interval: int = 1_000
    image_log_interval: int = 500
    num_workers: int = 0
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    probe_layer: int | None = None
    patch_size: int = 16
    decoder_dim: int | None = None
    decoder_depth: int = 2
    decoder_heads: int = 8
    force_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0
    checkpoint_base_dir: str = "checkpoints/future_query_probe"


def _checkpoint_dir(args: Args) -> pathlib.Path:
    return (pathlib.Path(args.checkpoint_base_dir) / args.config_name / args.exp_name).resolve()


def _prepare_checkpoint_dir(path: pathlib.Path, *, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("Cannot use --overwrite and --resume at the same time.")
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(f"Checkpoint directory {path} already exists. Use --overwrite or --resume.")
    path.mkdir(parents=True, exist_ok=True)


def _latest_checkpoint_step(path: pathlib.Path) -> int | None:
    steps = [int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(steps) if steps else None


def _save_probe_checkpoint(path: pathlib.Path, state: ProbeTrainState, args: Args) -> None:
    step = int(jax.device_get(state.step))
    step_dir = path / f"{step:08d}"
    item = {
        "step": np.asarray(step, dtype=np.int64),
        "params": state.params.to_pure_dict(),
        "opt_state": state.opt_state,
    }
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(step_dir, item)
    (step_dir / "args.json").write_text(json.dumps(dataclasses.asdict(args), indent=2, sort_keys=True))


def _restore_probe_checkpoint(path: pathlib.Path, state: ProbeTrainState) -> ProbeTrainState:
    step = _latest_checkpoint_step(path)
    if step is None:
        logging.info("No probe checkpoint found under %s; starting from scratch.", path)
        return state
    step_dir = path / f"{step:08d}"
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(step_dir)
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding_spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        restored = ckptr.restore(
            step_dir,
            ocp.args.PyTreeRestore(
                item=metadata,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding_spec),
                    metadata,
                ),
            ),
        )
    params = state.params
    params.replace_by_pure_dict(restored["params"])
    logging.info("Restored probe checkpoint from %s.", step_dir)
    return dataclasses.replace(
        state,
        step=jnp.asarray(int(restored["step"]), dtype=jnp.int64),
        params=params,
        opt_state=restored["opt_state"],
    )


def _load_weights_and_validate(loader, params: at.Params) -> at.Params:
    loaded_params = loader.load(params)
    flat_ref = traverse_util.flatten_dict(params)
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    filtered = {}
    failed = []
    for key, ref_value in flat_ref.items():
        loaded_value = flat_loaded.get(key)
        reason = None
        if loaded_value is None:
            reason = "missing in checkpoint"
        elif not hasattr(loaded_value, "shape"):
            reason = "no shape attribute"
        elif loaded_value.shape != ref_value.shape:
            reason = f"shape mismatch (ckpt={loaded_value.shape}, model={ref_value.shape})"
        if reason is None:
            filtered[key] = loaded_value
        else:
            filtered[key] = ref_value
            failed.append((key, reason))

    logging.info("Loaded %d/%d frozen model parameters.", len(flat_ref) - len(failed), len(flat_ref))
    if failed:
        logging.warning("Frozen model parameters left at initialization: %d", len(failed))
        for key, reason in failed[:20]:
            logging.warning("  %s -> %s", "/".join(key), reason)
        if len(failed) > 20:
            logging.warning("  ... %d more", len(failed) - 20)
    return traverse_util.unflatten_dict(filtered)


def _init_frozen_model(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    *,
    pretrained_params: str | None,
) -> tuple[nnx.GraphDef, nnx.State]:
    model = config.model.create(rng)
    graphdef, state = nnx.split(model)
    weight_loader = (
        _weight_loaders.CheckpointWeightLoader(pretrained_params)
        if pretrained_params is not None
        else config.weight_loader
    )
    logging.info(
        "Loading frozen backbone params from %s.",
        pretrained_params if pretrained_params is not None else f"config weight_loader: {weight_loader}",
    )
    loaded = _load_weights_and_validate(weight_loader, state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    return graphdef, state


def _restore_aux_images(
    observation: _model.Observation,
    processed: _model.Observation,
) -> _model.Observation:
    return processed.replace(
        flow_img=observation.flow_img,
        wrist_flow_img=observation.wrist_flow_img,
        future_rgb_img=observation.future_rgb_img,
        future_wrist_rgb_img=observation.future_wrist_rgb_img,
    )


def _extract_student_future_hiddens(
    model,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    probe_layer: int,
) -> tuple[
    at.Float[at.Array, "b 1 d"],
    at.Float[at.Array, "b n d"],
    at.Float[at.Array, "b t e"],
    at.Float[at.Array, "b h w c"],
]:
    processed = _model.preprocess_observation(rng, observation, train=False, effort_type=model.effort_type)
    processed = _restore_aux_images(observation, processed)
    if processed.flow_img is None:
        raise ValueError("Future query probe requires observation.flow_img as the head-view flow target.")

    history_effort, future_effort = model._split_effort(processed, require_future=True, dtype=actions.dtype)
    if future_effort is None:
        raise ValueError("Future query probe requires future effort in observation.effort.")

    batch_size = processed.state.shape[0]
    zero_actions = jnp.zeros((batch_size, model.action_horizon, model.action_dim), dtype=actions.dtype)
    time = jnp.ones((batch_size,), dtype=actions.dtype)

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

    (outputs, selected_layers), _ = model.PaliGemma.llm(
        [prefix_tokens, student_tokens, None],
        mask=full_attn,
        positions=positions,
        adarms_cond=[None, student_adarms, None],
        return_layer_indices=(int(probe_layer),),
    )
    del outputs
    student_hidden = selected_layers[0][1]
    future_force_slice = slice(2, 3)
    flow_slice = slice(3, 3 + model.flow_token_count)
    force_hidden = jax.lax.stop_gradient(student_hidden[:, future_force_slice, :].astype(jnp.float32))
    flow_hidden = jax.lax.stop_gradient(student_hidden[:, flow_slice, :].astype(jnp.float32))
    return force_hidden, flow_hidden, future_effort.astype(jnp.float32), jnp.asarray(processed.flow_img, dtype=jnp.float32)


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
        force_hidden, flow_hidden, future_effort, flow_img = _extract_student_future_hiddens(
            model,
            step_rng,
            observation,
            actions,
            probe_layer=probe_layer,
        )
        pred_force, pred_flow = probe(force_hidden, flow_hidden)
        if pred_force.shape != future_effort.shape:
            raise ValueError(f"Force probe shape {pred_force.shape} does not match target {future_effort.shape}.")
        if pred_flow.shape != flow_img.shape:
            raise ValueError(f"Flow probe shape {pred_flow.shape} does not match target {flow_img.shape}.")
        loss_force = jnp.mean(jnp.square(pred_force - future_effort))
        loss_flow = jnp.mean(jnp.square(pred_flow - flow_img))
        loss = args.force_loss_weight * loss_force + args.flow_loss_weight * loss_flow
        return loss, {
            "loss": loss,
            "loss/force": loss_force,
            "loss/flow": loss_flow,
        }

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


def eval_predictions(
    probe_layer: int,
    model_def: nnx.GraphDef,
    model_params: nnx.State,
    probe_def: nnx.GraphDef,
    state: ProbeTrainState,
    batch,
    rng: at.KeyArrayLike,
) -> tuple[at.Array, at.Array]:
    model = nnx.merge(model_def, model_params)
    model.eval()
    probe = nnx.merge(probe_def, state.params)
    force_hidden, flow_hidden, _, flow_img = _extract_student_future_hiddens(
        model,
        rng,
        batch[0],
        batch[1],
        probe_layer=probe_layer,
    )
    _, pred_flow = probe(force_hidden, flow_hidden)
    return pred_flow, flow_img


def _flow_images_for_wandb(pred_flow: np.ndarray, target_flow: np.ndarray, max_items: int = 4) -> list[wandb.Image]:
    pred_flow = np.asarray(pred_flow[:max_items])
    target_flow = np.asarray(target_flow[:max_items])

    def to_uint8(x):
        x = np.clip((x + 1.0) * 127.5, 0, 255)
        return x.astype(np.uint8)

    images = []
    for pred, target in zip(pred_flow, target_flow, strict=False):
        images.append(wandb.Image(np.concatenate([to_uint8(target), to_uint8(pred)], axis=1)))
    return images


def main(args: Args) -> None:
    init_logging()
    logging.info("Running future query probe on: %s", platform.node())
    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by device count {jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))
    mesh = sharding.make_mesh(num_fsdp_devices=1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    logging.info("Using JAX mesh %s with devices: %s", mesh, jax.devices())

    base_config = _config.get_config(args.config_name)
    config = dataclasses.replace(
        base_config,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    if not isinstance(config.model, pi0_config.Pi0LatentFlowConfig):
        raise ValueError(f"Config {args.config_name!r} is not a Pi0LatentFlow config.")
    probe_layer = int(args.probe_layer if args.probe_layer is not None else config.model.distill_layer_indices[-1])

    ckpt_dir = _checkpoint_dir(args)
    _prepare_checkpoint_dir(ckpt_dir, overwrite=args.overwrite, resume=args.resume)

    wandb.init(
        mode="online" if args.wandb_enabled else "disabled",
        project=config.project_name,
        name=args.exp_name,
        config={
            **dataclasses.asdict(args),
            "base_config": args.config_name,
            "probe_layer": probe_layer,
            "checkpoint_dir": str(ckpt_dir),
        },
    )

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    logging.info("Initialized data loader:\n%s", training_utils.array_tree_to_info(first_batch))

    rng = jax.random.key(args.seed)
    rng, model_rng, probe_rng, train_rng = jax.random.split(rng, 4)
    model_def, model_params = _init_frozen_model(config, model_rng, pretrained_params=args.pretrained_params)
    model_params = jax.device_put(model_params, replicated_sharding)
    student_width = int(_gemma.get_config(config.model.action_expert_variant).width)
    probe = FutureQueryProbe(
        student_width=student_width,
        flow_token_count=int(config.model.flow_token_count),
        action_horizon=int(config.model.action_horizon),
        effort_dim=int(config.model.effort_dim if config.model.effort_dim is not None else config.model.effort_dim_in),
        decoder_dim=int(args.decoder_dim or student_width),
        decoder_depth=int(args.decoder_depth),
        decoder_heads=int(args.decoder_heads),
        image_size=224,
        patch_size=int(args.patch_size),
        rngs=nnx.Rngs(probe_rng),
    )
    probe_def, probe_params = nnx.split(probe)
    tx = optax.adamw(args.learning_rate)
    probe_state = ProbeTrainState(step=0, params=probe_params, opt_state=tx.init(probe_params), tx=tx)
    if args.resume:
        probe_state = _restore_probe_checkpoint(ckpt_dir, probe_state)
    probe_state = jax.device_put(probe_state, replicated_sharding)

    flat_probe = traverse_util.flatten_dict(probe_state.params.to_pure_dict())
    probe_param_count = sum(np.prod(v.shape) for v in flat_probe.values() if hasattr(v, "shape"))
    logging.info("Probe trainable parameters: %d", probe_param_count)
    logging.info("Frozen backbone params are not part of the optimizer state.")

    ptrain_step = jax.jit(
        lambda frozen_params, state, batch: train_step(
            args,
            probe_layer,
            model_def,
            frozen_params,
            probe_def,
            state,
            batch,
            train_rng,
        ),
        in_shardings=(replicated_sharding, replicated_sharding, data_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_predictions = jax.jit(
        lambda frozen_params, state, batch: eval_predictions(
            probe_layer,
            model_def,
            frozen_params,
            probe_def,
            state,
            batch,
            train_rng,
        ),
        in_shardings=(replicated_sharding, replicated_sharding, data_sharding),
        out_shardings=(data_sharding, data_sharding),
    )

    start_step = int(jax.device_get(probe_state.step))
    pbar = tqdm.tqdm(
        range(start_step, args.num_train_steps),
        initial=start_step,
        total=args.num_train_steps,
        dynamic_ncols=True,
    )
    infos = []
    batch = first_batch
    for step in pbar:
        with sharding.set_mesh(mesh):
            probe_state, info = ptrain_step(model_params, probe_state, batch)
        infos.append(info)

        if step % args.log_interval == 0:
            stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            info_str = ", ".join(f"{k}={float(v):.8f}" for k, v in reduced.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced, step=step)
            infos = []

        if args.image_log_interval > 0 and step % args.image_log_interval == 0:
            with sharding.set_mesh(mesh):
                pred_flow, target_flow = peval_predictions(model_params, probe_state, batch)
            wandb.log(
                {"flow_target_pred": _flow_images_for_wandb(jax.device_get(pred_flow), jax.device_get(target_flow))},
                step=step,
            )

        if (step % args.save_interval == 0 and step > start_step) or step == args.num_train_steps - 1:
            jax.block_until_ready(probe_state)
            _save_probe_checkpoint(ckpt_dir, probe_state, args)

        batch = next(data_iter)

    logging.info("Finished future query probe training. Checkpoints: %s", ckpt_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
