import dataclasses

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.models.gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.pi0_tavla import Pi0TaVLA, make_attn_mask
from openpi.shared import array_typing as at
from typing_extensions import override


def _posemb_1d_from_grid(embed_dim: int, pos: at.Float[at.Array, "n"]) -> at.Float[at.Array, "n d"]:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, got {embed_dim}")
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2.0)))
    out = jnp.einsum("n,d->nd", pos.reshape(-1), omega)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


def _posemb_2d(embed_dim: int, grid_size: int) -> at.Float[at.Array, "n d"]:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be divisible by 2, got {embed_dim}")
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing="xy")
    emb_h = _posemb_1d_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = _posemb_1d_from_grid(embed_dim // 2, grid[1].reshape(-1))
    return jnp.concatenate([emb_h, emb_w], axis=1)


class FutureImageDecoderBlock(nnx.Module):
    def __init__(self, dim: int, num_heads: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
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
        attn = jnp.einsum("bhid,bhjd->bhij", q, k) * (head_dim ** -0.5)
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


class Pi0Seer(Pi0TaVLA):
    def __init__(self, config: pi0_config.Pi0SeerConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self.foreseen_token_count_per_view = int(config.foreseen_token_count_per_view)
        self.future_image_views = tuple(config.future_image_views)
        self.future_rgb_step = int(config.future_rgb_step)
        self.image_decoder_patch_size = int(config.image_decoder_patch_size)
        self.image_decoder_input_size = int(config.image_decoder_input_size)
        self.future_image_loss_weight = float(config.future_image_loss_weight)
        self.predict_future_image_during_inference = bool(config.predict_future_image_during_inference)
        self.use_future_rgb_instead_of_flow = bool(config.use_future_rgb_instead_of_flow)
        self.normalize_future_patch_targets = bool(config.normalize_future_patch_targets)

        self.foreseen_view_to_field = {
            "base_0_rgb": "future_rgb_img",
            "left_wrist_0_rgb": "future_wrist_rgb_img",
        }
        unsupported_views = [view for view in self.future_image_views if view not in self.foreseen_view_to_field]
        if unsupported_views:
            raise ValueError(
                "Pi0Seer only supports future image views "
                f"{tuple(self.foreseen_view_to_field)}, got unsupported views {unsupported_views}."
            )

        self.num_future_views = len(self.future_image_views)
        self.total_foreseen_tokens = self.foreseen_token_count_per_view * self.num_future_views
        self.prefix_token_width = int(paligemma_config.width)

        self.foreseen_tokens = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.total_foreseen_tokens, self.prefix_token_width),
                dtype=jnp.float32,
            )
        )
        self.image_decoder_hidden_dim = self.prefix_token_width
        self.image_decoder_projector = nnx.Linear(
            self.prefix_token_width, self.image_decoder_hidden_dim, rngs=rngs
        )
        self.num_mask_tokens = (self.image_decoder_input_size // self.image_decoder_patch_size) ** 2
        self.mask_token = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.image_decoder_hidden_dim,), dtype=jnp.float32)
        )
        self.image_decoder_block_1 = FutureImageDecoderBlock(self.image_decoder_hidden_dim, num_heads=8, rngs=rngs)
        self.image_decoder_block_2 = FutureImageDecoderBlock(self.image_decoder_hidden_dim, num_heads=8, rngs=rngs)
        self.image_decoder_norm = nnx.LayerNorm(num_features=self.image_decoder_hidden_dim, rngs=rngs)
        self.image_decoder_pred = nnx.Linear(
            self.image_decoder_hidden_dim,
            self.image_decoder_patch_size * self.image_decoder_patch_size * 3,
            rngs=rngs,
        )
    def _image_decoder_position_embedding(self) -> at.Float[at.Array, "1 n d"]:
        patch_grid_size = int(round(self.num_mask_tokens ** 0.5))
        return jnp.concatenate(
            [
                _posemb_1d_from_grid(
                    self.image_decoder_hidden_dim,
                    jnp.arange(self.foreseen_token_count_per_view, dtype=jnp.float32),
                ),
                _posemb_2d(self.image_decoder_hidden_dim, patch_grid_size),
            ],
            axis=0,
        )[None, :, :]

    def _broadcast_foreseen_tokens(self, batch_size: int, dtype: jnp.dtype) -> at.Float[at.Array, "b n d"]:
        tokens = jnp.asarray(self.foreseen_tokens.value, dtype=dtype)
        return jnp.broadcast_to(tokens[None, :, :], (batch_size, tokens.shape[0], tokens.shape[1]))

    @override
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens, input_mask, ar_mask = super().embed_prefix(obs)
        foreseen_tokens = self._broadcast_foreseen_tokens(tokens.shape[0], tokens.dtype)
        foreseen_mask = jnp.ones(foreseen_tokens.shape[:2], dtype=jnp.bool_)
        foreseen_ar_mask = jnp.zeros((self.total_foreseen_tokens,), dtype=jnp.bool_)
        tokens = jnp.concatenate([tokens, foreseen_tokens], axis=1)
        input_mask = jnp.concatenate([input_mask, foreseen_mask], axis=1)
        ar_mask = jnp.concatenate([ar_mask, foreseen_ar_mask], axis=0)
        return tokens, input_mask, ar_mask

    def _get_future_image_for_view(
        self, observation: _model.Observation, view_name: str
    ) -> at.Float[at.Array, "b h w c"]:
        field_name = self.foreseen_view_to_field[view_name]
        image = getattr(observation, field_name)
        if image is None:
            raise ValueError(
                f"Pi0Seer requires `{field_name}` for future image supervision with view `{view_name}`."
            )
        return jnp.asarray(image, dtype=jnp.float32)

    def _patchify_image(self, image: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b n p"]:
        patch = self.image_decoder_patch_size
        if image.shape[1] != self.image_decoder_input_size or image.shape[2] != self.image_decoder_input_size:
            raise ValueError(
                "Future image has unexpected resolution, expected "
                f"{self.image_decoder_input_size}x{self.image_decoder_input_size}, got {image.shape[1:3]}."
            )
        return einops.rearrange(
            image,
            "b (h ph) (w pw) c -> b (h w) (ph pw c)",
            ph=patch,
            pw=patch,
        )

    def _normalize_patchified_image(self, patches: at.Float[at.Array, "b n p"]) -> at.Float[at.Array, "b n p"]:
        mean = jnp.mean(patches, axis=-1, keepdims=True)
        var = jnp.var(patches, axis=-1, keepdims=True)
        return (patches - mean) / jnp.sqrt(var + 1e-6)

    def _decode_future_images(
        self, foreseen_hidden: at.Float[at.Array, "b n d"]
    ) -> at.Float[at.Array, "b v m p"]:
        bsz = foreseen_hidden.shape[0]
        decoded = self.image_decoder_projector(foreseen_hidden)
        decoded = decoded.reshape(
            bsz * self.num_future_views,
            self.foreseen_token_count_per_view,
            self.image_decoder_hidden_dim,
        )
        mask_tokens = jnp.asarray(self.mask_token.value, dtype=decoded.dtype)
        mask_tokens = jnp.broadcast_to(
            mask_tokens[None, None, :],
            (decoded.shape[0], self.num_mask_tokens, self.image_decoder_hidden_dim),
        )
        decoder_input = jnp.concatenate([decoded, mask_tokens], axis=1)
        decoder_input = decoder_input + jnp.asarray(self._image_decoder_position_embedding(), dtype=decoder_input.dtype)
        decoder_input = self.image_decoder_block_1(decoder_input)
        decoder_input = self.image_decoder_block_2(decoder_input)
        patch_hidden = self.image_decoder_norm(decoder_input[:, -self.num_mask_tokens :, :])
        patch_pred = self.image_decoder_pred(patch_hidden)
        return patch_pred.reshape(bsz, self.num_future_views, self.num_mask_tokens, -1)

    def _future_image_labels(self, observation: _model.Observation) -> at.Float[at.Array, "b v m p"]:
        labels = [self._patchify_image(self._get_future_image_for_view(observation, view)) for view in self.future_image_views]
        patches = jnp.stack(labels, axis=1)
        if self.normalize_future_patch_targets:
            patches = self._normalize_patchified_image(
                patches.reshape(-1, self.num_mask_tokens, patches.shape[-1])
            ).reshape(patches.shape)
        return patches

    @override
    def compute_loss_with_stats(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, effort_type=self.effort_type)
        if self.effort_type in (
            _model.EffortType.EXPERT_FUT,
            _model.EffortType.EXPERT_HIS_C_FUT,
            _model.EffortType.EXPERT_HIS_C_L_FUT,
        ):
            future_steps = actions.shape[1]
            if observation.effort is None:
                raise ValueError("Pi0Seer requires observation.effort for future-effort training modes.")
            future_effort = observation.effort[:, -future_steps:, :]
            observation = observation.replace(effort=observation.effort[:, :-future_steps, :])
            actions = jnp.concatenate([actions, future_effort], axis=-1)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        if self.effort_type != _model.EffortType.EXPERT_HIS_C_L_FUT:
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        else:
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon - 1 : -1])

        if self.effort_type in (
            _model.EffortType.EXPERT_FUT,
            _model.EffortType.EXPERT_HIS_C_FUT,
            _model.EffortType.EXPERT_HIS_C_L_FUT,
        ):
            action_loss = jnp.mean(jnp.square(v_t[..., : self.action_dim] - u_t[..., : self.action_dim]), axis=-1)
            effort_loss = jnp.mean(jnp.square(v_t[..., self.action_dim :] - u_t[..., self.action_dim :]), axis=-1)
            base_loss = action_loss + 0.1 * effort_loss
            effort_loss_stat = jnp.mean(effort_loss, axis=-1)
        else:
            action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            base_loss = action_loss
            effort_loss_stat = jnp.zeros((base_loss.shape[0],), dtype=base_loss.dtype)

        foreseen_hidden = prefix_out[:, -self.total_foreseen_tokens :, :]
        future_image_pred = self._decode_future_images(foreseen_hidden)
        future_image_target = self._future_image_labels(observation)
        future_image_loss = jnp.mean(jnp.square(future_image_pred - future_image_target), axis=(1, 2, 3))
        action_loss_stat = jnp.mean(action_loss, axis=-1)
        while future_image_loss.ndim < base_loss.ndim:
            future_image_loss = future_image_loss[:, None]
        total_loss = base_loss + self.future_image_loss_weight * future_image_loss
        stats = {
            "loss/action": action_loss_stat,
            "loss/effort": effort_loss_stat,
            "loss/future_image": future_image_loss[:, 0] if future_image_loss.ndim > 1 else future_image_loss,
            "loss/total": jnp.mean(total_loss, axis=-1),
        }
        return total_loss, stats

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train)
        return loss
