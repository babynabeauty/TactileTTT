import logging
import numpy as np # Import numpy for data loading

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from openpi.models import pi0_config
from openpi.models import model_tavla as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.shared.effort_type import EffortType
logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0TaVLA(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0TaVLAConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.effort_dim = config.effort_dim
        self.pi05 = config.pi05
        print("pi0.5 mode:", self.pi05)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.effort_type = config.effort_type
        if config.discrete_effort_input:
            self.effort_type = EffortType.LLM_HIS_Lang
        
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        if self.effort_type in (EffortType.LLM, EffortType.LLM_HIS_C, EffortType.LLM_HIS_T):
            self.effort_proj_in = nnx.Linear(config.effort_dim_in, 2 * paligemma_config.width, rngs=rngs)
            self.effort_proj_out = nnx.Linear(2 * paligemma_config.width, paligemma_config.width, rngs=rngs)
        elif self.effort_type in (EffortType.EXPERT, EffortType.EXPERT_HIS_C, EffortType.EXPERT_HIS_T, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            self.effort_proj_in = nnx.Linear(config.effort_dim_in, 2 * action_expert_config.width, rngs=rngs)
            self.effort_proj_out = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.effort_proj_in = None
            self.effort_proj_out = None
        
        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            self.action_in_proj = nnx.Linear(config.action_dim + config.effort_dim, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim + config.effort_dim, rngs=rngs)
        else:
            self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _project_effort(self, effort: at.Float[at.Array, "b *d"]) -> at.Float[at.Array, "b emb"]:
        effort_hidden = self.effort_proj_in(effort)
        effort_hidden = nnx.swish(effort_hidden)
        return self.effort_proj_out(effort_hidden)
        
    def _process_effort_tokens(self, obs: _model.Observation, mode: str) -> tuple[list, list, list]:
        tokens_list = []
        input_mask_list = []
        ar_mask_list = []
        
        # suffix token will not be attend by postfix
        ar_mask_value = mode == "suffix"
        
        if ((mode == "prefix" and self.effort_type in (EffortType.LLM, EffortType.LLM_HIS_C, EffortType.LLM_HIS_T, EffortType.LLM_HIS_FAST)) or
            (mode == "suffix" and self.effort_type in (EffortType.EXPERT, EffortType.EXPERT_HIS_C, EffortType.EXPERT_HIS_T,
                                                       EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT, EffortType.EXPERT_HIS_C_Conv))):
            
            if self.effort_type in (EffortType.LLM, EffortType.EXPERT):
                effort_token = self._project_effort(obs.effort[:, -1])[:, None, :] # assert last offset is 0(current)
                tokens_list.append(effort_token)
                input_mask_list.append(jnp.ones(effort_token.shape[:2], dtype=jnp.bool_))
                ar_mask_list.append(ar_mask_value)
            
            elif self.effort_type in (EffortType.LLM_HIS_C, EffortType.EXPERT_HIS_C,
                                      EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT, EffortType.EXPERT_HIS_C_Conv):
                if self.effort_type == EffortType.EXPERT_HIS_C_Conv:
                    effort_token = self._project_effort(obs.effort)
                    for i in range(effort_token.shape[1]):
                        ar_mask_list.append(ar_mask_value)
                else:
                    batch_size, _, _ = obs.effort.shape
                    effort_flat = obs.effort.reshape(batch_size, -1)
                    effort_token = self._project_effort(effort_flat)[:, None, :]
                    ar_mask_list.append(ar_mask_value)
                input_mask_list.append(jnp.ones(effort_token.shape[:2], dtype=jnp.bool_))
                tokens_list.append(effort_token)
            elif self.effort_type in (EffortType.LLM_HIS_T, EffortType.EXPERT_HIS_T):
                for i in range(obs.effort.shape[1]):
                    effort_token = self._project_effort(obs.effort[:, i])[:, None, :]
                    tokens_list.append(effort_token)
                    input_mask_list.append(jnp.ones(effort_token.shape[:2], dtype=jnp.bool_))
                    ar_mask_list.append(ar_mask_value)
                
        return tokens_list, input_mask_list, ar_mask_list

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        # add effort tokens
        effort_tokens, effort_input_mask, effort_ar_mask = self._process_effort_tokens(obs, mode="prefix")
        tokens.extend(effort_tokens)
        input_mask.extend(effort_input_mask)
        ar_mask.extend(effort_ar_mask)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        adarms_cond = None

        if self.effort_type != EffortType.EXPERT_HIS_C_L_FUT:
            # add effort tokens
            effort_tokens, effort_input_mask, effort_ar_mask = self._process_effort_tokens(obs, mode="suffix")
            tokens.extend(effort_tokens)
            input_mask.extend(effort_input_mask)
            ar_mask.extend(effort_ar_mask)

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        if self.effort_type == EffortType.EXPERT_HIS_C_L_FUT:
            # add effort tokens
            effort_tokens, effort_input_mask, effort_ar_mask = self._process_effort_tokens(obs, mode="suffix")
            tokens.extend(effort_tokens)
            input_mask.extend(effort_input_mask)
            ar_mask.extend(effort_ar_mask)
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, effort_type=self.effort_type)
        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            future_steps = actions.shape[1]
            future_effort = observation.effort[:, -future_steps:, :]
            assert actions.shape[-1] == self.action_dim
            observation = observation.replace(effort=observation.effort[:, :-future_steps, :])
            actions = jnp.concatenate([actions, future_effort], axis=-1)
            
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
            adarms_cond=[None, adarms_cond]
        )
        if self.effort_type != EffortType.EXPERT_HIS_C_L_FUT:
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        else:
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon-1:-1])

        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            action_loss = jnp.mean(jnp.square(v_t[..., :self.action_dim] - u_t[..., :self.action_dim]), axis=-1)
            effort_loss = jnp.mean(jnp.square(v_t[..., self.action_dim:] - u_t[..., self.action_dim:]), axis=-1)
            return action_loss + 0.1 * effort_loss
        else:
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim + self.effort_dim))
            else:
                noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            if self.effort_type != EffortType.EXPERT_HIS_C_L_FUT:
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            else:
                v_t = suffix_out[:, -self.action_horizon-1:-1]

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if self.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
            x_0 = x_0[..., :self.action_dim]
        return x_0
