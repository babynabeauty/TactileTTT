import math

import einops
from diffusers import FlaxAutoencoderKL
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models import siglip as _siglip
from openpi.models.pi0_tavla import make_attn_mask, posemb_sincos
from openpi.models.tactile_tokenizer import AdaptiveFingertipPatchTokenizer
from openpi.models.tactile_tokenizer import DexterousForceTokenizer
from openpi.models.tactile_tokenizer import PatchInformedFingerTokenizer
from openpi.models.tactile_tokenizer import PlainRawTactileMLPTokenizer
from openpi.models.tactile_tokenizer import RawTactileSpatialTokenizer
from openpi.models.tactile_ttt import TactileTTTMemory
from openpi.shared import array_typing as at


_FLOW_VAE_REGISTRY: dict[str, tuple[FlaxAutoencoderKL, at.Params]] = {}


def register_flow_vae(model_name: str, flow_vae: FlaxAutoencoderKL, flow_vae_params: at.Params) -> None:
    _FLOW_VAE_REGISTRY[model_name] = (flow_vae, flow_vae_params)


def preload_flow_vae(model_name: str) -> None:
    if model_name in _FLOW_VAE_REGISTRY:
        return
    flow_vae, flow_vae_params = FlaxAutoencoderKL.from_pretrained(
        model_name,
        from_pt=True,
        dtype=jnp.float32,
    )
    register_flow_vae(model_name, flow_vae, flow_vae_params)


def _get_flow_vae(model_name: str) -> tuple[FlaxAutoencoderKL, at.Params]:
    if model_name not in _FLOW_VAE_REGISTRY:
        raise ValueError(
            f"Flow VAE '{model_name}' has not been preloaded. "
            "Call preload_flow_vae(...) in Python before initializing training."
        )
    return _FLOW_VAE_REGISTRY[model_name]


class _TactileRefinerBlock(nnx.Module):
    """Small pre-norm Transformer block for tactile residual refinement."""

    def __init__(self, *, width: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.head_dim = self.width // self.num_heads

        self.q = nnx.Linear(width, width, rngs=rngs)
        self.k = nnx.Linear(width, width, rngs=rngs)
        self.v = nnx.Linear(width, width, rngs=rngs)
        self.attn_out = nnx.Linear(width, width, rngs=rngs)
        self.ffn_in = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.ffn_out = nnx.Linear(mlp_dim, width, rngs=rngs)
        self.attn_norm = nnx.LayerNorm(num_features=width, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(num_features=width, rngs=rngs)

    def _attention(self, x: jax.Array) -> jax.Array:
        query = einops.rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)
        key = einops.rearrange(self.k(x), "b n (h d) -> b h n d", h=self.num_heads)
        value = einops.rearrange(self.v(x), "b n (h d) -> b h n d", h=self.num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(float(self.head_dim))
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
        return einops.rearrange(attended, "b h n d -> b n (h d)")

    def __call__(self, tokens: jax.Array) -> jax.Array:
        tokens = tokens + self.attn_out(self._attention(self.attn_norm(tokens)))
        tokens = tokens + self.ffn_out(nnx.swish(self.ffn_in(self.ffn_norm(tokens))))
        return tokens


class _AsyncTactileRefinerBlock(nnx.Module):
    """Decoder-style block: hand queries self-attend, then read tactile/contact context."""

    def __init__(self, *, width: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.head_dim = self.width // self.num_heads

        self.self_q = nnx.Linear(width, width, rngs=rngs)
        self.self_k = nnx.Linear(width, width, rngs=rngs)
        self.self_v = nnx.Linear(width, width, rngs=rngs)
        self.self_out = nnx.Linear(width, width, rngs=rngs)
        self.self_norm = nnx.LayerNorm(num_features=width, rngs=rngs)

        self.cross_q = nnx.Linear(width, width, rngs=rngs)
        self.cross_k = nnx.Linear(width, width, rngs=rngs)
        self.cross_v = nnx.Linear(width, width, rngs=rngs)
        self.cross_out = nnx.Linear(width, width, rngs=rngs)
        self.cross_norm = nnx.LayerNorm(num_features=width, rngs=rngs)

        self.ffn_in = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.ffn_out = nnx.Linear(mlp_dim, width, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(num_features=width, rngs=rngs)

    def _attention(
        self,
        query_tokens: jax.Array,
        context_tokens: jax.Array,
        q_proj: nnx.Linear,
        k_proj: nnx.Linear,
        v_proj: nnx.Linear,
    ) -> jax.Array:
        query = einops.rearrange(q_proj(query_tokens), "b n (h d) -> b h n d", h=self.num_heads)
        key = einops.rearrange(k_proj(context_tokens), "b n (h d) -> b h n d", h=self.num_heads)
        value = einops.rearrange(v_proj(context_tokens), "b n (h d) -> b h n d", h=self.num_heads)
        logits = jnp.einsum("bhqd,bhkd->bhqk", query, key) / math.sqrt(float(self.head_dim))
        weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, value)
        return einops.rearrange(attended, "b h n d -> b n (h d)")

    def __call__(self, hand_query: jax.Array, context: jax.Array) -> jax.Array:
        normed = self.self_norm(hand_query)
        hand_query = hand_query + self.self_out(
            self._attention(normed, normed, self.self_q, self.self_k, self.self_v)
        )
        hand_query = hand_query + self.cross_out(
            self._attention(self.cross_norm(hand_query), context, self.cross_q, self.cross_k, self.cross_v)
        )
        hand_query = hand_query + self.ffn_out(nnx.swish(self.ffn_in(self.ffn_norm(hand_query))))
        return hand_query


class Pi0LatentFlow(_model.BaseModel):
    """Standalone dual-expert model with future-force and future-flow alignment."""

    def __init__(self, config: pi0_config.Pi0LatentFlowConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        effort_dim_in = getattr(config, "effort_dim_in", None)
        if effort_dim_in is None:
            effort_dim_in = config.effort_dim
        if effort_dim_in is None or int(effort_dim_in) <= 0:
            raise ValueError("Pi0MORDualAlignForceFlow requires a positive `effort_dim_in`.")

        self.effort_dim_in = int(effort_dim_in)
        self.effort_dim = int(config.effort_dim if config.effort_dim is not None else self.effort_dim_in)
        self.pi05 = config.pi05
        # Pi0.5 serializes proprioception into the language prompt.  Keep the
        # continuous state token only for Pi0, matching the base Pi0/Pi0.5 model.
        self.state_token_count = 0 if self.pi05 else 1
        self.effort_type = config.effort_type

        self.force_input_frames = int(config.force_input_frames)
        self.structured_tactile = bool(config.structured_tactile)
        self.tactile_num_fingers = int(config.tactile_num_fingers)
        self.tactile_dim_per_finger = int(config.tactile_dim_per_finger)
        self.tactile_points_per_finger = int(config.tactile_points_per_finger)
        self.future_tactile_segments = int(config.future_tactile_segments)
        self.future_steps_per_segment = int(config.future_steps_per_segment)
        self.pool_tactile_history = bool(getattr(config, "pool_tactile_history", False))
        self.tactile_history_segments = int(getattr(config, "tactile_history_segments", 0))
        self.cached_vlm_async_ae_enabled = bool(getattr(config, "cached_vlm_async_ae_enabled", False))
        self.cached_vlm_async_offsets = tuple(
            int(offset) for offset in getattr(config, "cached_vlm_async_offsets", (0, 4, 8, 12))
        )
        self.cached_vlm_async_loss_weight = float(getattr(config, "cached_vlm_async_loss_weight", 1.0))
        self.cached_vlm_async_future_align_loss_weight = float(
            getattr(config, "cached_vlm_async_future_align_loss_weight", 0.1)
        )
        self.cached_vlm_async_prefix_consistency_weight = float(
            getattr(config, "cached_vlm_async_prefix_consistency_weight", 0.0)
        )
        self.cached_vlm_async_use_predicted_prefix_queries = bool(
            getattr(config, "cached_vlm_async_use_predicted_prefix_queries", True)
        )
        self.cached_vlm_async_loss_mask = getattr(config, "cached_vlm_async_loss_mask", "full")
        self.cached_vlm_async_history_mode = getattr(config, "cached_vlm_async_history_mode", "pooled")
        self.tactile_patch_tokenizer = bool(getattr(config, "tactile_patch_tokenizer", False))
        self.tactile_patch_informed_tokenizer = bool(getattr(config, "tactile_patch_informed_tokenizer", False))
        self.tactile_raw_mlp_tokenizer = bool(getattr(config, "tactile_raw_mlp_tokenizer", False))
        self.disable_future_tactile = bool(getattr(config, "disable_future_tactile", False))
        self.direct_future_tactile_align = bool(getattr(config, "direct_future_tactile_align", False))
        self.tactile_ttt_enabled = bool(getattr(config, "tactile_ttt_enabled", False))
        self.tactile_ttt_write_segments = int(getattr(config, "tactile_ttt_write_segments", 4))
        self.sequence_training = self.tactile_ttt_enabled
        self.use_teacher_ae = not (self.disable_future_tactile or self.direct_future_tactile_align)
        self.tactile_patch_fingers = tuple(int(finger) for finger in getattr(config, "tactile_patch_fingers", (0, 1, 2)))
        self.tactile_num_patches = int(getattr(config, "tactile_num_patches", 5))
        self.tactile_patch_aux_loss_weight = float(getattr(config, "tactile_patch_aux_loss_weight", 0.0))
        self.tactile_tokens_per_step = self.tactile_num_fingers
        if self.structured_tactile and self.tactile_patch_tokenizer:
            self.tactile_tokens_per_step = self.tactile_num_fingers + len(self.tactile_patch_fingers) * self.tactile_num_patches
        self.future_force_token_count = 0 if self.disable_future_tactile else (
            self.future_tactile_segments * self.tactile_tokens_per_step if self.structured_tactile else 1
        )
        if self.tactile_ttt_enabled:
            self.history_force_token_count = self.tactile_num_fingers
        elif self.structured_tactile:
            if self.tactile_history_segments:
                self.history_force_token_count = self.tactile_history_segments * self.tactile_tokens_per_step
            elif self.pool_tactile_history and self.cached_vlm_async_history_mode == "pooled_current":
                self.history_force_token_count = 2 * self.tactile_tokens_per_step
            elif self.pool_tactile_history:
                self.history_force_token_count = self.tactile_tokens_per_step
            else:
                self.history_force_token_count = self.force_input_frames * self.tactile_tokens_per_step
        else:
            self.history_force_token_count = 1
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )
        self.tactile_sample_hz = float(config.tactile_sample_hz)
        self.distill_layer_indices = tuple(int(i) for i in config.distill_layer_indices)
        self.student_action_loss_weight = float(config.student_action_loss_weight)
        self.teacher_action_loss_weight = float(config.teacher_action_loss_weight)
        self.future_force_align_loss_weight = float(config.future_force_align_loss_weight)
        self.future_flow_align_loss_weight = float(config.future_flow_align_loss_weight)
        self.use_future_flow = bool(getattr(config, "use_future_flow", True))
        self.flow_token_count = int(config.flow_token_count)
        self.future_flow_source = getattr(config, "future_flow_source", "image")
        self.scene_flow_input_dim = int(getattr(config, "scene_flow_input_dim", 10))
        self.future_flow_channels = 3
        self.flow_vae_name = getattr(config, "flow_vae_name", "stabilityai/sdxl-vae")
        self.use_future_rgb_instead_of_flow = bool(config.use_future_rgb_instead_of_flow)
        self.future_rgb_step = int(config.future_rgb_step)
        self.student_future_query_noise_scale_max = float(config.student_future_query_noise_scale_max)
        self.student_future_query_noise_start_ratio = float(config.student_future_query_noise_start_ratio)
        self.student_future_query_noise_end_ratio = float(config.student_future_query_noise_end_ratio)
        self.arm_hand_mask_attention = bool(config.arm_hand_mask_attention)
        self.arm_action_dim = int(config.arm_action_dim)
        self.hand_action_dim = int(config.hand_action_dim)
        self.hand_action_start = self.arm_action_dim
        self.tactile_refiner_enabled = bool(getattr(config, "tactile_refiner_enabled", False))
        self.tactile_refiner_layers = int(getattr(config, "tactile_refiner_layers", 2))
        self.tactile_refiner_width = int(getattr(config, "tactile_refiner_width", 256))
        self.tactile_refiner_heads = int(getattr(config, "tactile_refiner_heads", 4))
        self.tactile_refiner_mlp_dim = int(getattr(config, "tactile_refiner_mlp_dim", 1024))
        self.hand_synergy_dim = int(getattr(config, "hand_synergy_dim", 4))
        self.hand_synergy_loss_weight = float(getattr(config, "hand_synergy_loss_weight", 0.03))
        self.tactile_refiner_delta_loss_weight = float(getattr(config, "tactile_refiner_delta_loss_weight", 0.0001))
        self.tactile_refiner_gate_bias = float(getattr(config, "tactile_refiner_gate_bias", -2.0))
        self.tactile_refiner_delta_scale = float(getattr(config, "tactile_refiner_delta_scale", 0.1))
        self.async_tactile_refiner_enabled = bool(getattr(config, "async_tactile_refiner_enabled", False))
        self.async_refiner_offsets = tuple(int(offset) for offset in getattr(config, "async_refiner_offsets", (4, 8, 12)))
        self.async_refiner_layers = int(getattr(config, "async_refiner_layers", 2))
        self.async_refiner_width = int(getattr(config, "async_refiner_width", 256))
        self.async_refiner_heads = int(getattr(config, "async_refiner_heads", 4))
        self.async_refiner_mlp_dim = int(getattr(config, "async_refiner_mlp_dim", 1024))
        self.async_refiner_loss_weight = float(getattr(config, "async_refiner_loss_weight", 0.2))
        self.async_refiner_delta_loss_weight = float(getattr(config, "async_refiner_delta_loss_weight", 0.0001))
        self.async_refiner_gate_loss_weight = float(getattr(config, "async_refiner_gate_loss_weight", 0.0))
        self.async_refiner_gate_bias = float(getattr(config, "async_refiner_gate_bias", -2.0))
        self.async_refiner_delta_scale = float(getattr(config, "async_refiner_delta_scale", 0.1))
        self.async_tactile_flow_refiner_enabled = bool(
            getattr(config, "async_tactile_flow_refiner_enabled", False)
        )
        self.async_flow_refiner_offsets = tuple(
            int(offset) for offset in getattr(config, "async_flow_refiner_offsets", (4, 8, 12))
        )
        self.async_flow_refiner_layers = int(getattr(config, "async_flow_refiner_layers", 2))
        self.async_flow_refiner_width = int(getattr(config, "async_flow_refiner_width", 256))
        self.async_flow_refiner_heads = int(getattr(config, "async_flow_refiner_heads", 4))
        self.async_flow_refiner_mlp_dim = int(getattr(config, "async_flow_refiner_mlp_dim", 1024))
        self.async_flow_refiner_loss_weight = float(getattr(config, "async_flow_refiner_loss_weight", 1.0))
        self.async_flow_refiner_tau_split = float(getattr(config, "async_flow_refiner_tau_split", 0.4))
        self.uses_train_progress = True
        self._debug_lengths_logged = False


        paligemma_config = _gemma.get_config(config.paligemma_variant)
        student_config = _gemma.get_config(config.action_expert_variant)
        teacher_variant = getattr(config, "force_expert_variant", config.action_expert_variant)
        teacher_config = _gemma.get_config(teacher_variant)
        self.student_width = int(student_config.width)
        self.teacher_width = int(teacher_config.width)
        self.distill_projector_hidden_dim = int(
            config.distill_projector_hidden_dim
            if config.distill_projector_hidden_dim is not None
            else teacher_config.width
        )
        llm_configs = (
            [paligemma_config, student_config]
            if not self.use_teacher_ae
            else [paligemma_config, student_config, teacher_config]
        )

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=llm_configs,
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=(
                [False] + [True] * (len(llm_configs) - 1)
                if config.pi05
                else [False] * len(llm_configs)
            ),
        )

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

        if config.pi05:
            self.student_time_mlp_in = nnx.Linear(student_config.width, student_config.width, rngs=rngs)
            self.student_time_mlp_out = nnx.Linear(student_config.width, student_config.width, rngs=rngs)
            self.teacher_time_mlp_in = nnx.Linear(teacher_config.width, teacher_config.width, rngs=rngs)
            self.teacher_time_mlp_out = nnx.Linear(teacher_config.width, teacher_config.width, rngs=rngs)
        else:
            self.student_time_mlp_in = nnx.Linear(2 * student_config.width, student_config.width, rngs=rngs)
            self.student_time_mlp_out = nnx.Linear(student_config.width, student_config.width, rngs=rngs)
            self.teacher_time_mlp_in = nnx.Linear(2 * teacher_config.width, teacher_config.width, rngs=rngs)
            self.teacher_time_mlp_out = nnx.Linear(teacher_config.width, teacher_config.width, rngs=rngs)

        self.state_proj_student = nnx.Linear(config.action_dim, student_config.width, rngs=rngs)
        self.state_proj_teacher = nnx.Linear(config.action_dim, teacher_config.width, rngs=rngs)
        self.action_in_proj_student = nnx.Linear(config.action_dim, student_config.width, rngs=rngs)
        self.action_in_proj_teacher = nnx.Linear(config.action_dim, teacher_config.width, rngs=rngs)
        self.action_out_proj_student = nnx.Linear(student_config.width, config.action_dim, rngs=rngs)
        self.action_out_proj_teacher = nnx.Linear(teacher_config.width, config.action_dim, rngs=rngs)

        if self.structured_tactile:
            if self.tactile_points_per_finger > 1:
                tokenizer_kwargs = dict(
                    hidden_dim=config.tactile_tokenizer_dim,
                    num_fingers=self.tactile_num_fingers,
                    num_points=self.tactile_points_per_finger,
                    dim_per_point=self.tactile_dim_per_finger,
                    future_segments=self.future_tactile_segments,
                    future_steps_per_segment=self.future_steps_per_segment,
                    contact_top_k=config.tactile_raw_contact_top_k,
                    contact_threshold=config.tactile_raw_contact_threshold,
                    contact_temperature=config.tactile_raw_contact_temperature,
                )
                if self.tactile_patch_tokenizer:
                    tokenizer_cls = AdaptiveFingertipPatchTokenizer
                    tokenizer_kwargs.update(
                        patch_fingers=self.tactile_patch_fingers,
                        num_patches=self.tactile_num_patches,
                    )
                elif self.tactile_patch_informed_tokenizer:
                    tokenizer_cls = PatchInformedFingerTokenizer
                    tokenizer_kwargs.update(num_patches=self.tactile_num_patches)
                elif self.tactile_raw_mlp_tokenizer:
                    tokenizer_cls = PlainRawTactileMLPTokenizer
                else:
                    tokenizer_cls = RawTactileSpatialTokenizer
                self.student_force_tokenizer = tokenizer_cls(output_dim=student_config.width, rngs=rngs, **tokenizer_kwargs)
                self.teacher_force_tokenizer = tokenizer_cls(output_dim=teacher_config.width, rngs=rngs, **tokenizer_kwargs)
            else:
                tokenizer_kwargs = dict(
                    hidden_dim=config.tactile_tokenizer_dim,
                    num_fingers=self.tactile_num_fingers,
                    dim_per_finger=self.tactile_dim_per_finger,
                    future_segments=self.future_tactile_segments,
                    future_steps_per_segment=self.future_steps_per_segment,
                )
                self.student_force_tokenizer = DexterousForceTokenizer(
                    output_dim=student_config.width, rngs=rngs, **tokenizer_kwargs
                )
                self.teacher_force_tokenizer = DexterousForceTokenizer(
                    output_dim=teacher_config.width, rngs=rngs, **tokenizer_kwargs
                )
            self.student_query_base = nnx.Param(
                0.02 * jax.random.normal(rngs.params(), (student_config.width,), dtype=jnp.float32)
            )
            self.student_query_segment_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs.params(), (self.future_tactile_segments, student_config.width), dtype=jnp.float32
                )
            )
            self.student_query_finger_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs.params(), (self.tactile_tokens_per_step, student_config.width), dtype=jnp.float32
                )
            )
            if self.pool_tactile_history:
                self.student_history_pool_logits = nnx.Param(
                    jnp.zeros((self.force_input_frames,), dtype=jnp.float32)
                )
                self.teacher_history_pool_logits = nnx.Param(
                    jnp.zeros((self.force_input_frames,), dtype=jnp.float32)
                )
                if self.cached_vlm_async_history_mode == "pooled_current":
                    self.student_history_type_embedding = nnx.Param(
                        0.02
                        * jax.random.normal(
                            rngs.params(), (2, student_config.width), dtype=jnp.float32
                        )
                    )
                    self.teacher_history_type_embedding = nnx.Param(
                        0.02
                        * jax.random.normal(
                            rngs.params(), (2, teacher_config.width), dtype=jnp.float32
                        )
                    )
            if self.tactile_ttt_enabled:
                self.tactile_ttt = TactileTTTMemory(
                    token_dim=student_config.width,
                    memory_dim=config.tactile_ttt_memory_dim,
                    inner_lr=config.tactile_ttt_inner_lr,
                    contact_top_k=config.tactile_raw_contact_top_k,
                    contact_threshold=config.tactile_raw_contact_threshold,
                    contact_temperature=config.tactile_raw_contact_temperature,
                    rngs=rngs,
                )
        else:
            history_dim = self.force_input_frames * self.effort_dim_in
            future_dim = config.action_horizon * self.effort_dim_in
            self.history_force_proj_student = nnx.Linear(history_dim, student_config.width, rngs=rngs)
            self.history_force_proj_teacher = nnx.Linear(history_dim, teacher_config.width, rngs=rngs)
            self.future_force_proj_teacher = nnx.Linear(future_dim, teacher_config.width, rngs=rngs)
            self.student_query = nnx.Param(
                0.02 * jax.random.normal(rngs.params(), (student_config.width,), dtype=jnp.float32)
            )
        self.student_future_mask_token = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (student_config.width,), dtype=jnp.float32)
        )
        self.prompt_distill_proj_in = nnx.Linear(
            student_config.width, self.distill_projector_hidden_dim, rngs=rngs
        )
        self.prompt_distill_proj_out = nnx.Linear(
            self.distill_projector_hidden_dim, teacher_config.width, rngs=rngs
        )

        self.student_future_flow_query = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.flow_token_count, self.student_width),
                dtype=jnp.float32,
            )
        )
        self.teacher_future_flow_query = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.flow_token_count, self.teacher_width),
                dtype=jnp.float32,
            )
        )
        self.flow_distill_proj_in = nnx.Linear(
            self.student_width, self.distill_projector_hidden_dim, rngs=rngs
        )
        self.flow_distill_proj_out = nnx.Linear(
            self.distill_projector_hidden_dim, self.teacher_width, rngs=rngs
        )

        self.flow_vae_latent_channels = int(config.flow_vae_latent_channels)
        self.flow_vae_patch_merge_factor = 2
        self.flow_vae_proj_in = nnx.Linear(
            self.flow_vae_latent_channels * (self.flow_vae_patch_merge_factor ** 2),
            self.teacher_width,
            rngs=rngs,
        )
        self.flow_vae_proj_out = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_vae_norm = nnx.LayerNorm(num_features=self.teacher_width, rngs=rngs)
        self.scene_flow_proj_in = nnx.Linear(self.scene_flow_input_dim, self.teacher_width, rngs=rngs)
        self.scene_flow_proj_out = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.scene_flow_norm = nnx.LayerNorm(num_features=self.teacher_width, rngs=rngs)
        self.flow_vae_query_proj = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_vae_key_proj = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_vae_value_proj = nnx.Linear(self.teacher_width, self.teacher_width, rngs=rngs)
        self.flow_token_embedding = nnx.Param(
            0.02
            * jax.random.normal(
                rngs.params(),
                (self.flow_token_count, self.teacher_width),
                dtype=jnp.float32,
            )
        )

        if self.tactile_patch_aux_loss_weight > 0:
            self.teacher_patch_dist_head = nnx.Linear(self.teacher_width, self.tactile_num_patches, rngs=rngs)

        if self.tactile_refiner_enabled:
            synergy_input_dim = self.student_width * 4
            synergy_hidden_dim = self.student_width
            self.hand_synergy_in = nnx.Linear(synergy_input_dim, synergy_hidden_dim, rngs=rngs)
            self.hand_synergy_out = nnx.Linear(synergy_hidden_dim, self.hand_synergy_dim, rngs=rngs)
            self.hand_synergy_decoder_in = nnx.Linear(self.hand_synergy_dim, 128, rngs=rngs)
            self.hand_synergy_decoder_out = nnx.Linear(128, self.hand_action_dim, rngs=rngs)

            self.refiner_history_proj = nnx.Linear(self.student_width, self.tactile_refiner_width, rngs=rngs)
            self.refiner_future_proj = nnx.Linear(self.student_width, self.tactile_refiner_width, rngs=rngs)
            self.refiner_arm_proj = nnx.Linear(self.student_width, self.tactile_refiner_width, rngs=rngs)
            self.refiner_hand_proj = nnx.Linear(self.student_width, self.tactile_refiner_width, rngs=rngs)
            self.refiner_synergy_proj = nnx.Linear(self.hand_synergy_dim, self.tactile_refiner_width, rngs=rngs)
            self.refiner_coarse_proj = nnx.Linear(self.hand_action_dim, self.tactile_refiner_width, rngs=rngs)
            self.tactile_refiner_blocks = [
                _TactileRefinerBlock(
                    width=self.tactile_refiner_width,
                    num_heads=self.tactile_refiner_heads,
                    mlp_dim=self.tactile_refiner_mlp_dim,
                    rngs=rngs,
                )
                for _ in range(self.tactile_refiner_layers)
            ]
            self.refiner_delta_out = nnx.Linear(self.tactile_refiner_width, self.hand_action_dim, rngs=rngs)
            self.refiner_gate_out = nnx.Linear(self.tactile_refiner_width, 1, rngs=rngs)

        if self.async_tactile_refiner_enabled:
            self.async_hand_query_proj = nnx.Linear(self.hand_action_dim, self.async_refiner_width, rngs=rngs)
            self.async_arm_context_proj = nnx.Linear(self.arm_action_dim, self.async_refiner_width, rngs=rngs)
            self.async_state_context_proj = nnx.Linear(config.action_dim, self.async_refiner_width, rngs=rngs)
            self.async_fresh_tactile_proj = nnx.Linear(self.student_width, self.async_refiner_width, rngs=rngs)
            self.async_future_contact_proj = nnx.Linear(self.student_width, self.async_refiner_width, rngs=rngs)
            self.async_time_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs.params(), (self.action_horizon, self.async_refiner_width), dtype=jnp.float32
                )
            )
            self.async_offset_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs.params(), (self.action_horizon + 1, self.async_refiner_width), dtype=jnp.float32
                )
            )
            self.async_refiner_blocks = [
                _AsyncTactileRefinerBlock(
                    width=self.async_refiner_width,
                    num_heads=self.async_refiner_heads,
                    mlp_dim=self.async_refiner_mlp_dim,
                    rngs=rngs,
                )
                for _ in range(self.async_refiner_layers)
            ]
            self.async_delta_out = nnx.Linear(self.async_refiner_width, self.hand_action_dim, rngs=rngs)
            self.async_gate_out = nnx.Linear(self.async_refiner_width, 1, rngs=rngs)

        if self.async_tactile_flow_refiner_enabled:
            self.async_flow_action_proj = nnx.Linear(self.hand_action_dim, self.async_flow_refiner_width, rngs=rngs)
            self.async_flow_arm_context_proj = nnx.Linear(
                self.arm_action_dim, self.async_flow_refiner_width, rngs=rngs
            )
            self.async_flow_state_context_proj = nnx.Linear(config.action_dim, self.async_flow_refiner_width, rngs=rngs)
            self.async_flow_fresh_tactile_proj = nnx.Linear(
                self.student_width, self.async_flow_refiner_width, rngs=rngs
            )
            self.async_flow_future_contact_proj = nnx.Linear(
                self.student_width, self.async_flow_refiner_width, rngs=rngs
            )
            self.async_flow_time_mlp_in = nnx.Linear(
                self.async_flow_refiner_width, self.async_flow_refiner_width, rngs=rngs
            )
            self.async_flow_time_mlp_out = nnx.Linear(
                self.async_flow_refiner_width, self.async_flow_refiner_width, rngs=rngs
            )
            self.async_flow_step_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs.params(), (self.action_horizon, self.async_flow_refiner_width), dtype=jnp.float32
                )
            )
            self.async_flow_offset_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs.params(), (self.action_horizon + 1, self.async_flow_refiner_width), dtype=jnp.float32
                )
            )
            self.async_flow_refiner_blocks = [
                _AsyncTactileRefinerBlock(
                    width=self.async_flow_refiner_width,
                    num_heads=self.async_flow_refiner_heads,
                    mlp_dim=self.async_flow_refiner_mlp_dim,
                    rngs=rngs,
                )
                for _ in range(self.async_flow_refiner_layers)
            ]
            self.async_flow_velocity_out = nnx.Linear(
                self.async_flow_refiner_width, self.hand_action_dim, rngs=rngs
            )

        if self.cached_vlm_async_ae_enabled:
            self.cached_async_offset_embedding = nnx.Param(
                0.02
                * jax.random.normal(
                    rngs.params(), (self.action_horizon + 1, self.student_width), dtype=jnp.float32
                )
            )

        self.deterministic = True

    def _llm_streams(
        self,
        prefix: at.Array | None,
        student: at.Array | None,
        teacher: at.Array | None,
    ) -> list[at.Array | None]:
        return [prefix, student] if not self.use_teacher_ae else [prefix, student, teacher]

    def _llm_adarms(
        self,
        student: at.Array | None,
        teacher: at.Array | None,
    ) -> list[at.Array | None]:
        return [None, student] if not self.use_teacher_ae else [None, student, teacher]

    def _pad_or_crop_effort(
        self,
        effort: at.Float[at.Array, "b t d"],
        steps: int,
        *,
        from_end: bool,
    ) -> at.Float[at.Array, "b t d"]:
        if effort.shape[1] >= steps:
            return effort[:, -steps:, :] if from_end else effort[:, :steps, :]
        if effort.shape[1] == 0:
            raise ValueError("Pi0MORDualAlignForceFlow received an empty effort sequence.")
        pad_source = effort[:, :1, :] if from_end else effort[:, -1:, :]
        pad = jnp.repeat(pad_source, steps - effort.shape[1], axis=1)
        return jnp.concatenate([pad, effort], axis=1) if from_end else jnp.concatenate([effort, pad], axis=1)

    def _split_effort(
        self,
        observation: _model.Observation,
        *,
        require_future: bool,
        dtype: at.DTypeLike,
    ) -> tuple[at.Float[at.Array, "b h e"], at.Float[at.Array, "b f e"] | None]:
        if observation.effort is None:
            raise ValueError("Pi0MORDualAlignForceFlow requires `observation.effort`.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected_ndim = 5 if self.structured_tactile and self.tactile_points_per_finger > 1 else 4 if self.structured_tactile else 3
        if effort.ndim != expected_ndim:
            raise ValueError(f"Expected effort with {expected_ndim} dimensions, got {effort.shape}.")

        future_effort = None
        if effort.shape[1] > self.action_horizon:
            history_effort = effort[:, : effort.shape[1] - self.action_horizon, :]
            future_effort = effort[:, -self.action_horizon :, :]
        else:
            history_effort = effort

        if require_future and future_effort is None:
            raise ValueError(
                "Teacher training requires merged history+future effort. "
                f"Expected more than action_horizon={self.action_horizon} effort steps, got {effort.shape[1]}."
            )

        history_effort = self._pad_or_crop_effort(history_effort, self.force_input_frames, from_end=True)
        if future_effort is not None:
            future_effort = self._pad_or_crop_effort(future_effort, self.action_horizon, from_end=False)
        return history_effort, future_effort

    def _pool_history_tokens(self, tokens: at.Array, logits: at.Array | None) -> at.Array:
        if not self.pool_tactile_history:
            return tokens
        tokens = einops.rearrange(
            tokens,
            "b (h k) d -> b h k d",
            h=self.force_input_frames,
            k=self.tactile_tokens_per_step,
        )
        weights = jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float32)).astype(tokens.dtype)
        return jnp.einsum("h,bhkd->bkd", weights, tokens)

    def _pool_history_current_tokens(
        self,
        tokens: at.Array,
        logits: at.Array,
        type_embedding: at.Array,
    ) -> at.Array:
        tokens = einops.rearrange(
            tokens,
            "b (h k) d -> b h k d",
            h=self.force_input_frames,
            k=self.tactile_tokens_per_step,
        )
        if self.force_input_frames < 2:
            raise ValueError("pooled_current history mode requires at least two history frames.")
        past_tokens = tokens[:, :-1, :, :]
        current_tokens = tokens[:, -1, :, :]
        past_logits = jnp.asarray(logits[:-1], dtype=jnp.float32)
        past_weights = jax.nn.softmax(past_logits).astype(tokens.dtype)
        past_summary = jnp.einsum("h,bhkd->bkd", past_weights, past_tokens)
        type_embedding = jnp.asarray(type_embedding, dtype=tokens.dtype)
        past_summary = past_summary + type_embedding[0][None, None, :]
        current_tokens = current_tokens + type_embedding[1][None, None, :]
        return jnp.concatenate([past_summary, current_tokens], axis=1)

    def _project_history_force_student(
        self, history_effort: at.Array
    ) -> at.Array:
        if self.structured_tactile:
            if self.tactile_history_segments:
                return self.student_force_tokenizer.encode_temporal_segments(
                    history_effort,
                    jnp.asarray(self.history_times, dtype=jnp.float32),
                    num_segments=self.tactile_history_segments,
                    future=False,
                )
            tokens = self.student_force_tokenizer.encode_history(
                history_effort, jnp.asarray(self.history_times, dtype=jnp.float32)
            )
            if self.pool_tactile_history and self.cached_vlm_async_history_mode == "pooled_current":
                return self._pool_history_current_tokens(
                    tokens,
                    self.student_history_pool_logits.value,
                    self.student_history_type_embedding.value,
                )
            logits = self.student_history_pool_logits.value if self.pool_tactile_history else None
            return self._pool_history_tokens(tokens, logits)
        hidden = self.history_force_proj_student(einops.rearrange(history_effort, "b h e -> b (h e)"))
        return hidden[:, None, :]

    def initial_tactile_ttt_state(self, batch_size: int, *, dtype: jnp.dtype = jnp.float32) -> at.Array:
        if not self.tactile_ttt_enabled:
            raise ValueError("This model was not configured with TactileTTT.")
        return self.tactile_ttt.initial_state(batch_size, dtype=dtype)

    def _tactile_ttt_step(
        self,
        history_effort: at.Array,
        fast_weight: at.Array | None,
        *,
        active: at.Array | None = None,
    ) -> tuple[at.Array, at.Array, dict[str, at.Array]]:
        if not self.tactile_ttt_enabled:
            raise ValueError("This model was not configured with TactileTTT.")
        if fast_weight is None:
            fast_weight = self.initial_tactile_ttt_state(history_effort.shape[0])

        history_times = jnp.asarray(self.history_times, dtype=jnp.float32)
        write_tokens = self.student_force_tokenizer.encode_temporal_segments(
            history_effort,
            history_times,
            num_segments=self.tactile_ttt_write_segments,
            future=False,
        )
        current_tokens = self.student_force_tokenizer.encode_history(
            history_effort[:, -1:, ...],
            history_times[-1:],
        )
        return self.tactile_ttt.step(
            fast_weight,
            write_tokens,
            current_tokens,
            history_effort,
            active=active,
        )

    def _project_history_force_teacher(
        self, history_effort: at.Array
    ) -> at.Array:
        if self.structured_tactile:
            tokens = self.teacher_force_tokenizer.encode_history(
                history_effort, jnp.asarray(self.history_times, dtype=jnp.float32)
            )
            if self.pool_tactile_history and self.cached_vlm_async_history_mode == "pooled_current":
                return self._pool_history_current_tokens(
                    tokens,
                    self.teacher_history_pool_logits.value,
                    self.teacher_history_type_embedding.value,
                )
            logits = self.teacher_history_pool_logits.value if self.pool_tactile_history else None
            return self._pool_history_tokens(tokens, logits)
        hidden = self.history_force_proj_teacher(einops.rearrange(history_effort, "b h e -> b (h e)"))
        return hidden[:, None, :]

    def _project_future_force_teacher(
        self, future_effort: at.Array
    ) -> at.Array:
        if self.structured_tactile:
            return self.teacher_force_tokenizer.encode_future(
                future_effort, jnp.asarray(self.future_times, dtype=jnp.float32)
            )
        hidden = self.future_force_proj_teacher(einops.rearrange(future_effort, "b h e -> b (h e)"))
        return hidden[:, None, :]

    def _project_future_force_student_target(
        self, future_effort: at.Array
    ) -> at.Array:
        if self.structured_tactile:
            return self.student_force_tokenizer.encode_future(
                future_effort, jnp.asarray(self.future_times, dtype=jnp.float32)
            )
        raise ValueError("direct_future_tactile_align currently requires structured_tactile=True.")

    def _student_query_token(self, batch_size: int, dtype: jnp.dtype) -> at.Float[at.Array, "b 1 d"]:
        if self.disable_future_tactile:
            return jnp.zeros((batch_size, 0, self.student_width), dtype=dtype)
        if self.structured_tactile:
            query = (
                self.student_query_base.value[None, None, :]
                + self.student_query_segment_embedding.value[:, None, :]
                + self.student_query_finger_embedding.value[None, :, :]
            )
            query = einops.rearrange(query, "s k d -> (s k) d").astype(dtype)
            return jnp.broadcast_to(query[None, :, :], (batch_size, query.shape[0], query.shape[1]))
        query = jnp.asarray(self.student_query.value, dtype=dtype)
        return jnp.broadcast_to(query[None, None, :], (batch_size, 1, query.shape[0]))

    def _student_query_noise_scale(self, train_progress: at.Float[at.Array, ""] | float | None) -> at.Float[at.Array, ""]:
        if train_progress is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        progress = jnp.clip(jnp.asarray(train_progress, dtype=jnp.float32), 0.0, 1.0)
        start = jnp.asarray(self.student_future_query_noise_start_ratio, dtype=jnp.float32)
        end = jnp.asarray(self.student_future_query_noise_end_ratio, dtype=jnp.float32)
        max_scale = jnp.asarray(self.student_future_query_noise_scale_max, dtype=jnp.float32)
        ramp = (progress - start) / jnp.maximum(end - start, 1e-6)
        return max_scale * jnp.clip(ramp, 0.0, 1.0)

    def _student_future_query_tokens(
        self,
        batch_size: int,
        dtype: jnp.dtype,
        *,
        train: bool,
        noise_rng: at.KeyArrayLike | None,
        train_progress: at.Float[at.Array, ""] | float | None = None,
        query_noise_scale: at.Float[at.Array, ""] | float | None = None,
    ) -> tuple[
        at.Float[at.Array, "b 1 d"],
        at.Float[at.Array, "b n d"],
        at.Bool[at.Array, "b"],
        at.Bool[at.Array, "b n"],
        at.Float[at.Array, "b"],
    ]:
        future_force_query = self._student_query_token(batch_size, dtype)
        future_flow_queries = self._student_future_flow_tokens(batch_size, dtype)
        active_flow_token_count = future_flow_queries.shape[1]
        noise_scale_f32 = (
            self._student_query_noise_scale(train_progress)
            if query_noise_scale is None
            else jnp.asarray(query_noise_scale, dtype=jnp.float32)
        )
        noise_scale_f32 = jnp.maximum(noise_scale_f32, 0.0)
        noise_scale = noise_scale_f32.astype(dtype)
        if not train and query_noise_scale is None:
            return (
                future_force_query,
                future_flow_queries,
                jnp.ones((batch_size,), dtype=jnp.bool_),
                jnp.ones((batch_size, active_flow_token_count), dtype=jnp.bool_),
                jnp.zeros((batch_size,), dtype=jnp.float32),
            )
        if query_noise_scale is None and float(self.student_future_query_noise_scale_max) <= 0.0:
            return (
                future_force_query,
                future_flow_queries,
                jnp.ones((batch_size,), dtype=jnp.bool_),
                jnp.ones((batch_size, active_flow_token_count), dtype=jnp.bool_),
                jnp.zeros((batch_size,), dtype=jnp.float32),
            )
        if noise_rng is None:
            raise ValueError("noise_rng is required when training with student future query noise enabled.")

        force_noise_rng, flow_noise_rng = jax.random.split(noise_rng)
        force_rms = jnp.sqrt(
            jnp.mean(jnp.square(future_force_query.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6
        )
        flow_rms = jnp.sqrt(
            jnp.mean(jnp.square(future_flow_queries.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6
        )
        force_noise = noise_scale * force_rms.astype(dtype) * jax.random.normal(
            force_noise_rng, future_force_query.shape, dtype=dtype
        )
        flow_noise = noise_scale * flow_rms.astype(dtype) * jax.random.normal(
            flow_noise_rng, future_flow_queries.shape, dtype=dtype
        )

        future_force_query = future_force_query + force_noise
        future_flow_queries = future_flow_queries + flow_noise
        noised_token_rate = jnp.ones((batch_size,), dtype=jnp.float32) * jnp.where(
            noise_scale_f32 > 0.0, 1.0, 0.0
        )
        force_clean_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
        flow_clean_mask = jnp.ones((batch_size, active_flow_token_count), dtype=jnp.bool_)
        return future_force_query, future_flow_queries, force_clean_mask, flow_clean_mask, noised_token_rate

    def _project_prompt_distill(self, hidden: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t d"]:
        hidden = self.prompt_distill_proj_in(hidden)
        hidden = nnx.swish(hidden)
        return self.prompt_distill_proj_out(hidden)

    def _project_flow_distill(self, hidden: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t d"]:
        hidden = self.flow_distill_proj_in(hidden)
        hidden = nnx.swish(hidden)
        return self.flow_distill_proj_out(hidden)

    @staticmethod
    def _apply_loss_mask(
        losses: at.Float[at.Array, "b"],
        keep_mask: at.Bool[at.Array, "b"],
    ) -> at.Float[at.Array, "b"]:
        return jnp.where(keep_mask, losses, jnp.zeros_like(losses))

    @staticmethod
    def _masked_mean(
        losses: at.Float[at.Array, "b"],
        keep_mask: at.Bool[at.Array, "b"],
    ) -> at.Float[at.Array, ""]:
        weights = keep_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights) / denom

    @staticmethod
    def _cosine_distance(
        lhs: at.Float[at.Array, "b t d"],
        rhs: at.Float[at.Array, "b t d"],
    ) -> at.Float[at.Array, " b"]:
        lhs = lhs.astype(jnp.float32)
        rhs = rhs.astype(jnp.float32)
        lhs_norm = lhs / jnp.sqrt(jnp.sum(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6)
        rhs_norm = rhs / jnp.sqrt(jnp.sum(jnp.square(rhs), axis=-1, keepdims=True) + 1e-6)
        cosine = jnp.sum(lhs_norm * rhs_norm, axis=-1)
        return jnp.mean(1.0 - cosine, axis=-1)

    @staticmethod
    def _cosine_distance_masked(
        lhs: at.Float[at.Array, "b t d"],
        rhs: at.Float[at.Array, "b t d"],
        token_mask: at.Bool[at.Array, "b t"],
    ) -> at.Float[at.Array, " b"]:
        lhs = lhs.astype(jnp.float32)
        rhs = rhs.astype(jnp.float32)
        lhs_norm = lhs / jnp.sqrt(jnp.sum(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6)
        rhs_norm = rhs / jnp.sqrt(jnp.sum(jnp.square(rhs), axis=-1, keepdims=True) + 1e-6)
        cosine = jnp.sum(lhs_norm * rhs_norm, axis=-1)
        losses = 1.0 - cosine
        weights = token_mask.astype(losses.dtype)
        denom = jnp.maximum(jnp.sum(weights, axis=-1), jnp.asarray(1.0, dtype=losses.dtype))
        return jnp.sum(losses * weights, axis=-1) / denom

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def _embed_action_tokens(
        self,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        expert: str,
    ) -> tuple[at.Float[at.Array, "b ah d"], at.Float[at.Array, "b d"] | None]:
        if expert == "student":
            action_tokens = self.action_in_proj_student(noisy_actions)
            width = self.action_in_proj_student.out_features
            time_mlp_in = self.student_time_mlp_in
            time_mlp_out = self.student_time_mlp_out
        elif expert == "teacher":
            action_tokens = self.action_in_proj_teacher(noisy_actions)
            width = self.action_in_proj_teacher.out_features
            time_mlp_in = self.teacher_time_mlp_in
            time_mlp_out = self.teacher_time_mlp_out
        else:
            raise ValueError(f"Unknown expert: {expert}")

        time_emb = posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            return action_tokens, time_emb

        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=noisy_actions.shape[1])
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = time_mlp_out(action_time_tokens)
        return action_time_tokens, None

    def _split_action_inputs(self, actions: _model.Actions) -> tuple[_model.Actions, _model.Actions]:
        """Keep arm and hand values in separate full-width tensors for shared projections."""
        arm_actions = jnp.zeros_like(actions).at[..., : self.arm_action_dim].set(
            actions[..., : self.arm_action_dim]
        )
        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        hand_actions = jnp.zeros_like(actions).at[..., hand_slice].set(actions[..., hand_slice])
        padding_start = self.hand_action_start + self.hand_action_dim
        arm_actions = arm_actions.at[..., padding_start:].set(actions[..., padding_start:])
        return arm_actions, hand_actions

    def _suffix_slices(self, action_len: int) -> dict[str, slice]:
        """Return semantic token slices for the configured suffix ordering."""
        if not self.structured_tactile:
            raise ValueError("Semantic suffix slices require structured tactile tokens.")
        observation_end = self.state_token_count + self.history_force_token_count
        active_flow_count = self.flow_token_count if self.use_future_flow else 0

        if self.arm_hand_mask_attention:
            arm_slice = slice(observation_end, observation_end + action_len)
            future_force_start = arm_slice.stop
            flow_start = future_force_start + self.future_force_token_count
            hand_start = flow_start + active_flow_count
            return {
                "arm": arm_slice,
                "future_force": slice(future_force_start, flow_start),
                "future_flow": slice(flow_start, hand_start),
                "hand": slice(hand_start, hand_start + action_len),
            }

        future_force_start = observation_end
        flow_start = future_force_start + self.future_force_token_count
        action_start = flow_start + active_flow_count
        return {
            "future_force": slice(future_force_start, flow_start),
            "future_flow": slice(flow_start, action_start),
            "action": slice(action_start, action_start + action_len),
        }

    def _decode_base_action_velocity(self, hidden: at.Array, *, expert: str) -> _model.Actions:
        projection = (
            self.action_out_proj_student
            if expert == "student"
            else self.action_out_proj_teacher
            if expert == "teacher"
            else None
        )
        if projection is None:
            raise ValueError(f"Unknown expert: {expert}")
        if not self.arm_hand_mask_attention:
            return projection(hidden[:, -self.action_horizon :])

        slices = self._suffix_slices(self.action_horizon)
        arm_full = projection(hidden[:, slices["arm"]])
        hand_full = projection(hidden[:, slices["hand"]])
        velocity = jnp.zeros_like(arm_full)
        velocity = velocity.at[..., : self.arm_action_dim].set(arm_full[..., : self.arm_action_dim])
        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        velocity = velocity.at[..., hand_slice].set(hand_full[..., hand_slice])
        padding_start = self.hand_action_start + self.hand_action_dim
        return velocity.at[..., padding_start:].set(arm_full[..., padding_start:])

    def _tactile_refine_student_velocity(
        self,
        hidden: at.Array,
        base_velocity: _model.Actions,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        if not self.tactile_refiner_enabled:
            return base_velocity, {}
        if not self.arm_hand_mask_attention:
            raise ValueError("Tactile refiner requires arm_hand_mask_attention=True.")

        slices = self._suffix_slices(self.action_horizon)
        history_slice = slice(
            self.state_token_count,
            self.state_token_count + self.history_force_token_count,
        )
        hand_hidden = hidden[:, slices["hand"]]
        arm_hidden = hidden[:, slices["arm"]]
        history_hidden = hidden[:, history_slice]
        future_hidden = hidden[:, slices["future_force"]]

        history_context = jnp.mean(history_hidden, axis=1)
        future_context = jnp.mean(future_hidden, axis=1)
        history_context = jnp.broadcast_to(history_context[:, None, :], hand_hidden.shape)
        future_context = jnp.broadcast_to(future_context[:, None, :], hand_hidden.shape)

        synergy_input = jnp.concatenate(
            [hand_hidden, arm_hidden, history_context, future_context],
            axis=-1,
        )
        synergy_hidden = nnx.swish(self.hand_synergy_in(synergy_input))
        z_hand = self.hand_synergy_out(synergy_hidden)
        coarse_hand = self.hand_synergy_decoder_out(nnx.swish(self.hand_synergy_decoder_in(z_hand)))

        refiner_tokens = jnp.concatenate(
            [
                self.refiner_history_proj(history_hidden),
                self.refiner_future_proj(future_hidden),
                self.refiner_arm_proj(arm_hidden),
                self.refiner_synergy_proj(z_hand),
                self.refiner_coarse_proj(coarse_hand),
                self.refiner_hand_proj(hand_hidden),
            ],
            axis=1,
        )
        for block in self.tactile_refiner_blocks:
            refiner_tokens = block(refiner_tokens)

        hand_refined = refiner_tokens[:, -self.action_horizon :]
        delta_hand = self.tactile_refiner_delta_scale * self.refiner_delta_out(hand_refined)
        gate = jax.nn.sigmoid(self.refiner_gate_out(hand_refined) + self.tactile_refiner_gate_bias)

        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        base_hand = base_velocity[..., hand_slice]
        final_hand = base_hand + gate * delta_hand
        final_velocity = base_velocity.at[..., hand_slice].set(final_hand)
        return final_velocity, {
            "z_hand": z_hand,
            "coarse_hand": coarse_hand,
            "delta_hand": delta_hand,
            "gate": gate,
        }

    def _decode_student_action_velocity_with_stats(
        self, hidden: at.Array
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        base_velocity = self._decode_base_action_velocity(hidden, expert="student")
        return self._tactile_refine_student_velocity(hidden, base_velocity)

    def _decode_action_velocity(self, hidden: at.Array, *, expert: str) -> _model.Actions:
        base_velocity = self._decode_base_action_velocity(hidden, expert=expert)
        if expert == "student":
            refined_velocity, _ = self._tactile_refine_student_velocity(hidden, base_velocity)
            return refined_velocity
        return base_velocity

    def _action_losses(self, prediction: _model.Actions, target: _model.Actions) -> tuple[at.Array, at.Array, at.Array]:
        arm_loss = jnp.mean(
            jnp.square(prediction[..., : self.arm_action_dim] - target[..., : self.arm_action_dim]),
            axis=(-2, -1),
        )
        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        hand_loss = jnp.mean(
            jnp.square(prediction[..., hand_slice] - target[..., hand_slice]),
            axis=(-2, -1),
        )
        physical_prediction = jnp.concatenate(
            [prediction[..., : self.arm_action_dim], prediction[..., hand_slice]], axis=-1
        )
        physical_target = jnp.concatenate([target[..., : self.arm_action_dim], target[..., hand_slice]], axis=-1)
        total_loss = jnp.mean(jnp.square(physical_prediction - physical_target), axis=(-2, -1))
        return total_loss, arm_loss, hand_loss

    def _action_loss_with_offset_mask(
        self,
        prediction: _model.Actions,
        target: _model.Actions,
        offset: at.Array,
    ) -> at.Array:
        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        physical_prediction = jnp.concatenate(
            [prediction[..., : self.arm_action_dim], prediction[..., hand_slice]], axis=-1
        )
        physical_target = jnp.concatenate([target[..., : self.arm_action_dim], target[..., hand_slice]], axis=-1)
        if self.cached_vlm_async_loss_mask == "full":
            return jnp.mean(jnp.square(physical_prediction - physical_target), axis=(-2, -1))
        mask = (
            jnp.arange(self.action_horizon, dtype=jnp.int32)[None, :, None]
            >= jnp.asarray(offset, dtype=jnp.int32)
        ).astype(physical_prediction.dtype)
        denom = jnp.maximum(jnp.sum(mask), 1.0) * float(self.arm_action_dim + self.hand_action_dim)
        return jnp.sum(jnp.square((physical_prediction - physical_target) * mask), axis=(-2, -1)) / denom

    def _history_effort_for_offset(
        self,
        history_effort: at.Array,
        future_effort: at.Array,
        offset: at.Array,
    ) -> at.Array:
        effort = jnp.concatenate([history_effort, future_effort], axis=1)
        return jax.lax.dynamic_slice_in_dim(
            effort,
            jnp.asarray(offset, dtype=jnp.int32),
            self.force_input_frames,
            axis=1,
        )

    def _cached_async_token_mask(self, batch_size: int, offset: at.Array) -> at.Array:
        if self.cached_vlm_async_loss_mask == "full":
            token_mask = jnp.ones((self.future_force_token_count,), dtype=jnp.bool_)
        else:
            segment_ids = jnp.repeat(
                jnp.arange(self.future_tactile_segments, dtype=jnp.int32),
                self.tactile_tokens_per_step,
            )
            segment_start = segment_ids * self.future_steps_per_segment
            token_mask = segment_start >= jnp.asarray(offset, dtype=jnp.int32)
        return jnp.broadcast_to(token_mask[None, :], (batch_size, token_mask.shape[0]))

    def _cached_async_future_query_override(
        self,
        base_future_hidden: at.Array,
        offset: at.Array,
        dtype: jnp.dtype,
    ) -> at.Array:
        query = self._student_query_token(base_future_hidden.shape[0], dtype)
        if not self.cached_vlm_async_use_predicted_prefix_queries:
            return query
        prefix_segments = jnp.asarray(offset, dtype=jnp.int32) // int(self.future_steps_per_segment)
        token_ids = jnp.arange(self.future_force_token_count, dtype=jnp.int32)
        prefix_token_count = prefix_segments * int(self.tactile_tokens_per_step)
        prefix_mask = (token_ids < prefix_token_count)[None, :, None]
        prefix = jax.lax.stop_gradient(base_future_hidden).astype(dtype)
        return jnp.where(prefix_mask, prefix, query)

    def _future_patch_distribution_target(self, future_effort: at.Array) -> at.Array:
        if (
            self.tactile_patch_aux_loss_weight <= 0
            or not self.tactile_patch_informed_tokenizer
            or not hasattr(self.teacher_force_tokenizer, "patch_contact_distribution")
        ):
            return jnp.zeros(
                (future_effort.shape[0], self.future_force_token_count, self.tactile_num_patches),
                dtype=jnp.float32,
            )

        patch_dist = self.teacher_force_tokenizer.patch_contact_distribution(future_effort)
        batch_size = patch_dist.shape[0]
        patch_dist = patch_dist.reshape(
            batch_size,
            self.future_tactile_segments,
            self.future_steps_per_segment,
            self.tactile_num_fingers,
            self.tactile_num_patches,
        )
        weights = jax.nn.softmax(self.teacher_force_tokenizer.segment_pool_logits.value.astype(jnp.float32))
        segment_dist = jnp.einsum("p,bspfr->bsfr", weights, patch_dist)
        segment_dist = segment_dist / jnp.maximum(jnp.sum(segment_dist, axis=-1, keepdims=True), 1e-6)
        return segment_dist.reshape(batch_size, self.future_force_token_count, self.tactile_num_patches)

    def _patch_distribution_aux_loss(
        self,
        *,
        teacher_future_tokens: at.Array,
        future_effort: at.Array,
    ) -> at.Array:
        if self.tactile_patch_aux_loss_weight <= 0:
            return jnp.zeros((teacher_future_tokens.shape[0],), dtype=teacher_future_tokens.dtype)
        target_dist = self._future_patch_distribution_target(future_effort)
        logits = self.teacher_patch_dist_head(teacher_future_tokens)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        return -jnp.mean(jnp.sum(jax.lax.stop_gradient(target_dist) * log_probs, axis=-1), axis=-1)

    def _fresh_tactile_tokens_for_async_refiner(
        self,
        history_effort: at.Array,
        future_effort: at.Array,
        offset: at.Array,
    ) -> at.Array:
        if not self.structured_tactile:
            raise ValueError("Async tactile refiner currently requires structured tactile effort.")
        effort = jnp.concatenate([history_effort, future_effort], axis=1)
        fresh_effort = jax.lax.dynamic_slice_in_dim(
            effort,
            jnp.asarray(offset, dtype=jnp.int32),
            self.force_input_frames,
            axis=1,
        )
        fresh_times = (
            jnp.arange(self.force_input_frames, dtype=jnp.float32)
            - float(self.force_input_frames - 1)
        ) / float(self.tactile_sample_hz)
        return self.student_force_tokenizer.encode_history(fresh_effort, fresh_times)

    def _prefix_kv_cache(
        self,
        prefix_tokens: at.Array,
        prefix_mask: at.Array,
        prefix_ar_mask: at.Array,
    ) -> tuple[object, at.Array]:
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            self._llm_streams(prefix_tokens, None, None),
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=self._llm_adarms(None, None),
        )
        return kv_cache, prefix_positions

    def _student_suffix_with_prefix_cache(
        self,
        *,
        observation: _model.Observation,
        history_effort: at.Array,
        prefix_mask: at.Array,
        prefix_kv_cache: object,
        noisy_actions: _model.Actions,
        timestep: at.Array,
    ) -> tuple[at.Array, at.Array, at.Array, at.Array | None]:
        student_tokens, student_mask, student_ar_mask, student_adarms, *_ = self.embed_student_suffix(
            observation,
            history_effort,
            noisy_actions,
            timestep,
            train=False,
            noise_rng=None,
            query_noise_scale=None,
        )
        student_attn_mask = make_attn_mask(student_mask, student_ar_mask)
        prefix_to_student = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
        prefix_to_student = jnp.logical_and(prefix_to_student, student_mask[:, :, None])
        full_attn_mask = jnp.concatenate([prefix_to_student, student_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(student_mask, axis=-1) - 1
        outputs, kv_cache = self.PaliGemma.llm(
            self._llm_streams(None, student_tokens, None),
            mask=full_attn_mask,
            positions=positions,
            kv_cache=prefix_kv_cache,
            adarms_cond=self._llm_adarms(student_adarms, None),
        )
        _, student_out, _ = outputs[:3]
        return student_out, kv_cache, student_mask, student_adarms

    def _student_suffix_with_prefix_cache_multilayer(
        self,
        *,
        observation: _model.Observation,
        history_effort: at.Array,
        prefix_mask: at.Array,
        prefix_kv_cache: object,
        noisy_actions: _model.Actions,
        timestep: at.Array,
        future_force_query_override: at.Array | None = None,
        async_offset: at.Array | None = None,
    ) -> tuple[at.Array, tuple[at.Array, ...]]:
        student_tokens, student_mask, student_ar_mask, student_adarms, *_ = self.embed_student_suffix(
            observation,
            history_effort,
            noisy_actions,
            timestep,
            train=False,
            noise_rng=None,
            query_noise_scale=None,
            future_force_query_override=future_force_query_override,
            async_offset=async_offset,
        )
        student_attn_mask = make_attn_mask(student_mask, student_ar_mask)
        prefix_to_student = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
        prefix_to_student = jnp.logical_and(prefix_to_student, student_mask[:, :, None])
        full_attn_mask = jnp.concatenate([prefix_to_student, student_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(student_mask, axis=-1) - 1
        (outputs, selected_layers), _ = self.PaliGemma.llm(
            self._llm_streams(None, student_tokens, None),
            mask=full_attn_mask,
            positions=positions,
            kv_cache=prefix_kv_cache,
            adarms_cond=self._llm_adarms(student_adarms, None),
            return_layer_indices=self.distill_layer_indices,
        )
        return outputs[1], tuple(layer[1] for layer in selected_layers)

    def _async_refine_hand_action(
        self,
        *,
        base_action: _model.Actions,
        target_action: _model.Actions,
        student_contact_hidden: at.Array,
        history_effort: at.Array,
        future_effort: at.Array,
        state: at.Array,
        offset: at.Array,
    ) -> tuple[at.Array, dict[str, at.Array]]:
        if not self.async_tactile_refiner_enabled:
            zeros = jnp.zeros(base_action.shape[0], dtype=base_action.dtype)
            return zeros, {
                "loss": zeros,
                "delta_reg": zeros,
                "gate_reg": zeros,
                "gate_mean": zeros,
                "delta_abs_mean": zeros,
                "offset": jnp.asarray(offset, dtype=jnp.float32),
            }

        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        base_action = jax.lax.stop_gradient(base_action)
        student_contact_hidden = jax.lax.stop_gradient(student_contact_hidden)

        base_hand = base_action[..., hand_slice]
        base_arm = base_action[..., : self.arm_action_dim]
        target_hand = target_action[..., hand_slice]

        time_embedding = jnp.asarray(self.async_time_embedding.value, dtype=base_hand.dtype)
        offset_embedding = jnp.asarray(self.async_offset_embedding.value, dtype=base_hand.dtype)[offset]
        hand_query = (
            self.async_hand_query_proj(base_hand)
            + time_embedding[None, :, :]
            + offset_embedding[None, None, :]
        )

        fresh_tactile_tokens = jax.lax.stop_gradient(
            self._fresh_tactile_tokens_for_async_refiner(history_effort, future_effort, offset)
        )
        arm_context = self.async_arm_context_proj(base_arm) + time_embedding[None, :, :]
        state_context = self.async_state_context_proj(jax.lax.stop_gradient(state))[:, None, :]
        context = jnp.concatenate(
            [
                self.async_fresh_tactile_proj(fresh_tactile_tokens),
                self.async_future_contact_proj(student_contact_hidden),
                arm_context,
                state_context,
            ],
            axis=1,
        )

        for block in self.async_refiner_blocks:
            hand_query = block(hand_query, context)

        delta_hand = self.async_refiner_delta_scale * self.async_delta_out(hand_query)
        gate = jax.nn.sigmoid(self.async_gate_out(hand_query) + self.async_refiner_gate_bias)
        suffix_mask = (
            jnp.arange(self.action_horizon, dtype=jnp.int32)[None, :, None]
            >= jnp.asarray(offset, dtype=jnp.int32)
        ).astype(base_hand.dtype)
        refined_hand = base_hand + suffix_mask * gate * delta_hand

        suffix_steps = jnp.maximum(jnp.sum(suffix_mask), 1.0)
        denom = suffix_steps * float(self.hand_action_dim)
        async_loss = jnp.sum(jnp.square((refined_hand - target_hand) * suffix_mask), axis=(-2, -1)) / denom
        delta_reg = jnp.sum(jnp.square(delta_hand * suffix_mask), axis=(-2, -1)) / denom
        gate_reg = jnp.sum(gate * suffix_mask, axis=(-2, -1)) / suffix_steps
        return async_loss, {
            "loss": async_loss,
            "delta_reg": delta_reg,
            "gate_reg": gate_reg,
            "gate_mean": gate_reg,
            "delta_abs_mean": jnp.sum(jnp.abs(delta_hand) * suffix_mask, axis=(-2, -1)) / denom,
            "offset": jnp.asarray(offset, dtype=jnp.float32),
        }

    def _async_refine_hand_velocity(
        self,
        *,
        noisy_actions: _model.Actions,
        target_velocity: _model.Actions,
        student_contact_hidden: at.Array,
        history_effort: at.Array,
        future_effort: at.Array,
        state: at.Array,
        offset: at.Array,
        timestep: at.Array,
    ) -> tuple[at.Array, dict[str, at.Array]]:
        if not self.async_tactile_flow_refiner_enabled:
            zeros = jnp.zeros(noisy_actions.shape[0], dtype=noisy_actions.dtype)
            return zeros, {
                "loss": zeros,
                "velocity_abs_mean": zeros,
                "offset": jnp.asarray(offset, dtype=jnp.float32),
                "timestep": jnp.zeros_like(zeros),
            }

        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        student_contact_hidden = jax.lax.stop_gradient(student_contact_hidden)

        noisy_hand = noisy_actions[..., hand_slice]
        noisy_arm = noisy_actions[..., : self.arm_action_dim]
        target_hand_velocity = target_velocity[..., hand_slice]

        step_embedding = jnp.asarray(self.async_flow_step_embedding.value, dtype=noisy_hand.dtype)
        offset_embedding = jnp.asarray(self.async_flow_offset_embedding.value, dtype=noisy_hand.dtype)[offset]
        time_embedding = posemb_sincos(
            timestep,
            self.async_flow_refiner_width,
            min_period=4e-3,
            max_period=4.0,
        )
        time_embedding = self.async_flow_time_mlp_out(nnx.swish(self.async_flow_time_mlp_in(time_embedding)))

        hand_query = (
            self.async_flow_action_proj(noisy_hand)
            + step_embedding[None, :, :]
            + offset_embedding[None, None, :]
            + time_embedding[:, None, :]
        )

        fresh_tactile_tokens = self._fresh_tactile_tokens_for_async_refiner(history_effort, future_effort, offset)
        arm_context = self.async_flow_arm_context_proj(noisy_arm) + step_embedding[None, :, :]
        state_context = self.async_flow_state_context_proj(jax.lax.stop_gradient(state))[:, None, :]
        context = jnp.concatenate(
            [
                self.async_flow_fresh_tactile_proj(fresh_tactile_tokens),
                self.async_flow_future_contact_proj(student_contact_hidden),
                arm_context,
                state_context,
            ],
            axis=1,
        )

        for block in self.async_flow_refiner_blocks:
            hand_query = block(hand_query, context)

        predicted_hand_velocity = self.async_flow_velocity_out(hand_query)
        suffix_mask = (
            jnp.arange(self.action_horizon, dtype=jnp.int32)[None, :, None]
            >= jnp.asarray(offset, dtype=jnp.int32)
        ).astype(noisy_hand.dtype)
        suffix_steps = jnp.maximum(jnp.sum(suffix_mask), 1.0)
        denom = suffix_steps * float(self.hand_action_dim)
        flow_loss = (
            jnp.sum(jnp.square((predicted_hand_velocity - target_hand_velocity) * suffix_mask), axis=(-2, -1))
            / denom
        )
        velocity_abs_mean = jnp.sum(jnp.abs(predicted_hand_velocity) * suffix_mask, axis=(-2, -1)) / denom
        return flow_loss, {
            "loss": flow_loss,
            "velocity_abs_mean": velocity_abs_mean,
            "offset": jnp.asarray(offset, dtype=jnp.float32),
            "timestep": timestep,
        }

    def _student_future_flow_tokens(self, batch_size: int, dtype: jnp.dtype) -> at.Float[at.Array, "b n d"]:
        if not self.use_future_flow:
            return jnp.zeros((batch_size, 0, self.student_width), dtype=dtype)
        query = jnp.asarray(self.student_future_flow_query.value, dtype=dtype)
        return jnp.broadcast_to(query[None, :, :], (batch_size, query.shape[0], query.shape[1]))

    def _build_suffix_ar_mask(self, action_len: int) -> at.Bool[at.Array, " s"]:
        active_flow_token_count = self.flow_token_count if self.use_future_flow else 0
        if self.structured_tactile:
            observation_count = self.state_token_count + self.history_force_token_count
            future_count = self.future_force_token_count + active_flow_token_count
            observation_mask = [True] + [False] * (observation_count - 1)
            future_mask = ([True] + [False] * (future_count - 1)) if future_count > 0 else []
            action_mask = [True] + [False] * (action_len - 1)
            if self.arm_hand_mask_attention:
                return jnp.array(
                    observation_mask
                    + action_mask
                    + future_mask
                    + action_mask
                )
            return jnp.array(
                observation_mask
                + future_mask
                + action_mask
            )
        observation_count = self.history_force_token_count + self.state_token_count
        future_count = self.future_force_token_count + active_flow_token_count
        observation_mask = [False] * observation_count
        future_mask = ([True] + [False] * (future_count - 1)) if future_count > 0 else []
        action_mask = [True] + [False] * (action_len - 1)
        return jnp.array(observation_mask + future_mask + action_mask)

    @staticmethod
    def _restore_aux_images(
        processed: _model.Observation,
        original_flow_img: at.Array | None,
        original_wrist_flow_img: at.Array | None,
        original_future_rgb_img: at.Array | None,
        original_future_wrist_rgb_img: at.Array | None,
        original_scene_flow: at.Array | None = None,
    ) -> _model.Observation:
        updates = {}
        if original_flow_img is not None:
            updates["flow_img"] = original_flow_img
        if original_wrist_flow_img is not None:
            updates["wrist_flow_img"] = original_wrist_flow_img
        if original_future_rgb_img is not None:
            updates["future_rgb_img"] = original_future_rgb_img
        if original_future_wrist_rgb_img is not None:
            updates["future_wrist_rgb_img"] = original_future_wrist_rgb_img
        if original_scene_flow is not None:
            updates["scene_flow"] = original_scene_flow
        if not updates:
            return processed
        return processed.replace(**updates)

    @staticmethod
    def _require_aux_image(
        image: at.Array | None,
        *,
        field_name: str,
        mode_name: str,
    ) -> at.Array:
        if image is None:
            raise ValueError(f"Pi0LatentFlow requires `{field_name}` for {mode_name}.")
        return image

    def _get_future_visual_images(self, obs: _model.Observation) -> tuple[at.Array, at.Array]:
        if self.use_future_rgb_instead_of_flow:
            return (
                self._require_aux_image(
                    obs.future_rgb_img,
                    field_name="observation.future_rgb_img",
                    mode_name="RGB ablation when `use_future_rgb_instead_of_flow=True`",
                ),
                self._require_aux_image(
                    obs.future_wrist_rgb_img,
                    field_name="observation.future_wrist_rgb_img",
                    mode_name="RGB ablation when `use_future_rgb_instead_of_flow=True`",
                ),
            )
        return (
            self._require_aux_image(
                obs.flow_img,
                field_name="observation.flow_img",
                mode_name="flow distillation",
            ),
            self._require_aux_image(
                obs.wrist_flow_img,
                field_name="observation.wrist_flow_img",
                mode_name="flow distillation",
            ),
        )

    def _encode_future_visual_image(self, image: at.Float[at.Array, "b h w c"]) -> at.Float[at.Array, "b s d"]:
        x = jnp.asarray(image, dtype=jnp.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected future visual image with shape [B, H, W, C], got {x.shape}.")
        if x.shape[-1] != self.future_flow_channels:
            raise ValueError(
                f"Expected future visual image with {self.future_flow_channels} channels, got shape={x.shape}."
            )
        # FlaxAutoencoderKL.encode expects BCHW input and internally converts to NHWC.
        x = jnp.transpose(x, (0, 3, 1, 2))

        flow_vae, flow_vae_params = _get_flow_vae(self.flow_vae_name)
        posterior = flow_vae.apply(
            {"params": flow_vae_params},
            x,
            deterministic=True,
            method=flow_vae.encode,
        ).latent_dist
        x = posterior.mode() * flow_vae.config.scaling_factor
        x = jax.lax.stop_gradient(x)
        patch = self.flow_vae_patch_merge_factor
        if x.shape[1] % patch != 0 or x.shape[2] % patch != 0:
            raise ValueError(
                "Future visual latent spatial size must be divisible by "
                f"{patch}, got shape={x.shape}."
            )
        x = einops.rearrange(
            x,
            "b (h ph) (w pw) c -> b h w (ph pw c)",
            ph=patch,
            pw=patch,
        )
        latent_tokens = einops.rearrange(x, "b h w c -> b (h w) c")
        latent_tokens = self.flow_vae_proj_in(latent_tokens)
        latent_tokens = nnx.swish(latent_tokens)
        latent_tokens = self.flow_vae_proj_out(latent_tokens)
        return self.flow_vae_norm(latent_tokens)

    def _compress_future_flows(self, obs: _model.Observation) -> at.Float[at.Array, "b n d"]:
        if not self.use_future_flow:
            batch_size = obs.state.shape[0]
            return jnp.zeros((batch_size, 0, self.teacher_width), dtype=obs.state.dtype)
        if self.future_flow_source == "scene_flow":
            return self._compress_future_scene_flow(obs)

        future_images = self._get_future_visual_images(obs)
        latent_tokens = jnp.concatenate(
            [self._encode_future_visual_image(image) for image in future_images],
            axis=1,
        )
        query = jnp.asarray(self.teacher_future_flow_query.value, dtype=latent_tokens.dtype)
        query = query + jnp.asarray(self.flow_token_embedding.value, dtype=latent_tokens.dtype)
        query = jnp.broadcast_to(query[None, :, :], (latent_tokens.shape[0], query.shape[0], query.shape[1]))
        query = self.flow_vae_query_proj(query)
        keys = self.flow_vae_key_proj(latent_tokens)
        values = self.flow_vae_value_proj(latent_tokens)

        logits = jnp.einsum("bqd,bkd->bqk", query, keys)
        logits = logits / jnp.sqrt(jnp.asarray(self.teacher_width, dtype=logits.dtype))
        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(values.dtype)
        return jnp.einsum("bqk,bkd->bqd", attn, values)

    def _compress_future_scene_flow(self, obs: _model.Observation) -> at.Float[at.Array, "b n d"]:
        if obs.scene_flow is None:
            raise ValueError("Pi0LatentFlow requires `observation.scene_flow` when future_flow_source='scene_flow'.")
        x = jnp.asarray(obs.scene_flow, dtype=jnp.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected scene_flow with shape [B, N, D], got {x.shape}.")
        if x.shape[-1] < self.scene_flow_input_dim:
            pad = jnp.zeros((*x.shape[:-1], self.scene_flow_input_dim - x.shape[-1]), dtype=x.dtype)
            x = jnp.concatenate([x, pad], axis=-1)
        elif x.shape[-1] > self.scene_flow_input_dim:
            x = x[..., : self.scene_flow_input_dim]

        point_tokens = self.scene_flow_proj_in(x)
        point_tokens = nnx.swish(point_tokens)
        point_tokens = self.scene_flow_proj_out(point_tokens)
        point_tokens = self.scene_flow_norm(point_tokens)

        query = jnp.asarray(self.teacher_future_flow_query.value, dtype=point_tokens.dtype)
        query = query + jnp.asarray(self.flow_token_embedding.value, dtype=point_tokens.dtype)
        query = jnp.broadcast_to(query[None, :, :], (point_tokens.shape[0], query.shape[0], query.shape[1]))
        query = self.flow_vae_query_proj(query)
        keys = self.flow_vae_key_proj(point_tokens)
        values = self.flow_vae_value_proj(point_tokens)

        logits = jnp.einsum("bqd,bkd->bqk", query, keys)
        logits = logits / jnp.sqrt(jnp.asarray(self.teacher_width, dtype=logits.dtype))
        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(values.dtype)
        return jnp.einsum("bqk,bkd->bqd", attn, values)

    @at.typecheck
    def embed_student_suffix(
        self,
        obs: _model.Observation,
        history_effort: at.Array,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        train: bool = False,
        noise_rng: at.KeyArrayLike | None = None,
        train_progress: at.Float[at.Array, ""] | float | None = None,
        query_noise_scale: at.Float[at.Array, ""] | float | None = None,
        future_force_query_override: at.Array | None = None,
        history_token_override: at.Array | None = None,
        async_offset: at.Array | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
        at.Bool[at.Array, "b"],
        at.Bool[at.Array, "b n"],
        at.Float[at.Array, "b"],
    ]:
        history_token = (
            self._project_history_force_student(history_effort)
            if history_token_override is None
            else jnp.asarray(history_token_override)
        )
        future_force_query, future_flow_queries, force_clean_mask, flow_clean_mask, noised_token_rate = self._student_future_query_tokens(
            obs.state.shape[0],
            history_token.dtype,
            train=train,
            noise_rng=noise_rng,
            train_progress=train_progress,
            query_noise_scale=query_noise_scale,
        )
        if future_force_query_override is not None:
            future_force_query = jnp.asarray(future_force_query_override, dtype=history_token.dtype)
        if async_offset is not None and self.cached_vlm_async_ae_enabled:
            offset_embedding = jnp.asarray(self.cached_async_offset_embedding.value, dtype=history_token.dtype)[
                jnp.asarray(async_offset, dtype=jnp.int32)
            ]
            history_token = history_token + offset_embedding[None, None, :]
            future_force_query = future_force_query + offset_embedding[None, None, :]
        state_token = (
            jnp.zeros((obs.state.shape[0], 0, self.student_width), dtype=history_token.dtype)
            if self.pi05
            else self.state_proj_student(obs.state)[:, None, :]
        )
        if self.arm_hand_mask_attention:
            arm_actions, hand_actions = self._split_action_inputs(noisy_actions)
            arm_tokens, adarms_cond = self._embed_action_tokens(arm_actions, timestep, expert="student")
            hand_tokens, _ = self._embed_action_tokens(hand_actions, timestep, expert="student")
        else:
            action_tokens, adarms_cond = self._embed_action_tokens(noisy_actions, timestep, expert="student")

        if self.arm_hand_mask_attention:
            suffix_parts = [
                state_token,
                history_token,
                arm_tokens,
                future_force_query,
                future_flow_queries,
                hand_tokens,
            ]
        else:
            suffix_parts = (
                [state_token, history_token, future_force_query, future_flow_queries, action_tokens]
                if self.structured_tactile
                else [history_token, state_token, future_force_query, future_flow_queries, action_tokens]
            )
        tokens = jnp.concatenate(suffix_parts, axis=1)
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = self._build_suffix_ar_mask(noisy_actions.shape[1])
        return tokens, input_mask, ar_mask, adarms_cond, force_clean_mask, flow_clean_mask, noised_token_rate

    @at.typecheck
    def embed_teacher_suffix(
        self,
        obs: _model.Observation,
        history_effort: at.Array,
        future_effort: at.Array,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        history_token = self._project_history_force_teacher(history_effort)
        future_force_token = self._project_future_force_teacher(future_effort)
        future_flow_tokens = self._compress_future_flows(obs)
        state_token = (
            jnp.zeros((obs.state.shape[0], 0, self.teacher_width), dtype=history_token.dtype)
            if self.pi05
            else self.state_proj_teacher(obs.state)[:, None, :]
        )
        if self.arm_hand_mask_attention:
            arm_actions, hand_actions = self._split_action_inputs(noisy_actions)
            arm_tokens, adarms_cond = self._embed_action_tokens(arm_actions, timestep, expert="teacher")
            hand_tokens, _ = self._embed_action_tokens(hand_actions, timestep, expert="teacher")
        else:
            action_tokens, adarms_cond = self._embed_action_tokens(noisy_actions, timestep, expert="teacher")

        if self.arm_hand_mask_attention:
            suffix_parts = [
                state_token,
                history_token,
                arm_tokens,
                future_force_token,
                future_flow_tokens,
                hand_tokens,
            ]
        else:
            suffix_parts = (
                [state_token, history_token, future_force_token, future_flow_tokens, action_tokens]
                if self.structured_tactile
                else [history_token, state_token, future_force_token, future_flow_tokens, action_tokens]
            )
        tokens = jnp.concatenate(suffix_parts, axis=1)
        input_mask = jnp.ones(tokens.shape[:2], dtype=jnp.bool_)
        ar_mask = self._build_suffix_ar_mask(noisy_actions.shape[1])
        return tokens, input_mask, ar_mask, adarms_cond

    def _forward_train_joint_multilayer(
        self,
        *,
        prefix_tokens: at.Float[at.Array, "b p d"],
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, " p"],
        student_suffix_tokens: at.Float[at.Array, "b a d"],
        student_suffix_mask: at.Bool[at.Array, "b a"],
        student_suffix_ar_mask: at.Bool[at.Array, " a"],
        student_adarms: at.Float[at.Array, "b d"] | None,
        teacher_suffix_tokens: at.Float[at.Array, "b f d"],
        teacher_suffix_mask: at.Bool[at.Array, "b f"],
        teacher_suffix_ar_mask: at.Bool[at.Array, " f"],
        teacher_adarms: at.Float[at.Array, "b d"] | None,
    ) -> tuple[
        at.Float[at.Array, "b a d"],
        at.Float[at.Array, "b f d"],
        tuple[at.Float[at.Array, "b a d"], ...],
        tuple[at.Float[at.Array, "b f d"], ...],
    ]:
        bsz = prefix_mask.shape[0]
        p_len = prefix_mask.shape[1]
        s_len = student_suffix_mask.shape[1]
        t_len = teacher_suffix_mask.shape[1]

        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        student_attn = make_attn_mask(student_suffix_mask, student_suffix_ar_mask)
        teacher_attn = make_attn_mask(teacher_suffix_mask, teacher_suffix_ar_mask)

        student_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=s_len)
        student_to_prefix = jnp.logical_and(student_to_prefix, student_suffix_mask[:, :, None])
        teacher_to_prefix = einops.repeat(prefix_mask, "b p -> b t p", t=t_len)
        teacher_to_prefix = jnp.logical_and(teacher_to_prefix, teacher_suffix_mask[:, :, None])

        prefix_row = jnp.concatenate(
            [
                prefix_attn,
                jnp.zeros((bsz, p_len, s_len), dtype=jnp.bool_),
                jnp.zeros((bsz, p_len, t_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        student_row = jnp.concatenate(
            [
                student_to_prefix,
                student_attn,
                jnp.zeros((bsz, s_len, t_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        teacher_row = jnp.concatenate(
            [
                teacher_to_prefix,
                jnp.zeros((bsz, t_len, s_len), dtype=jnp.bool_),
                teacher_attn,
            ],
            axis=-1,
        )
        full_attn = jnp.concatenate([prefix_row, student_row, teacher_row], axis=1)

        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
        student_positions = prefix_len + jnp.cumsum(student_suffix_mask, axis=-1) - 1
        teacher_positions = prefix_len + jnp.cumsum(teacher_suffix_mask, axis=-1) - 1
        positions = jnp.concatenate([prefix_positions, student_positions, teacher_positions], axis=1)

        (outputs, selected_layers), _ = self.PaliGemma.llm(
            self._llm_streams(prefix_tokens, student_suffix_tokens, teacher_suffix_tokens),
            mask=full_attn,
            positions=positions,
            adarms_cond=self._llm_adarms(student_adarms, teacher_adarms),
            return_layer_indices=self.distill_layer_indices,
        )
        student_out = outputs[1]
        teacher_out = outputs[2]
        student_layer_hiddens = tuple(layer[1] for layer in selected_layers)
        teacher_layer_hiddens = tuple(layer[2] for layer in selected_layers)
        return student_out, teacher_out, student_layer_hiddens, teacher_layer_hiddens

    def _forward_student_multilayer(
        self,
        *,
        prefix_tokens: at.Float[at.Array, "b p d"],
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, " p"],
        student_suffix_tokens: at.Float[at.Array, "b s d"],
        student_suffix_mask: at.Bool[at.Array, "b s"],
        student_suffix_ar_mask: at.Bool[at.Array, " s"],
        student_adarms: at.Float[at.Array, "b d"] | None,
    ) -> tuple[at.Float[at.Array, "b s d"], tuple[at.Float[at.Array, "b s d"], ...]]:
        bsz = prefix_mask.shape[0]
        p_len = prefix_mask.shape[1]
        s_len = student_suffix_mask.shape[1]

        prefix_attn = make_attn_mask(prefix_mask, prefix_ar_mask)
        student_attn = make_attn_mask(student_suffix_mask, student_suffix_ar_mask)
        student_to_prefix = einops.repeat(prefix_mask, "b p -> b s p", s=s_len)
        student_to_prefix = jnp.logical_and(student_to_prefix, student_suffix_mask[:, :, None])

        prefix_row = jnp.concatenate(
            [
                prefix_attn,
                jnp.zeros((bsz, p_len, s_len), dtype=jnp.bool_),
            ],
            axis=-1,
        )
        student_row = jnp.concatenate([student_to_prefix, student_attn], axis=-1)
        full_attn = jnp.concatenate([prefix_row, student_row], axis=1)

        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_len = jnp.sum(prefix_mask, axis=-1)[:, None]
        student_positions = prefix_len + jnp.cumsum(student_suffix_mask, axis=-1) - 1
        positions = jnp.concatenate([prefix_positions, student_positions], axis=1)

        (outputs, selected_layers), _ = self.PaliGemma.llm(
            self._llm_streams(prefix_tokens, student_suffix_tokens, None),
            mask=full_attn,
            positions=positions,
            adarms_cond=self._llm_adarms(student_adarms, None),
            return_layer_indices=self.distill_layer_indices,
        )
        student_out = outputs[1]
        student_layer_hiddens = tuple(layer[1] for layer in selected_layers)
        return student_out, student_layer_hiddens

    def _compute_chunk_loss_with_stats(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        train_progress: at.Float[at.Array, ""] | float | None = None,
        tactile_ttt_state: at.Array | None = None,
        sequence_active: at.Array | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array], at.Array | None]:
        (
            preprocess_rng,
            noise_rng,
            time_rng,
            query_noise_rng,
            async_offset_rng,
            async_flow_time_rng,
            cached_async_offset_rng,
        ) = jax.random.split(rng, 7)
        original_flow_img = observation.flow_img
        original_wrist_flow_img = observation.wrist_flow_img
        original_future_rgb_img = observation.future_rgb_img
        original_future_wrist_rgb_img = observation.future_wrist_rgb_img
        original_scene_flow = observation.scene_flow
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, effort_type=self.effort_type
        )
        observation = self._restore_aux_images(
            observation,
            original_flow_img,
            original_wrist_flow_img,
            original_future_rgb_img,
            original_future_wrist_rgb_img,
            original_scene_flow,
        )
        history_effort, future_effort = self._split_effort(
            observation,
            require_future=self.use_teacher_ae,
            dtype=actions.dtype,
        )
        if self.use_teacher_ae and future_effort is None:
            raise ValueError("Pi0MORDualAlignForceFlow teacher training requires future effort.")

        tactile_ttt_stats: dict[str, at.Array] = {}
        history_token_override = None
        if self.tactile_ttt_enabled:
            history_token_override, tactile_ttt_state, tactile_ttt_stats = self._tactile_ttt_step(
                history_effort,
                tactile_ttt_state,
                active=sequence_active,
            )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t_action = time_expanded * noise + (1 - time_expanded) * actions
        u_t_action = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        (
            student_tokens,
            student_mask,
            student_ar_mask,
            student_adarms,
            force_clean_mask,
            flow_clean_mask,
            noised_token_rate,
        ) = self.embed_student_suffix(
            observation,
            history_effort,
            x_t_action,
            time,
            train=train,
            noise_rng=query_noise_rng,
            train_progress=train_progress,
            history_token_override=history_token_override,
        )
        if self.use_teacher_ae:
            teacher_tokens, teacher_mask, teacher_ar_mask, teacher_adarms = self.embed_teacher_suffix(
                observation, history_effort, future_effort, x_t_action, time
            )
            student_out, teacher_out, student_layer_hiddens, teacher_layer_hiddens = self._forward_train_joint_multilayer(
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                prefix_ar_mask=prefix_ar_mask,
                student_suffix_tokens=student_tokens,
                student_suffix_mask=student_mask,
                student_suffix_ar_mask=student_ar_mask,
                student_adarms=student_adarms,
                teacher_suffix_tokens=teacher_tokens,
                teacher_suffix_mask=teacher_mask,
                teacher_suffix_ar_mask=teacher_ar_mask,
                teacher_adarms=teacher_adarms,
            )
        else:
            student_out, student_layer_hiddens = self._forward_student_multilayer(
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                prefix_ar_mask=prefix_ar_mask,
                student_suffix_tokens=student_tokens,
                student_suffix_mask=student_mask,
                student_suffix_ar_mask=student_ar_mask,
                student_adarms=student_adarms,
            )
            teacher_out = jnp.zeros((actions.shape[0], 0, self.teacher_width), dtype=student_out.dtype)
            teacher_layer_hiddens = ()

        student_base_v = self._decode_base_action_velocity(student_out, expert="student")
        student_v, tactile_refiner_stats = self._decode_student_action_velocity_with_stats(student_out)
        student_action_loss = jnp.mean(jnp.square(student_v - u_t_action), axis=(-2, -1))
        _, student_arm_loss, student_hand_loss = self._action_losses(student_v, u_t_action)
        if self.use_teacher_ae:
            teacher_v = self._decode_action_velocity(teacher_out, expert="teacher")
            teacher_action_loss = jnp.mean(jnp.square(teacher_v - u_t_action), axis=(-2, -1))
            _, teacher_arm_loss, teacher_hand_loss = self._action_losses(teacher_v, u_t_action)
        else:
            teacher_action_loss = jnp.zeros_like(student_action_loss)
            teacher_arm_loss = jnp.zeros_like(student_action_loss)
            teacher_hand_loss = jnp.zeros_like(student_action_loss)
        hand_slice = slice(self.hand_action_start, self.hand_action_start + self.hand_action_dim)
        if self.tactile_refiner_enabled:
            synergy_loss = jnp.mean(
                jnp.square(tactile_refiner_stats["coarse_hand"] - actions[..., hand_slice]),
                axis=(-2, -1),
            )
            delta_reg_loss = jnp.mean(jnp.square(tactile_refiner_stats["delta_hand"]), axis=(-2, -1))
        else:
            synergy_loss = jnp.zeros_like(student_action_loss)
            delta_reg_loss = jnp.zeros_like(student_action_loss)

        force_losses = []
        flow_losses = []
        if self.structured_tactile:
            suffix_slices = self._suffix_slices(self.action_horizon)
            future_force_slice = suffix_slices["future_force"]
            flow_slice = suffix_slices["future_flow"]
        else:
            future_force_start = self.history_force_token_count + self.state_token_count
            future_force_slice = slice(future_force_start, future_force_start + self.future_force_token_count)
            flow_start = future_force_start + self.future_force_token_count
            flow_slice = slice(flow_start, flow_start + self.flow_token_count)

        if self.async_tactile_refiner_enabled:
            async_offsets = jnp.asarray(self.async_refiner_offsets, dtype=jnp.int32)
            async_offset = async_offsets[
                jax.random.randint(async_offset_rng, (), minval=0, maxval=async_offsets.shape[0])
            ]
        elif self.async_tactile_flow_refiner_enabled:
            async_offsets = jnp.asarray(self.async_flow_refiner_offsets, dtype=jnp.int32)
            async_offset = async_offsets[
                jax.random.randint(async_offset_rng, (), minval=0, maxval=async_offsets.shape[0])
            ]
        else:
            async_offset = jnp.asarray(0, dtype=jnp.int32)
        base_action_estimate = x_t_action - time_expanded * student_base_v
        async_refiner_loss, async_refiner_stats = self._async_refine_hand_action(
            base_action=base_action_estimate,
            target_action=actions,
            student_contact_hidden=student_layer_hiddens[-1][:, future_force_slice, :],
            history_effort=history_effort,
            future_effort=future_effort,
            state=observation.state,
            offset=async_offset,
        )
        async_flow_time = (
            self.async_flow_refiner_tau_split
            * (jax.random.beta(async_flow_time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001)
            if self.async_tactile_flow_refiner_enabled
            else jnp.zeros(batch_shape, dtype=actions.dtype)
        )
        async_flow_time_expanded = async_flow_time[..., None, None]
        x_t_async_flow = async_flow_time_expanded * noise + (1 - async_flow_time_expanded) * actions
        async_flow_refiner_loss, async_flow_refiner_stats = self._async_refine_hand_velocity(
            noisy_actions=x_t_async_flow,
            target_velocity=u_t_action,
            student_contact_hidden=student_layer_hiddens[-1][:, future_force_slice, :],
            history_effort=history_effort,
            future_effort=future_effort,
            state=observation.state,
            offset=async_offset,
            timestep=async_flow_time,
        )

        if self.cached_vlm_async_ae_enabled:
            cached_offsets = jnp.asarray(self.cached_vlm_async_offsets, dtype=jnp.int32)
            cached_async_offset = cached_offsets[
                jax.random.randint(cached_async_offset_rng, (), minval=0, maxval=cached_offsets.shape[0])
            ]
            cached_history_effort = self._history_effort_for_offset(
                history_effort,
                future_effort,
                cached_async_offset,
            )
            cached_future_override = self._cached_async_future_query_override(
                student_layer_hiddens[-1][:, future_force_slice, :],
                cached_async_offset,
                x_t_action.dtype,
            )
            (
                cached_student_tokens,
                cached_student_mask,
                cached_student_ar_mask,
                cached_student_adarms,
                *_,
            ) = self.embed_student_suffix(
                observation,
                cached_history_effort,
                x_t_action,
                time,
                train=False,
                noise_rng=None,
                future_force_query_override=cached_future_override,
                async_offset=cached_async_offset,
            )
            cached_student_out, cached_student_layer_hiddens = self._forward_student_multilayer(
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                prefix_ar_mask=prefix_ar_mask,
                student_suffix_tokens=cached_student_tokens,
                student_suffix_mask=cached_student_mask,
                student_suffix_ar_mask=cached_student_ar_mask,
                student_adarms=cached_student_adarms,
            )
            cached_student_v = self._decode_action_velocity(cached_student_out, expert="student")
            cached_async_action_loss = self._action_loss_with_offset_mask(
                cached_student_v,
                u_t_action,
                cached_async_offset,
            )
            cached_token_mask = self._cached_async_token_mask(actions.shape[0], cached_async_offset)
            cached_force_losses = []
            if self.disable_future_tactile:
                cached_async_future_align_loss = jnp.zeros_like(student_action_loss)
            elif self.direct_future_tactile_align:
                direct_target = jax.lax.stop_gradient(self._project_future_force_student_target(future_effort))
                for cached_student_hidden in cached_student_layer_hiddens:
                    cached_force_losses.append(
                        self._cosine_distance_masked(
                            cached_student_hidden[:, future_force_slice, :],
                            direct_target,
                            cached_token_mask,
                        )
                    )
                cached_async_future_align_loss = jnp.mean(jnp.stack(cached_force_losses, axis=0), axis=0)
            else:
                for cached_student_hidden, teacher_hidden in zip(
                    cached_student_layer_hiddens, teacher_layer_hiddens, strict=True
                ):
                    cached_student_force_hidden = self._project_prompt_distill(
                        cached_student_hidden[:, future_force_slice, :]
                    )
                    teacher_force_hidden = jax.lax.stop_gradient(teacher_hidden[:, future_force_slice, :])
                    cached_force_losses.append(
                        self._cosine_distance_masked(
                            cached_student_force_hidden,
                            teacher_force_hidden,
                            cached_token_mask,
                        )
                    )
                cached_async_future_align_loss = jnp.mean(jnp.stack(cached_force_losses, axis=0), axis=0)
            prefix_segments = jnp.asarray(cached_async_offset, dtype=jnp.float32) / float(
                self.future_steps_per_segment
            )
            if self.disable_future_tactile:
                cached_prefix_consistency_loss = jnp.zeros_like(student_action_loss)
            else:
                cached_prefix_consistency_loss = self._cosine_distance_masked(
                    cached_future_override.astype(student_layer_hiddens[-1].dtype),
                    jax.lax.stop_gradient(student_layer_hiddens[-1][:, future_force_slice, :]),
                    jnp.logical_not(cached_token_mask),
                )
        else:
            cached_async_offset = jnp.asarray(0, dtype=jnp.int32)
            prefix_segments = jnp.asarray(0.0, dtype=actions.dtype)
            cached_async_action_loss = jnp.zeros_like(student_action_loss)
            cached_async_future_align_loss = jnp.zeros_like(student_action_loss)
            cached_prefix_consistency_loss = jnp.zeros_like(student_action_loss)

        if self.disable_future_tactile:
            raw_future_force_align_loss = jnp.zeros_like(student_action_loss)
        elif self.direct_future_tactile_align:
            direct_target = jax.lax.stop_gradient(self._project_future_force_student_target(future_effort))
            for student_hidden in student_layer_hiddens:
                force_losses.append(
                    self._apply_loss_mask(
                        self._cosine_distance(student_hidden[:, future_force_slice, :], direct_target),
                        force_clean_mask,
                    )
                )
            raw_future_force_align_loss = jnp.mean(jnp.stack(force_losses, axis=0), axis=0)
        else:
            for student_hidden, teacher_hidden in zip(student_layer_hiddens, teacher_layer_hiddens, strict=True):
                student_force_hidden = self._project_prompt_distill(student_hidden[:, future_force_slice, :])
                teacher_force_hidden = jax.lax.stop_gradient(teacher_hidden[:, future_force_slice, :])
                force_losses.append(
                    self._apply_loss_mask(
                        self._cosine_distance(student_force_hidden, teacher_force_hidden),
                        force_clean_mask,
                    )
                )

                if self.use_future_flow:
                    student_flow_hidden = self._project_flow_distill(student_hidden[:, flow_slice, :])
                    teacher_flow_hidden = jax.lax.stop_gradient(teacher_hidden[:, flow_slice, :])
                    flow_losses.append(
                        self._cosine_distance_masked(
                            student_flow_hidden,
                            teacher_flow_hidden,
                            flow_clean_mask,
                        )
                    )

            raw_future_force_align_loss = jnp.mean(jnp.stack(force_losses, axis=0), axis=0)
        raw_future_flow_align_loss = (
            jnp.mean(jnp.stack(flow_losses, axis=0), axis=0)
            if self.use_future_flow
            else jnp.zeros_like(raw_future_force_align_loss)
        )
        future_force_align_loss = raw_future_force_align_loss
        future_flow_align_loss = raw_future_flow_align_loss
        if self.use_teacher_ae:
            teacher_future_tokens_for_patch_aux = self._project_future_force_teacher(future_effort)
            patch_distribution_aux_loss = self._patch_distribution_aux_loss(
                teacher_future_tokens=teacher_future_tokens_for_patch_aux,
                future_effort=future_effort,
            )
        else:
            patch_distribution_aux_loss = jnp.zeros_like(student_action_loss)

        total_loss = (
            self.student_action_loss_weight * student_action_loss
            + self.teacher_action_loss_weight * teacher_action_loss
            + self.future_force_align_loss_weight * future_force_align_loss
            + self.future_flow_align_loss_weight * future_flow_align_loss
            + self.tactile_patch_aux_loss_weight * patch_distribution_aux_loss
            + self.hand_synergy_loss_weight * synergy_loss
            + self.tactile_refiner_delta_loss_weight * delta_reg_loss
            + self.async_refiner_loss_weight * async_refiner_loss
            + self.async_refiner_delta_loss_weight * async_refiner_stats["delta_reg"]
            + self.async_refiner_gate_loss_weight * async_refiner_stats["gate_reg"]
            + self.async_flow_refiner_loss_weight * async_flow_refiner_loss
            + self.cached_vlm_async_loss_weight * cached_async_action_loss
            + self.cached_vlm_async_future_align_loss_weight * cached_async_future_align_loss
            + self.cached_vlm_async_prefix_consistency_weight * cached_prefix_consistency_loss
        )
        stats = {
            "loss/student_action": student_action_loss,
            "loss/teacher_action": teacher_action_loss,
            "loss/student_action_arm": student_arm_loss,
            "loss/student_action_hand": student_hand_loss,
            "loss/teacher_action_arm": teacher_arm_loss,
            "loss/teacher_action_hand": teacher_hand_loss,
            "loss/distill_future_force": future_force_align_loss,
            "loss/distill_future_flow": future_flow_align_loss,
            "loss/distill_future_force_mean": jnp.mean(raw_future_force_align_loss),
            "loss/distill_future_flow_mean": jnp.mean(raw_future_flow_align_loss),
            "loss/patch_distribution_aux": patch_distribution_aux_loss,
            "loss/hand_synergy": synergy_loss,
            "loss/tactile_refiner_delta_reg": delta_reg_loss,
            "loss/async_refiner": async_refiner_loss,
            "loss/async_refiner_delta_reg": async_refiner_stats["delta_reg"],
            "loss/async_refiner_gate_reg": async_refiner_stats["gate_reg"],
            "async_refiner/gate_mean": async_refiner_stats["gate_mean"],
            "async_refiner/delta_abs_mean": async_refiner_stats["delta_abs_mean"],
            "async_refiner/offset": jnp.broadcast_to(
                async_refiner_stats["offset"], student_action_loss.shape
            ),
            "loss/async_flow_refiner": async_flow_refiner_loss,
            "async_flow_refiner/velocity_abs_mean": async_flow_refiner_stats["velocity_abs_mean"],
            "async_flow_refiner/offset": jnp.broadcast_to(
                async_flow_refiner_stats["offset"], student_action_loss.shape
            ),
            "async_flow_refiner/timestep": async_flow_refiner_stats["timestep"],
            "loss/cached_async_action": cached_async_action_loss,
            "loss/cached_async_future_align": cached_async_future_align_loss,
            "loss/cached_async_prefix_consistency": cached_prefix_consistency_loss,
            "cached_async/offset": jnp.broadcast_to(
                jnp.asarray(cached_async_offset, dtype=jnp.float32), student_action_loss.shape
            ),
            "cached_async/prefix_segments": jnp.broadcast_to(
                jnp.asarray(prefix_segments, dtype=jnp.float32), student_action_loss.shape
            ),
            "tactile_refiner/gate_mean": (
                jnp.mean(tactile_refiner_stats["gate"], axis=(-2, -1))
                if self.tactile_refiner_enabled
                else jnp.zeros_like(student_action_loss)
            ),
            "tactile_refiner/delta_abs_mean": (
                jnp.mean(jnp.abs(tactile_refiner_stats["delta_hand"]), axis=(-2, -1))
                if self.tactile_refiner_enabled
                else jnp.zeros_like(student_action_loss)
            ),
            "noise/student_future_query_token_rate": jnp.mean(noised_token_rate),
            "noise/student_future_query_scale": self._student_query_noise_scale(train_progress),
            "loss/total": total_loss,
        }
        if self.tactile_ttt_enabled:
            stats.update(
                {
                    "tactile_ttt/contact_gate": tactile_ttt_stats["contact_gate"],
                    "tactile_ttt/reconstruction": tactile_ttt_stats["reconstruction"],
                    "tactile_ttt/fast_weight_norm": tactile_ttt_stats["fast_weight_norm"],
                }
            )
        return total_loss, stats, tactile_ttt_state

    def compute_loss_with_stats(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        train_progress: at.Float[at.Array, ""] | float | None = None,
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        loss, stats, _ = self._compute_chunk_loss_with_stats(
            rng,
            observation,
            actions,
            train=train,
            train_progress=train_progress,
        )
        return loss, stats

    def compute_sequence_loss_with_stats(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        train_progress: at.Float[at.Array, ""] | float | None = None,
    ) -> tuple[jax.Array, dict[str, at.Array]]:
        """Train on ordered action chunks with one fast memory per episode."""
        if not self.tactile_ttt_enabled:
            raise ValueError("Sequence loss is only defined for the TactileTTT configuration.")
        if actions.ndim != 4:
            raise ValueError(f"Expected sequence actions [B,S,H,A], got {actions.shape}.")

        batch_size, sequence_length = actions.shape[:2]
        sequence_mask = observation.sequence_mask
        if sequence_mask is None:
            sequence_mask = jnp.ones((batch_size, sequence_length), dtype=jnp.float32)
        else:
            sequence_mask = jnp.asarray(sequence_mask, dtype=jnp.float32)

        fast_weight = self.initial_tactile_ttt_state(batch_size)
        losses = []
        per_step_stats: list[dict[str, at.Array]] = []
        step_rngs = jax.random.split(rng, sequence_length)
        for step_index in range(sequence_length):
            step_observation = jax.tree.map(
                lambda value, index=step_index: value[:, index],
                observation,
            )
            step_observation = step_observation.replace(sequence_mask=None)
            active = sequence_mask[:, step_index]
            step_loss, step_stats, fast_weight = self._compute_chunk_loss_with_stats(
                step_rngs[step_index],
                step_observation,
                actions[:, step_index],
                train=train,
                train_progress=train_progress,
                tactile_ttt_state=fast_weight,
                sequence_active=active,
            )
            losses.append(step_loss)
            per_step_stats.append(step_stats)

        stacked_losses = jnp.stack(losses, axis=1)
        denominator = jnp.maximum(jnp.sum(sequence_mask, axis=1), 1.0)
        sequence_loss = jnp.sum(stacked_losses * sequence_mask, axis=1) / denominator

        reduced_stats = {}
        for key in per_step_stats[0]:
            values = []
            for step_stats in per_step_stats:
                value = jnp.asarray(step_stats[key])
                if value.ndim == 0:
                    value = jnp.broadcast_to(value, (batch_size,))
                values.append(value)
            stacked = jnp.stack(values, axis=1)
            mask = sequence_mask
            while mask.ndim < stacked.ndim:
                mask = mask[..., None]
            denom = denominator
            while denom.ndim < stacked.ndim - 1:
                denom = denom[..., None]
            reduced_stats[key] = jnp.sum(stacked * mask, axis=1) / denom
        return sequence_loss, reduced_stats

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        train_progress: at.Float[at.Array, ""] | float | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train, train_progress=train_progress)
        return loss

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        debug_query_noise_scale: float | None = None,
    ) -> _model.Actions:
        if self.tactile_ttt_enabled:
            actions, _, _ = self.sample_actions_with_tactile_memory(
                rng,
                observation,
                self.initial_tactile_ttt_state(observation.state.shape[0]),
                num_steps=num_steps,
                noise=noise,
            )
            return actions
        original_flow_img = observation.flow_img
        original_wrist_flow_img = observation.wrist_flow_img
        original_future_rgb_img = observation.future_rgb_img
        original_future_wrist_rgb_img = observation.future_wrist_rgb_img
        original_scene_flow = observation.scene_flow
        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        observation = self._restore_aux_images(
            observation,
            original_flow_img,
            original_wrist_flow_img,
            original_future_rgb_img,
            original_future_wrist_rgb_img,
            original_scene_flow,
        )
        history_effort, _ = self._split_effort(observation, require_future=False, dtype=jnp.float32)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        action_noise_rng, query_noise_rng = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            self._llm_streams(prefix_tokens, None, None),
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=self._llm_adarms(None, None),
        )

        def step(carry):
            x_t, time, step_rng = carry
            step_rng, iter_query_noise_rng = jax.random.split(step_rng)
            student_tokens, student_mask, student_ar_mask, student_adarms, *_ = self.embed_student_suffix(
                observation,
                history_effort,
                x_t,
                jnp.broadcast_to(time, batch_size),
                train=False,
                noise_rng=iter_query_noise_rng,
                query_noise_scale=debug_query_noise_scale,
            )
            student_attn_mask = make_attn_mask(student_mask, student_ar_mask)
            prefix_to_student = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_student, student_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(student_mask, axis=-1) - 1

            outputs, _ = self.PaliGemma.llm(
                self._llm_streams(None, student_tokens, None),
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._llm_adarms(student_adarms, None),
            )
            student_out = outputs[1]
            v_t = self._decode_action_velocity(student_out, expert="student")
            return x_t + dt * v_t, time + dt, step_rng

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, query_noise_rng))
        return x_0

    def sample_actions_with_tactile_memory(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        tactile_ttt_state: at.Array,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, at.Array, dict[str, at.Array]]:
        """Sample one action chunk and carry the updated fast tactile memory."""
        if not self.tactile_ttt_enabled:
            raise ValueError("This model was not configured with TactileTTT.")

        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        history_effort, _ = self._split_effort(observation, require_future=False, dtype=jnp.float32)
        history_tokens, tactile_ttt_state, tactile_ttt_stats = self._tactile_ttt_step(
            history_effort,
            tactile_ttt_state,
        )

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        action_noise_rng, query_noise_rng = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            self._llm_streams(prefix_tokens, None, None),
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=self._llm_adarms(None, None),
        )

        def step(carry):
            x_t, time = carry
            student_tokens, student_mask, student_ar_mask, student_adarms, *_ = self.embed_student_suffix(
                observation,
                history_effort,
                x_t,
                jnp.broadcast_to(time, batch_size),
                train=False,
                noise_rng=query_noise_rng,
                history_token_override=history_tokens,
            )
            student_attn_mask = make_attn_mask(student_mask, student_ar_mask)
            prefix_to_student = einops.repeat(prefix_mask, "b p -> b s p", s=student_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_to_student, student_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(student_mask, axis=-1) - 1
            outputs, _ = self.PaliGemma.llm(
                self._llm_streams(None, student_tokens, None),
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=self._llm_adarms(student_adarms, None),
            )
            velocity = self._decode_action_velocity(outputs[1], expert="student")
            return x_t + dt * velocity, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        actions, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return actions, tactile_ttt_state, tactile_ttt_stats

    def _cached_vlm_async_denoise_with_prefix_cache(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        history_effort: at.Array,
        prefix_mask: at.Array,
        prefix_kv_cache: object,
        async_chunk_offset: at.Int[at.Array, ""] | int,
        prefix_future_hidden: at.Array | None,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, at.Array]:
        batch_size = observation.state.shape[0]
        action_noise_rng, _ = jax.random.split(rng)
        if noise is None:
            noise = jax.random.normal(action_noise_rng, (batch_size, self.action_horizon, self.action_dim))

        dt = -1.0 / num_steps
        offset = jnp.asarray(async_chunk_offset, dtype=jnp.int32)
        suffix_slices = self._suffix_slices(self.action_horizon)
        future_force_slice = suffix_slices["future_force"]
        initial_future_hidden = jnp.zeros(
            (batch_size, self.future_force_token_count, self.student_width),
            dtype=noise.dtype,
        )

        def step(carry):
            x_t, time, final_future_hidden = carry
            timestep = jnp.broadcast_to(time, (batch_size,))
            future_query_override = (
                self._cached_async_future_query_override(
                    prefix_future_hidden,
                    offset,
                    x_t.dtype,
                )
                if prefix_future_hidden is not None
                else None
            )
            async_out, async_layer_hiddens = self._student_suffix_with_prefix_cache_multilayer(
                observation=observation,
                history_effort=history_effort,
                prefix_mask=prefix_mask,
                prefix_kv_cache=prefix_kv_cache,
                noisy_actions=x_t,
                timestep=timestep,
                future_force_query_override=future_query_override,
                async_offset=offset,
            )
            v_t = self._decode_action_velocity(async_out, expert="student")
            final_future_hidden = async_layer_hiddens[-1][:, future_force_slice, :].astype(final_future_hidden.dtype)
            return x_t + dt * v_t, time + dt, final_future_hidden

        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2

        x_0, _, final_future_hidden = jax.lax.while_loop(cond, step, (noise, 1.0, initial_future_hidden))
        return x_0, final_future_hidden

    def _preprocess_cached_vlm_async_observation(
        self,
        observation: _model.Observation,
    ) -> _model.Observation:
        original_flow_img = observation.flow_img
        original_wrist_flow_img = observation.wrist_flow_img
        original_future_rgb_img = observation.future_rgb_img
        original_future_wrist_rgb_img = observation.future_wrist_rgb_img
        original_scene_flow = observation.scene_flow
        observation = _model.preprocess_observation(None, observation, train=False, effort_type=self.effort_type)
        return self._restore_aux_images(
            observation,
            original_flow_img,
            original_wrist_flow_img,
            original_future_rgb_img,
            original_future_wrist_rgb_img,
            original_scene_flow,
        )

    def sample_actions_cached_vlm_async_ae_slow(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, object, at.Array, at.Array]:
        if not self.cached_vlm_async_ae_enabled:
            raise ValueError(
                "Received a cached-VLM async AE inference request, but this checkpoint/config was not "
                "created with cached_vlm_async_ae_enabled=True. Start the server with "
                "pi0_xhand_tactile_structured_raw_dual_ae_cached_vlm_async_ae or run the client without "
                "--cached-vlm-async-ae."
            )
        observation = self._preprocess_cached_vlm_async_observation(observation)
        history_effort, _ = self._split_effort(observation, require_future=False, dtype=jnp.float32)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_kv_cache, _ = self._prefix_kv_cache(prefix_tokens, prefix_mask, prefix_ar_mask)
        actions, final_future_hidden = self._cached_vlm_async_denoise_with_prefix_cache(
            rng,
            observation,
            history_effort=history_effort,
            prefix_mask=prefix_mask,
            prefix_kv_cache=prefix_kv_cache,
            async_chunk_offset=0,
            prefix_future_hidden=None,
            noise=noise,
            num_steps=num_steps,
        )
        return actions, prefix_kv_cache, prefix_mask, final_future_hidden

    def sample_actions_cached_vlm_async_ae_fast(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        prefix_kv_cache: object,
        prefix_mask: at.Array,
        prefix_future_hidden: at.Array,
        async_chunk_offset: at.Int[at.Array, ""] | int = 0,
        async_fresh_effort: at.Array | None = None,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, at.Array]:
        if not self.cached_vlm_async_ae_enabled:
            raise ValueError(
                "Received a cached-VLM async AE fast request, but cached_vlm_async_ae_enabled=False."
            )
        observation = self._preprocess_cached_vlm_async_observation(observation)
        history_effort, _ = self._split_effort(observation, require_future=False, dtype=jnp.float32)
        fresh_history_effort = history_effort
        if async_fresh_effort is not None:
            fresh_history_effort = jnp.asarray(async_fresh_effort, dtype=jnp.float32)
            fresh_history_effort = self._pad_or_crop_effort(
                fresh_history_effort,
                self.force_input_frames,
                from_end=True,
            )
        if noise is None:
            action_noise_rng, _ = jax.random.split(rng)
            noise = jax.random.normal(
                action_noise_rng,
                (observation.state.shape[0], self.action_horizon, self.action_dim),
            )
        actions, final_future_hidden = self._cached_vlm_async_denoise_with_prefix_cache(
            rng,
            observation,
            history_effort=fresh_history_effort,
            prefix_mask=prefix_mask,
            prefix_kv_cache=prefix_kv_cache,
            async_chunk_offset=async_chunk_offset,
            prefix_future_hidden=prefix_future_hidden,
            noise=noise,
            num_steps=num_steps,
        )
        offset = jnp.asarray(async_chunk_offset, dtype=jnp.int32)
        step_mask = (
            jnp.arange(self.action_horizon, dtype=jnp.int32)[None, :, None] >= offset
        ).astype(actions.dtype)
        actions = noise * (1.0 - step_mask) + actions * step_mask
        return actions, final_future_hidden

    def sample_actions_cached_vlm_async_ae(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        async_mode: str = "slow_and_fast",
        async_chunk_offset: at.Int[at.Array, ""] | int = 0,
        async_fresh_effort: at.Array | None = None,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        if async_mode not in {"slow", "slow_and_fast"}:
            raise ValueError(
                "sample_actions_cached_vlm_async_ae no longer supports standalone fast requests. "
                "Use the policy server cache path."
            )
        actions, _, _, _ = self.sample_actions_cached_vlm_async_ae_slow(
            rng,
            observation,
            noise=noise,
            num_steps=num_steps,
        )
        return actions
