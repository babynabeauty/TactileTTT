import math

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.tactile_tokenizer import _QFormerBlock
from openpi.models.tactile_tokenizer import FingerRoleFutureTactileQFormer


def _flow_time_embedding(timestep: jax.Array, width: int) -> jax.Array:
    """Continuous sinusoidal embedding for the flow-matching noise level."""
    half = width // 2
    fraction = jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    phase = timestep.astype(jnp.float32)[:, None] * (2.0 * math.pi) / period[None, :]
    embedding = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)
    if embedding.shape[-1] < width:
        embedding = jnp.pad(embedding, ((0, 0), (0, width - embedding.shape[-1])))
    return embedding


class FlowMatchingHandActionDecoder(nnx.Module):
    """FLARE-style DiT decoder used only to make c_future action-aware."""

    def __init__(
        self,
        *,
        action_horizon: int,
        hand_action_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.action_horizon = int(action_horizon)
        self.hand_action_dim = int(hand_action_dim)
        self.action_in = nnx.Linear(self.hand_action_dim, width, rngs=rngs)
        self.time_in = nnx.Linear(width, width, rngs=rngs)
        self.time_out = nnx.Linear(width, width, rngs=rngs)
        self.action_position = nnx.Param(
            0.02 * jax.random.normal(rngs.params(), (self.action_horizon, width), dtype=jnp.float32)
        )
        self.blocks = [
            _QFormerBlock(width=width, num_heads=num_heads, mlp_dim=width * 4, rngs=rngs)
            for _ in range(depth)
        ]
        self.output_norm = nnx.LayerNorm(num_features=width, rngs=rngs)
        self.action_out = nnx.Linear(width, self.hand_action_dim, rngs=rngs)

    def __call__(
        self,
        noisy_hand_action: jax.Array,
        timestep: jax.Array,
        c_future: jax.Array,
        condition: jax.Array,
    ) -> jax.Array:
        time_feature = self.time_out(nnx.swish(self.time_in(_flow_time_embedding(timestep, condition.shape[-1]))))
        action_tokens = self.action_in(noisy_hand_action)
        action_tokens = action_tokens + self.action_position.value[None] + time_feature[:, None, :]
        state_history_token = condition[:, None, :] + time_feature[:, None, :]
        tokens = jnp.concatenate([state_history_token, action_tokens], axis=1)
        for block in self.blocks:
            tokens = block(tokens, c_future)
        return self.action_out(self.output_norm(tokens[:, 1:]))


class FutureTactileEncoderPretrain(nnx.Module):
    """Stage-1 pretraining for an action-aware future tactile encoder.

    This model intentionally avoids the VLM/action expert. It teaches a small
    future tactile Q-Former to extract c_future tokens that are useful for
    predicting future hand actions.
    """

    def __init__(self, config: pi0_config.FutureTactileEncoderPretrainConfig, rngs: nnx.Rngs):
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.max_token_len = int(config.max_token_len)
        self.effort_type = config.effort_type
        self.force_input_frames = int(config.force_input_frames)
        self.num_fingers = int(config.tactile_num_fingers)
        self.dim_per_finger = int(config.tactile_dim_per_finger)
        self.future_segments = int(config.future_tactile_segments)
        self.steps_per_segment = int(config.future_steps_per_segment)
        self.hand_action_start = int(config.hand_action_start)
        self.hand_action_dim = int(config.hand_action_dim)
        self.delta_loss_weight = float(config.hand_delta_loss_weight)
        self.encoder_width = int(config.encoder_width)
        self.hand_head_type = config.hand_head_type
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )

        self.future_encoder = FingerRoleFutureTactileQFormer(
            output_dim=self.encoder_width,
            hidden_dim=config.tactile_tokenizer_dim,
            num_fingers=self.num_fingers,
            dim_per_finger=self.dim_per_finger,
            future_segments=self.future_segments,
            future_steps_per_segment=self.steps_per_segment,
            num_layers=config.encoder_depth,
            num_heads=config.encoder_num_heads,
            rngs=rngs,
            fusion_layers=config.encoder_fusion_depth,
        )
        # Model transforms pad the physical 18-D robot state to action_dim,
        # matching the state interface used by Pi0 and the stage-2 policy.
        self.state_proj = nnx.Linear(config.action_dim, self.encoder_width, rngs=rngs)
        self.history_force_proj_in = nnx.Linear(self.dim_per_finger, config.tactile_tokenizer_dim, rngs=rngs)
        self.history_force_proj_out = nnx.Linear(config.tactile_tokenizer_dim, self.encoder_width, rngs=rngs)
        self.condition_norm = nnx.LayerNorm(num_features=self.encoder_width, rngs=rngs)
        if self.hand_head_type == "mean_repeat":
            self.hand_head_in = nnx.Linear(self.encoder_width, self.encoder_width, rngs=rngs)
            self.hand_head_out = nnx.Linear(self.encoder_width, self.hand_action_dim, rngs=rngs)
        elif self.hand_head_type == "finger_flatten_4step":
            self.hand_head_finger_fusion = nnx.Linear(
                self.num_fingers * self.encoder_width,
                self.encoder_width,
                rngs=rngs,
            )
            self.hand_head_norm = nnx.LayerNorm(num_features=self.encoder_width, rngs=rngs)
            self.hand_head_out = nnx.Linear(
                self.encoder_width,
                self.steps_per_segment * self.hand_action_dim,
                rngs=rngs,
            )
        elif self.hand_head_type == "flow_dit":
            self.hand_action_decoder = FlowMatchingHandActionDecoder(
                action_horizon=self.action_horizon,
                hand_action_dim=self.hand_action_dim,
                width=self.encoder_width,
                depth=config.action_decoder_depth,
                num_heads=config.action_decoder_num_heads,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unsupported hand_head_type={self.hand_head_type!r}.")

    def _split_effort(self, observation: _model.Observation, dtype):
        if observation.effort is None:
            raise ValueError("Future tactile encoder pretraining requires observation.effort.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected = (self.force_input_frames + self.action_horizon, self.num_fingers, self.dim_per_finger)
        if effort.ndim != 4 or effort.shape[1:] != expected:
            raise ValueError(f"Expected structured effort [B,{expected}], got {effort.shape}.")
        return effort[:, : self.force_input_frames], effort[:, self.force_input_frames :]

    def _history_context(self, history_effort: jax.Array, state: jax.Array) -> jax.Array:
        history = nnx.swish(self.history_force_proj_in(history_effort))
        history = self.history_force_proj_out(history)
        history = jnp.mean(history, axis=(1, 2))
        state_context = self.state_proj(state)
        return self.condition_norm(history + state_context)

    def encode_future(self, future_effort: jax.Array) -> jax.Array:
        return self.future_encoder.encode_future(
            future_effort, jnp.asarray(self.future_times, dtype=jnp.float32)
        )

    def predict_hand_action(self, c_future: jax.Array, condition: jax.Array) -> jax.Array:
        if self.hand_head_type == "flow_dit":
            raise ValueError("flow_dit predicts a velocity field and requires noisy actions and a timestep.")
        segment_tokens = einops.rearrange(
            c_future,
            "b (s f) d -> b s f d",
            s=self.future_segments,
            f=self.num_fingers,
        )
        if self.hand_head_type == "mean_repeat":
            segment_tokens = jnp.mean(segment_tokens, axis=2)
            segment_tokens = segment_tokens + condition[:, None, :]
            step_tokens = jnp.repeat(segment_tokens, repeats=self.steps_per_segment, axis=1)
            hidden = nnx.swish(self.hand_head_in(step_tokens))
            return self.hand_head_out(hidden)

        finger_features = einops.rearrange(segment_tokens, "b s f d -> b s (f d)")
        hidden = nnx.swish(self.hand_head_finger_fusion(finger_features))
        hidden = self.hand_head_norm(hidden + condition[:, None, :])
        segment_actions = self.hand_head_out(hidden)
        return einops.rearrange(
            segment_actions,
            "b s (p a) -> b (s p) a",
            p=self.steps_per_segment,
            a=self.hand_action_dim,
        )

    @staticmethod
    def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
        error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
        loss = jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)
        return jnp.mean(loss, axis=tuple(range(1, loss.ndim)))

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False):
        del train
        history_effort, future_effort = self._split_effort(observation, dtype=actions.dtype)
        c_future = self.encode_future(future_effort)
        condition = self._history_context(history_effort, observation.state)
        target_hand_action = actions[
            :, :, self.hand_action_start : self.hand_action_start + self.hand_action_dim
        ]

        if self.hand_head_type == "flow_dit":
            noise_rng, time_rng = jax.random.split(rng)
            noise = jax.random.normal(noise_rng, target_hand_action.shape, dtype=target_hand_action.dtype)
            time = jax.random.beta(time_rng, 1.5, 1, (target_hand_action.shape[0],)) * 0.999 + 0.001
            noisy_hand_action = time[:, None, None] * noise + (1.0 - time[:, None, None]) * target_hand_action
            target_velocity = noise - target_hand_action
            predicted_velocity = self.hand_action_decoder(
                noisy_hand_action, time, c_future, condition
            )
            flow_loss = jnp.mean(jnp.square(predicted_velocity - target_velocity), axis=(1, 2))
            return flow_loss, {
                "loss/flow_matching": flow_loss,
                "loss/hand_action": flow_loss,
                "loss/hand_delta": jnp.zeros_like(flow_loss),
                "loss/total": flow_loss,
            }

        pred_hand_action = self.predict_hand_action(c_future, condition)
        hand_loss = self._smooth_l1(pred_hand_action, target_hand_action)
        delta_loss = self._smooth_l1(
            pred_hand_action[:, 1:] - pred_hand_action[:, :-1],
            target_hand_action[:, 1:] - target_hand_action[:, :-1],
        )
        total = hand_loss + self.delta_loss_weight * delta_loss
        return total, {
            "loss/hand_action": hand_loss,
            "loss/hand_delta": delta_loss,
            "loss/total": total,
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train)
        return loss

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        raise NotImplementedError("FutureTactileEncoderPretrain is a stage-1 training-only model.")
