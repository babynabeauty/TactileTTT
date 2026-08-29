import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model_tavla as _model
from openpi.models import pi0_config
from openpi.models.tactile_tokenizer import PatchInformedFingerTokenizer
from openpi.models.tactile_tokenizer import PlainRawTactileMLPTokenizer
from openpi.models.tactile_tokenizer import RawTactileSpatialTokenizer


def _create_tactile_encoder(config, rngs: nnx.Rngs):
    encoder_type = config.pretrain_tactile_encoder
    encoder_cls = {
        "patch_informed": PatchInformedFingerTokenizer,
        "raw_spatial": RawTactileSpatialTokenizer,
        "raw_mlp": PlainRawTactileMLPTokenizer,
    }[encoder_type]
    common_kwargs = dict(
        output_dim=config.encoder_width,
        hidden_dim=config.tactile_tokenizer_dim,
        num_fingers=config.tactile_num_fingers,
        num_points=config.tactile_points_per_finger,
        dim_per_point=config.tactile_dim_per_finger,
        future_segments=config.future_tactile_segments,
        future_steps_per_segment=config.future_steps_per_segment,
        contact_top_k=config.tactile_raw_contact_top_k,
        contact_threshold=config.tactile_raw_contact_threshold,
        contact_temperature=config.tactile_raw_contact_temperature,
        rngs=rngs,
    )
    if encoder_type == "patch_informed":
        common_kwargs["num_patches"] = config.tactile_num_patches
    return encoder_cls(**common_kwargs)


def _sigmoid_bce_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    logits = logits.astype(jnp.float32)
    targets = targets.astype(jnp.float32)
    return jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _smooth_l1(prediction: jax.Array, target: jax.Array) -> jax.Array:
    error = jnp.abs(prediction.astype(jnp.float32) - target.astype(jnp.float32))
    return jnp.where(error < 1.0, 0.5 * jnp.square(error), error - 0.5)


class XHandPatchTactileEncoderPretrain(nnx.Module):
    """Stage-1 pretraining for the patch-informed XHand tactile encoder.

    This model trains only the raw tactile encoder and lightweight decoder heads.
    The encoder keeps the policy-facing token budget at one token per finger,
    while the heads force each finger token to retain local 5-patch contact
    structure.
    """

    def __init__(self, config: pi0_config.XHandPatchTactileEncoderPretrainConfig, rngs: nnx.Rngs):
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.max_token_len = int(config.max_token_len)
        self.effort_type = config.effort_type
        self.force_input_frames = int(config.force_input_frames)
        self.num_fingers = int(config.tactile_num_fingers)
        self.num_points = int(config.tactile_points_per_finger)
        self.dim_per_point = int(config.tactile_dim_per_finger)
        self.num_patches = int(config.tactile_num_patches)
        self.encoder_width = int(config.encoder_width)
        self.force_only_summary = bool(config.patch_force_only_summary)
        self.summary_dim = self.dim_per_point if self.force_only_summary else 2 * self.dim_per_point + 1
        self.distribution_loss_weight = float(config.patch_distribution_loss_weight)
        self.summary_loss_weight = float(config.patch_summary_loss_weight)
        self.contact_loss_weight = float(config.patch_contact_loss_weight)
        self.active_force_magnitude_loss_weight = float(
            config.patch_active_force_magnitude_loss_weight
        )
        self.inactive_force_loss_weight = float(config.patch_inactive_force_loss_weight)
        self.pretrain_history_time_samples = int(config.pretrain_history_time_samples)
        self.pretrain_future_time_samples = int(config.pretrain_future_time_samples)
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )

        self.patch_encoder = _create_tactile_encoder(config, rngs)
        self.patch_distribution_head = nnx.Linear(self.encoder_width, self.num_patches, rngs=rngs)
        self.patch_summary_head = nnx.Linear(
            self.encoder_width, self.num_patches * self.summary_dim, rngs=rngs
        )
        self.patch_contact_head = nnx.Linear(self.encoder_width, self.num_patches, rngs=rngs)

    def _split_effort(self, observation: _model.Observation, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
        if observation.effort is None:
            raise ValueError("XHand patch tactile encoder pretraining requires observation.effort.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected = (
            self.force_input_frames + self.action_horizon,
            self.num_fingers,
            self.num_points,
            self.dim_per_point,
        )
        if effort.ndim != 5 or effort.shape[1:] != expected:
            raise ValueError(f"Expected raw tactile effort [B,{expected}], got {effort.shape}.")
        return effort[:, : self.force_input_frames], effort[:, self.force_input_frames :]

    def _sample_time_axis(
        self,
        rng: jax.Array,
        effort: jax.Array,
        times: jax.Array,
        sample_count: int,
        *,
        train: bool,
    ) -> tuple[jax.Array, jax.Array]:
        total_steps = effort.shape[1]
        if sample_count <= 0 or sample_count >= total_steps:
            return effort, times
        if train:
            indices = jnp.sort(jax.random.permutation(rng, total_steps)[:sample_count])
        else:
            indices = jnp.linspace(0, total_steps - 1, sample_count).astype(jnp.int32)
        return jnp.take(effort, indices, axis=1), jnp.take(times, indices, axis=0)

    def _encode_all_steps(
        self,
        history_effort: jax.Array,
        future_effort: jax.Array,
        history_times: jax.Array,
        future_times: jax.Array,
    ) -> jax.Array:
        history_tokens = self.patch_encoder._encode_steps(
            history_effort,
            history_times,
            future=False,
            include_temporal=False,
        )
        future_tokens = self.patch_encoder._encode_steps(
            future_effort,
            future_times,
            future=True,
            include_temporal=False,
        )
        return jnp.concatenate([history_tokens, future_tokens], axis=1)

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False):
        action_dtype = actions.dtype
        del actions
        history_effort, future_effort = self._split_effort(observation, dtype=action_dtype)
        history_times = jnp.asarray(self.history_times, dtype=jnp.float32)
        future_times = jnp.asarray(self.future_times, dtype=jnp.float32)
        history_rng, future_rng = jax.random.split(rng)
        history_effort, history_times = self._sample_time_axis(
            history_rng,
            history_effort,
            history_times,
            self.pretrain_history_time_samples,
            train=train,
        )
        future_effort, future_times = self._sample_time_axis(
            future_rng,
            future_effort,
            future_times,
            self.pretrain_future_time_samples,
            train=train,
        )
        effort = jnp.concatenate([history_effort, future_effort], axis=1)
        tokens = self._encode_all_steps(history_effort, future_effort, history_times, future_times)

        target_dist, target_summary, target_contact = self.patch_encoder.patch_reconstruction_targets(effort)
        if self.force_only_summary:
            target_summary = target_summary[..., : self.dim_per_point]
        dist_logits = self.patch_distribution_head(tokens)
        summary_pred = self.patch_summary_head(tokens)
        summary_pred = einops.rearrange(
            summary_pred,
            "b t f (r c) -> b t f r c",
            r=self.num_patches,
            c=self.summary_dim,
        )
        contact_logits = self.patch_contact_head(tokens)

        dist_loss = -jnp.sum(
            jax.lax.stop_gradient(target_dist) * jax.nn.log_softmax(dist_logits.astype(jnp.float32), axis=-1),
            axis=-1,
        )
        dist_loss = jnp.mean(dist_loss, axis=(1, 2))
        summary_error = _smooth_l1(summary_pred, jax.lax.stop_gradient(target_summary))
        if self.force_only_summary:
            active_mask = jax.lax.stop_gradient(target_contact.astype(jnp.float32))[..., None]
            inactive_mask = 1.0 - active_mask
            inactive_zero_error = _smooth_l1(summary_pred, jnp.zeros_like(summary_pred))
            pred_force_magnitude = jnp.sqrt(
                jnp.sum(jnp.square(summary_pred.astype(jnp.float32)), axis=-1) + 1e-8
            )
            target_force_magnitude = jnp.sqrt(
                jnp.sum(
                    jnp.square(jax.lax.stop_gradient(target_summary.astype(jnp.float32))),
                    axis=-1,
                )
                + 1e-8
            )
            active_patch_mask = active_mask[..., 0]
            active_magnitude_error = _smooth_l1(
                pred_force_magnitude,
                target_force_magnitude,
            )
            active_denom = jnp.maximum(
                jnp.sum(active_mask, axis=(1, 2, 3, 4)) * self.dim_per_point,
                1.0,
            )
            active_patch_denom = jnp.maximum(
                jnp.sum(active_patch_mask, axis=(1, 2, 3)),
                1.0,
            )
            inactive_denom = jnp.maximum(
                jnp.sum(inactive_mask, axis=(1, 2, 3, 4)) * self.dim_per_point,
                1.0,
            )
            active_force_loss = (
                jnp.sum(summary_error * active_mask, axis=(1, 2, 3, 4))
                / active_denom
            )
            active_force_magnitude_loss = (
                jnp.sum(active_magnitude_error * active_patch_mask, axis=(1, 2, 3))
                / active_patch_denom
            )
            inactive_force_loss = (
                jnp.sum(inactive_zero_error * inactive_mask, axis=(1, 2, 3, 4))
                / inactive_denom
            )
            summary_loss = (
                active_force_loss
                + self.active_force_magnitude_loss_weight * active_force_magnitude_loss
                + self.inactive_force_loss_weight * inactive_force_loss
            )
            active_pred_force_magnitude = (
                jnp.sum(pred_force_magnitude * active_patch_mask, axis=(1, 2, 3))
                / active_patch_denom
            )
            active_target_force_magnitude = (
                jnp.sum(target_force_magnitude * active_patch_mask, axis=(1, 2, 3))
                / active_patch_denom
            )
        else:
            summary_loss = jnp.mean(summary_error, axis=(1, 2, 3, 4))
            active_force_loss = jnp.zeros_like(summary_loss)
            active_force_magnitude_loss = jnp.zeros_like(summary_loss)
            inactive_force_loss = jnp.zeros_like(summary_loss)
            active_pred_force_magnitude = jnp.zeros_like(summary_loss)
            active_target_force_magnitude = jnp.zeros_like(summary_loss)
        contact_loss = jnp.mean(
            _sigmoid_bce_with_logits(contact_logits, jax.lax.stop_gradient(target_contact)),
            axis=(1, 2, 3),
        )
        total = (
            self.distribution_loss_weight * dist_loss
            + self.summary_loss_weight * summary_loss
            + self.contact_loss_weight * contact_loss
        )

        pred_contact = jax.nn.sigmoid(contact_logits.astype(jnp.float32)) > 0.5
        target_contact_bool = target_contact > 0.5
        contact_accuracy = jnp.mean(pred_contact == target_contact_bool, axis=(1, 2, 3))
        return total, {
            "loss/patch_distribution": dist_loss,
            "loss/patch_summary": summary_loss,
            "loss/active_patch_force": active_force_loss,
            "loss/active_patch_force_magnitude": active_force_magnitude_loss,
            "loss/inactive_patch_force": inactive_force_loss,
            "loss/patch_contact": contact_loss,
            "loss/total": total,
            "metric/active_pred_force_magnitude": active_pred_force_magnitude,
            "metric/active_target_force_magnitude": active_target_force_magnitude,
            "metric/contact_accuracy": contact_accuracy,
            "metric/pred_contact_ratio": jnp.mean(pred_contact.astype(jnp.float32), axis=(1, 2, 3)),
            "metric/target_contact_ratio": jnp.mean(target_contact, axis=(1, 2, 3)),
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train)
        return loss

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        raise NotImplementedError("XHandPatchTactileEncoderPretrain is a stage-1 training-only model.")


class XHandPatchMeanForceEncoderPretrain(nnx.Module):
    """Stage-1 pretraining with only patch-level mean 3D force reconstruction.

    This is a cleaner variant of ``XHandPatchTactileEncoderPretrain``. It uses
    the same policy-facing patch-informed encoder, but replaces the three
    decoder heads with a single head:

      finger token -> 5 patch mean 3D forces

    The checkpoint keeps the encoder under ``patch_encoder``, so the existing
    policy weight loader can initialize student/teacher tactile encoders from
    this model in the same way as the older three-head pretraining.
    """

    def __init__(self, config: pi0_config.XHandPatchMeanForceEncoderPretrainConfig, rngs: nnx.Rngs):
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.max_token_len = int(config.max_token_len)
        self.effort_type = config.effort_type
        self.force_input_frames = int(config.force_input_frames)
        self.num_fingers = int(config.tactile_num_fingers)
        self.num_points = int(config.tactile_points_per_finger)
        self.dim_per_point = int(config.tactile_dim_per_finger)
        self.num_patches = int(config.tactile_num_patches)
        self.encoder_width = int(config.encoder_width)
        self.contact_threshold = float(config.tactile_raw_contact_threshold)
        self.contact_temperature = float(config.tactile_raw_contact_temperature)
        self.patch_mean_force_loss_weight = float(config.patch_mean_force_loss_weight)
        self.patch_mean_force_contact_loss_weight = float(config.patch_mean_force_contact_loss_weight)
        self.patch_mean_force_zero_loss_weight = float(config.patch_mean_force_zero_loss_weight)
        self.patch_mean_force_distribution_loss_weight = float(
            config.patch_mean_force_distribution_loss_weight
        )
        self.pretrain_history_time_samples = int(config.pretrain_history_time_samples)
        self.pretrain_future_time_samples = int(config.pretrain_future_time_samples)
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )

        self.patch_encoder = _create_tactile_encoder(config, rngs)
        self.patch_force_mean_head = nnx.Linear(
            self.encoder_width,
            self.num_patches * self.dim_per_point,
            rngs=rngs,
        )

    def _split_effort(self, observation: _model.Observation, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
        if observation.effort is None:
            raise ValueError("XHand patch mean-force pretraining requires observation.effort.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected = (
            self.force_input_frames + self.action_horizon,
            self.num_fingers,
            self.num_points,
            self.dim_per_point,
        )
        if effort.ndim != 5 or effort.shape[1:] != expected:
            raise ValueError(f"Expected raw tactile effort [B,{expected}], got {effort.shape}.")
        return effort[:, : self.force_input_frames], effort[:, self.force_input_frames :]

    def _sample_time_axis(
        self,
        rng: jax.Array,
        effort: jax.Array,
        times: jax.Array,
        sample_count: int,
        *,
        train: bool,
    ) -> tuple[jax.Array, jax.Array]:
        total_steps = effort.shape[1]
        if sample_count <= 0 or sample_count >= total_steps:
            return effort, times
        if train:
            indices = jnp.sort(jax.random.permutation(rng, total_steps)[:sample_count])
        else:
            indices = jnp.linspace(0, total_steps - 1, sample_count).astype(jnp.int32)
        return jnp.take(effort, indices, axis=1), jnp.take(times, indices, axis=0)

    def _encode_all_steps(
        self,
        history_effort: jax.Array,
        future_effort: jax.Array,
        history_times: jax.Array,
        future_times: jax.Array,
    ) -> jax.Array:
        history_tokens = self.patch_encoder._encode_steps(
            history_effort,
            history_times,
            future=False,
            include_temporal=False,
        )
        future_tokens = self.patch_encoder._encode_steps(
            future_effort,
            future_times,
            future=True,
            include_temporal=False,
        )
        return jnp.concatenate([history_tokens, future_tokens], axis=1)

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False):
        action_dtype = actions.dtype
        del actions
        history_effort, future_effort = self._split_effort(observation, dtype=action_dtype)
        history_times = jnp.asarray(self.history_times, dtype=jnp.float32)
        future_times = jnp.asarray(self.future_times, dtype=jnp.float32)
        history_rng, future_rng = jax.random.split(rng)
        history_effort, history_times = self._sample_time_axis(
            history_rng,
            history_effort,
            history_times,
            self.pretrain_history_time_samples,
            train=train,
        )
        future_effort, future_times = self._sample_time_axis(
            future_rng,
            future_effort,
            future_times,
            self.pretrain_future_time_samples,
            train=train,
        )
        effort = jnp.concatenate([history_effort, future_effort], axis=1)
        tokens = self._encode_all_steps(history_effort, future_effort, history_times, future_times)

        _, target_summary, target_contact = self.patch_encoder.patch_reconstruction_targets(effort)
        target_patch_force = target_summary[..., : self.dim_per_point]
        pred_patch_force = self.patch_force_mean_head(tokens)
        pred_patch_force = einops.rearrange(
            pred_patch_force,
            "b t f (r c) -> b t f r c",
            r=self.num_patches,
            c=self.dim_per_point,
        )

        patch_force_loss = jnp.mean(
            _smooth_l1(pred_patch_force, jax.lax.stop_gradient(target_patch_force)),
            axis=(1, 2, 3, 4),
        )

        pred_mag = jnp.linalg.norm(pred_patch_force.astype(jnp.float32), axis=-1)
        target_mag = jnp.linalg.norm(target_patch_force.astype(jnp.float32), axis=-1)
        contact_threshold = jnp.asarray(self.contact_threshold, dtype=jnp.float32)
        contact_temperature = jnp.asarray(max(self.contact_temperature, 1e-6), dtype=jnp.float32)
        pred_contact_logits = (pred_mag - contact_threshold) / contact_temperature
        contact_loss = jnp.mean(
            _sigmoid_bce_with_logits(pred_contact_logits, jax.lax.stop_gradient(target_contact)),
            axis=(1, 2, 3),
        )
        inactive_mask = 1.0 - jax.lax.stop_gradient(target_contact.astype(jnp.float32))
        inactive_denom = jnp.maximum(jnp.sum(inactive_mask, axis=(1, 2, 3)), 1.0)
        zero_patch_loss = jnp.sum(pred_mag * inactive_mask, axis=(1, 2, 3)) / inactive_denom

        # Supervise the relative load location without adding an independent
        # decoder head. Both distributions are derived from the same patch
        # mean-force vectors, so force and distribution predictions cannot
        # contradict each other. Frames/fingers with no active patch have no
        # meaningful distribution and are excluded from this loss.
        target_active_mag = target_mag * jax.lax.stop_gradient(target_contact.astype(jnp.float32))
        target_mag_sum = jnp.sum(target_active_mag, axis=-1, keepdims=True)
        target_distribution = target_active_mag / jnp.maximum(target_mag_sum, 1e-6)
        pred_distribution = pred_mag / jnp.maximum(jnp.sum(pred_mag, axis=-1, keepdims=True), 1e-6)
        distribution_cross_entropy = -jnp.sum(
            jax.lax.stop_gradient(target_distribution)
            * jnp.log(jnp.maximum(pred_distribution, 1e-6)),
            axis=-1,
        )
        has_distribution = (target_mag_sum[..., 0] > 1e-6).astype(jnp.float32)
        distribution_denom = jnp.maximum(jnp.sum(has_distribution, axis=(1, 2)), 1.0)
        distribution_loss = (
            jnp.sum(distribution_cross_entropy * has_distribution, axis=(1, 2))
            / distribution_denom
        )
        total = (
            self.patch_mean_force_loss_weight * patch_force_loss
            + self.patch_mean_force_contact_loss_weight * contact_loss
            + self.patch_mean_force_zero_loss_weight * zero_patch_loss
            + self.patch_mean_force_distribution_loss_weight * distribution_loss
        )

        pred_contact = pred_mag > contact_threshold
        target_contact_bool = target_contact > 0.5
        contact_accuracy = jnp.mean(pred_contact == target_contact_bool, axis=(1, 2, 3))
        return total, {
            "loss/patch_mean_force": patch_force_loss,
            "loss/contact_from_force": contact_loss,
            "loss/zero_patch_suppression": zero_patch_loss,
            "loss/distribution_from_force": distribution_loss,
            "loss/total": total,
            "metric/contact_accuracy_from_force": contact_accuracy,
            "metric/pred_contact_ratio_from_force": jnp.mean(pred_contact.astype(jnp.float32), axis=(1, 2, 3)),
            "metric/target_contact_ratio": jnp.mean(target_contact, axis=(1, 2, 3)),
            "metric/pred_patch_force_mag_mean": jnp.mean(pred_mag, axis=(1, 2, 3)),
            "metric/target_patch_force_mag_mean": jnp.mean(target_mag, axis=(1, 2, 3)),
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train)
        return loss

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        raise NotImplementedError("XHandPatchMeanForceEncoderPretrain is a stage-1 training-only model.")


class XHandPatchStrengthEncoderPretrain(nnx.Module):
    """Stage-1 pretraining with patch-level contact strength reconstruction.

    This objective asks each policy-facing finger token to preserve where local
    contact happens and how strong it is, without forcing the token to preserve
    the full 3D force direction.
    """

    def __init__(self, config: pi0_config.XHandPatchStrengthEncoderPretrainConfig, rngs: nnx.Rngs):
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.max_token_len = int(config.max_token_len)
        self.effort_type = config.effort_type
        self.force_input_frames = int(config.force_input_frames)
        self.num_fingers = int(config.tactile_num_fingers)
        self.num_points = int(config.tactile_points_per_finger)
        self.dim_per_point = int(config.tactile_dim_per_finger)
        self.num_patches = int(config.tactile_num_patches)
        self.encoder_width = int(config.encoder_width)
        self.contact_threshold = float(config.tactile_raw_contact_threshold)
        self.contact_temperature = float(config.tactile_raw_contact_temperature)
        self.patch_strength_loss_weight = float(config.patch_strength_loss_weight)
        self.patch_strength_contact_loss_weight = float(config.patch_strength_contact_loss_weight)
        self.patch_strength_zero_loss_weight = float(config.patch_strength_zero_loss_weight)
        self.pretrain_history_time_samples = int(config.pretrain_history_time_samples)
        self.pretrain_future_time_samples = int(config.pretrain_future_time_samples)
        self.history_times = tuple(
            float(offset) / float(config.tactile_sample_hz) for offset in config.tactile_history_offsets
        )
        self.future_times = tuple(
            float(step) / float(config.tactile_sample_hz) for step in range(1, config.action_horizon + 1)
        )

        self.patch_encoder = _create_tactile_encoder(config, rngs)
        self.patch_strength_head = nnx.Linear(self.encoder_width, self.num_patches, rngs=rngs)

    def _split_effort(self, observation: _model.Observation, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
        if observation.effort is None:
            raise ValueError("XHand patch strength pretraining requires observation.effort.")
        effort = jnp.asarray(observation.effort, dtype=dtype)
        expected = (
            self.force_input_frames + self.action_horizon,
            self.num_fingers,
            self.num_points,
            self.dim_per_point,
        )
        if effort.ndim != 5 or effort.shape[1:] != expected:
            raise ValueError(f"Expected raw tactile effort [B,{expected}], got {effort.shape}.")
        return effort[:, : self.force_input_frames], effort[:, self.force_input_frames :]

    def _sample_time_axis(
        self,
        rng: jax.Array,
        effort: jax.Array,
        times: jax.Array,
        sample_count: int,
        *,
        train: bool,
    ) -> tuple[jax.Array, jax.Array]:
        total_steps = effort.shape[1]
        if sample_count <= 0 or sample_count >= total_steps:
            return effort, times
        if train:
            indices = jnp.sort(jax.random.permutation(rng, total_steps)[:sample_count])
        else:
            indices = jnp.linspace(0, total_steps - 1, sample_count).astype(jnp.int32)
        return jnp.take(effort, indices, axis=1), jnp.take(times, indices, axis=0)

    def _encode_all_steps(
        self,
        history_effort: jax.Array,
        future_effort: jax.Array,
        history_times: jax.Array,
        future_times: jax.Array,
    ) -> jax.Array:
        history_tokens = self.patch_encoder._encode_steps(
            history_effort,
            history_times,
            future=False,
            include_temporal=False,
        )
        future_tokens = self.patch_encoder._encode_steps(
            future_effort,
            future_times,
            future=True,
            include_temporal=False,
        )
        return jnp.concatenate([history_tokens, future_tokens], axis=1)

    def compute_loss_with_stats(self, rng, observation, actions, *, train=False):
        action_dtype = actions.dtype
        del actions
        history_effort, future_effort = self._split_effort(observation, dtype=action_dtype)
        history_times = jnp.asarray(self.history_times, dtype=jnp.float32)
        future_times = jnp.asarray(self.future_times, dtype=jnp.float32)
        history_rng, future_rng = jax.random.split(rng)
        history_effort, history_times = self._sample_time_axis(
            history_rng,
            history_effort,
            history_times,
            self.pretrain_history_time_samples,
            train=train,
        )
        future_effort, future_times = self._sample_time_axis(
            future_rng,
            future_effort,
            future_times,
            self.pretrain_future_time_samples,
            train=train,
        )
        effort = jnp.concatenate([history_effort, future_effort], axis=1)
        tokens = self._encode_all_steps(history_effort, future_effort, history_times, future_times)

        _, target_summary, target_contact = self.patch_encoder.patch_reconstruction_targets(effort)
        target_strength = target_summary[..., -1]
        pred_strength = jax.nn.softplus(self.patch_strength_head(tokens).astype(jnp.float32))

        strength_loss = jnp.mean(
            _smooth_l1(pred_strength, jax.lax.stop_gradient(target_strength)),
            axis=(1, 2, 3),
        )

        contact_temperature = jnp.asarray(max(self.contact_temperature, 1e-6), dtype=jnp.float32)
        contact_threshold = jnp.asarray(self.contact_threshold, dtype=jnp.float32)
        pred_contact_logits = (pred_strength - contact_threshold) / contact_temperature
        contact_loss = jnp.mean(
            _sigmoid_bce_with_logits(pred_contact_logits, jax.lax.stop_gradient(target_contact)),
            axis=(1, 2, 3),
        )
        inactive_mask = 1.0 - jax.lax.stop_gradient(target_contact.astype(jnp.float32))
        inactive_denom = jnp.maximum(jnp.sum(inactive_mask, axis=(1, 2, 3)), 1.0)
        zero_patch_loss = jnp.sum(pred_strength * inactive_mask, axis=(1, 2, 3)) / inactive_denom
        total = (
            self.patch_strength_loss_weight * strength_loss
            + self.patch_strength_contact_loss_weight * contact_loss
            + self.patch_strength_zero_loss_weight * zero_patch_loss
        )

        pred_contact = pred_strength > contact_threshold
        target_contact_bool = target_contact > 0.5
        contact_accuracy = jnp.mean(pred_contact == target_contact_bool, axis=(1, 2, 3))
        return total, {
            "loss/patch_strength": strength_loss,
            "loss/contact_from_strength": contact_loss,
            "loss/zero_patch_suppression": zero_patch_loss,
            "loss/total": total,
            "metric/contact_accuracy_from_strength": contact_accuracy,
            "metric/pred_contact_ratio_from_strength": jnp.mean(pred_contact.astype(jnp.float32), axis=(1, 2, 3)),
            "metric/target_contact_ratio": jnp.mean(target_contact, axis=(1, 2, 3)),
            "metric/pred_patch_strength_mean": jnp.mean(pred_strength, axis=(1, 2, 3)),
            "metric/target_patch_strength_mean": jnp.mean(target_strength, axis=(1, 2, 3)),
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        loss, _ = self.compute_loss_with_stats(rng, observation, actions, train=train)
        return loss

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        raise NotImplementedError("XHandPatchStrengthEncoderPretrain is a stage-1 training-only model.")
