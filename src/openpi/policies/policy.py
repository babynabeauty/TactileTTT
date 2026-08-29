from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import model_tavla as _model_tavla
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

DEBUG_INFER_PRINT_LIMIT = 3


def _summarize_debug_value(value: Any) -> str:
    if isinstance(value, str):
        return f"str={value!r}"
    if isinstance(value, bytes):
        return f"bytes len={len(value)}"
    if not hasattr(value, "shape"):
        return f"{type(value).__name__}={value}"

    array = np.asarray(value)
    summary = f"shape={array.shape}, dtype={array.dtype}"
    if array.size == 0:
        return summary + ", empty"
    if np.issubdtype(array.dtype, np.number):
        finite = array[np.isfinite(array)]
        if finite.size:
            summary += (
                f", min={float(finite.min()):.6g}, max={float(finite.max()):.6g}, "
                f"mean={float(finite.mean()):.6g}"
            )
        else:
            summary += ", no finite values"
    return summary


def _debug_print_tree(title: str, tree: dict, *, infer_index: int) -> None:
    print(f"=== {title} #{infer_index} ===", flush=True)
    flat = flax.traverse_util.flatten_dict(tree, sep="/")
    for key in sorted(flat):
        print(f"[server] {key}: {_summarize_debug_value(flat[key])}", flush=True)

    effort = flat.get("effort")
    if effort is not None:
        effort_array = np.asarray(effort)
        print(f"[server] effort detailed shape={effort_array.shape}", flush=True)
        if effort_array.ndim >= 3 and effort_array.shape[-2:] == (5, 3):
            current = effort_array[-1] if effort_array.ndim == 3 else effort_array.reshape(-1, 5, 3)[-1]
            print(f"[server] latest calc_force [5,3]:\n{np.array2string(current, precision=4)}", flush=True)
        elif effort_array.ndim >= 4 and effort_array.shape[-3:] == (5, 120, 3):
            latest = effort_array[-1] if effort_array.ndim == 4 else effort_array.reshape(-1, 5, 120, 3)[-1]
            magnitude = np.linalg.norm(latest, axis=-1)
            print(
                "[server] latest raw tactile per-finger "
                f"mag_max={np.array2string(magnitude.max(axis=-1), precision=4)}, "
                f"mag_mean={np.array2string(magnitude.mean(axis=-1), precision=4)}",
                flush=True,
            )


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._debug_infer_count = 0
        self._async_input_cache: dict[int, dict[str, Any]] = {}
        self._async_noise_cache: dict[int, jax.Array] = {}
        self._async_prefix_kv_cache: dict[int, Any] = {}
        self._async_prefix_mask_cache: dict[int, jax.Array] = {}
        self._async_future_hidden_cache: dict[int, jax.Array] = {}

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._sample_actions_cached_async_slow = (
                nnx_utils.module_jit(model.sample_actions_cached_vlm_async_ae_slow)
                if hasattr(model, "sample_actions_cached_vlm_async_ae_slow")
                else None
            )
            self._sample_actions_cached_async_fast = (
                nnx_utils.module_jit(model.sample_actions_cached_vlm_async_ae_fast)
                if hasattr(model, "sample_actions_cached_vlm_async_ae_fast")
                else None
            )
            self._sample_actions_tactile_ttt = (
                nnx_utils.module_jit(model.sample_actions_with_tactile_memory)
                if bool(getattr(model, "tactile_ttt_enabled", False))
                else None
            )
            self._tactile_ttt_state = (
                model.initial_tactile_ttt_state(1)
                if self._sample_actions_tactile_ttt is not None
                else None
            )
            self._rng = rng or jax.random.key(0)

    def reset(self) -> None:
        """Reset episode-scoped caches, including TactileTTT fast weights."""
        self._async_input_cache.clear()
        self._async_noise_cache.clear()
        self._async_prefix_kv_cache.clear()
        self._async_prefix_mask_cache.clear()
        self._async_future_hidden_cache.clear()
        if not self._is_pytorch_model and getattr(self, "_sample_actions_tactile_ttt", None) is not None:
            self._tactile_ttt_state = self._model.initial_tactile_ttt_state(1)

    def _async_chunk_id_int(self, async_chunk_id: Any) -> int:
        try:
            return int(np.asarray(async_chunk_id).item())
        except Exception:
            return -1

    def _cache_async_inputs(self, chunk_id: int, inputs: dict[str, Any]) -> None:
        if chunk_id < 0:
            return
        # Keep the chunk-start context. Fast requests reuse this cached VLM/state
        # context and pass fresh tactile separately.
        self._async_input_cache[chunk_id] = jax.tree.map(lambda x: np.asarray(x).copy(), inputs)
        for old_chunk_id in list(self._async_input_cache):
            if old_chunk_id < chunk_id - 2:
                self._async_input_cache.pop(old_chunk_id, None)
                self._async_noise_cache.pop(old_chunk_id, None)
                self._async_prefix_kv_cache.pop(old_chunk_id, None)
                self._async_prefix_mask_cache.pop(old_chunk_id, None)
                self._async_future_hidden_cache.pop(old_chunk_id, None)

    def _cached_async_inputs(self, chunk_id: int, fallback: dict[str, Any]) -> dict[str, Any]:
        if chunk_id < 0 or chunk_id not in self._async_input_cache:
            return fallback
        return jax.tree.map(lambda x: np.asarray(x).copy(), self._async_input_cache[chunk_id])

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        debug_index = self._debug_infer_count
        reset_tactile_memory = bool(
            np.asarray(obs.get("reset_tactile_memory", obs.get("episode_start", False))).item()
        )
        if reset_tactile_memory:
            self.reset()
        async_mode = obs.get("async_mode", obs.get("mode"))
        if isinstance(async_mode, np.ndarray):
            async_mode = str(async_mode.item())
        elif async_mode is not None:
            async_mode = str(async_mode)
        async_chunk_offset = obs.get("async_chunk_offset", 0)
        async_chunk_id = obs.get("async_chunk_id", -1)
        async_requested = async_mode in {"slow", "fast", "slow_and_fast"}
        fast_minimal = async_mode == "fast" and bool(np.asarray(obs.get("fast_minimal", False)).item())
        inputs = jax.tree.map(lambda x: x, obs)
        inputs.pop("reset_tactile_memory", None)
        inputs.pop("episode_start", None)
        if debug_index < DEBUG_INFER_PRINT_LIMIT:
            _debug_print_tree("SERVER RAW OBS FROM CLIENT", inputs, infer_index=debug_index)
        if fast_minimal:
            inputs = {
                "state": np.asarray(inputs["state"], dtype=np.float32),
                "effort": np.asarray(inputs["effort"], dtype=np.float32),
            }
        else:
            inputs = self._input_transform(inputs)
        if debug_index < DEBUG_INFER_PRINT_LIMIT:
            _debug_print_tree("SERVER MODEL INPUT AFTER TRANSFORM", inputs, infer_index=debug_index)
        self._debug_infer_count += 1
        async_chunk_id_int = self._async_chunk_id_int(async_chunk_id)
        model_inputs = inputs
        output_inputs = inputs
        async_fresh_inputs = None
        if async_requested and not self._is_pytorch_model:
            if async_mode in {"slow", "slow_and_fast"}:
                self._cache_async_inputs(async_chunk_id_int, inputs)
            elif async_mode == "fast":
                async_fresh_inputs = inputs
                model_inputs = self._cached_async_inputs(async_chunk_id_int, inputs)
                # Output transforms should still see the latest robot state.
                output_inputs = async_fresh_inputs
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], model_inputs)
            output_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], output_inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], model_inputs)
            output_inputs = inputs
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        sample_fn = self._sample_actions
        cached_async_kind = None
        if async_requested:
            if self._is_pytorch_model:
                raise RuntimeError("Cached async AE websocket mode is only wired for JAX policies in this server.")
            if self._sample_actions_cached_async_slow is None or self._sample_actions_cached_async_fast is None:
                # Old configs should not receive fast requests. For slow_and_fast,
                # fall back to the normal one-shot sampler so legacy deployment
                # remains usable if a client accidentally includes the mode field.
                if async_mode == "fast":
                    raise RuntimeError(
                        "Received mode='fast', but this model/server does not expose "
                        "the cached async slow/fast sampling path."
                    )
            else:
                model_action_horizon = int(getattr(self._model, "action_horizon"))
                model_action_dim = int(getattr(self._model, "action_dim"))
                if async_mode in {"slow", "slow_and_fast"}:
                    cached_async_kind = "slow"
                    action_noise = jax.random.normal(
                        sample_rng_or_pytorch_device,
                        (1, model_action_horizon, model_action_dim),
                    )
                    self._async_noise_cache[async_chunk_id_int] = action_noise
                    sample_kwargs["noise"] = action_noise
                elif async_mode == "fast":
                    cached_async_kind = "fast"
                    if (
                        async_chunk_id_int not in self._async_noise_cache
                        or async_chunk_id_int not in self._async_prefix_kv_cache
                        or async_chunk_id_int not in self._async_prefix_mask_cache
                        or async_chunk_id_int not in self._async_future_hidden_cache
                    ):
                        raise RuntimeError(
                            f"Received async fast request for chunk_id={async_chunk_id_int}, but the slow cache "
                            "is missing. Make sure a slow_and_fast request starts each chunk."
                        )
                    sample_kwargs["noise"] = self._async_noise_cache[async_chunk_id_int]
                    sample_kwargs["prefix_kv_cache"] = self._async_prefix_kv_cache[async_chunk_id_int]
                    sample_kwargs["prefix_mask"] = self._async_prefix_mask_cache[async_chunk_id_int]
                    sample_kwargs["prefix_future_hidden"] = self._async_future_hidden_cache[async_chunk_id_int]
                    sample_kwargs["async_chunk_offset"] = jnp.asarray(async_chunk_offset)
                    if async_fresh_inputs is not None and "effort" in async_fresh_inputs:
                        sample_kwargs["async_fresh_effort"] = output_inputs["effort"]
        # sample_kwargs.update(debug_query_noise_scale=0.3)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if "effort" in inputs:
            observation = _model_tavla.Observation.from_dict(inputs)
        else:
            observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if cached_async_kind == "slow":
            actions, prefix_kv_cache, prefix_mask, future_hidden = self._sample_actions_cached_async_slow(
                sample_rng_or_pytorch_device,
                observation,
                **sample_kwargs,
            )
            self._async_prefix_kv_cache[async_chunk_id_int] = prefix_kv_cache
            self._async_prefix_mask_cache[async_chunk_id_int] = prefix_mask
            self._async_future_hidden_cache[async_chunk_id_int] = future_hidden
        elif cached_async_kind == "fast":
            actions, future_hidden = self._sample_actions_cached_async_fast(
                sample_rng_or_pytorch_device,
                observation,
                **sample_kwargs,
            )
            self._async_future_hidden_cache[async_chunk_id_int] = future_hidden
        elif not self._is_pytorch_model and self._sample_actions_tactile_ttt is not None:
            actions, self._tactile_ttt_state, _ = self._sample_actions_tactile_ttt(
                sample_rng_or_pytorch_device,
                observation,
                self._tactile_ttt_state,
                **sample_kwargs,
            )
        else:
            actions = sample_fn(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": output_inputs["state"],
            "actions": actions,
        }
        if "effort" in inputs:
            outputs["effort"] = output_inputs["effort"]
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if async_requested:
            outputs["async_mode"] = async_mode
            outputs["async_chunk_offset"] = np.asarray(async_chunk_offset)
            outputs["async_chunk_id"] = np.asarray(async_chunk_id)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
