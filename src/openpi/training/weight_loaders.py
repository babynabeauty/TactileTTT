import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


def _flatten_params(params: at.Params) -> dict[tuple[object, ...], object]:
    """Flatten parameters while preserving integer module-list indices."""
    return flax.traverse_util.flatten_dict(params)


def _unflatten_params(params: dict[tuple[object, ...], object]) -> at.Params:
    return flax.traverse_util.unflatten_dict(params)


def _path_string(path: tuple[object, ...]) -> str:
    return "/".join(map(str, path))


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded_params = _augment_with_moe_shared_ffn_weights(loaded_params, params)
        loaded_params = _augment_with_mor_action_expert_weights(loaded_params, params)
        # Keep newly introduced adapters/tokenizers at their random initialization.
        return _merge_params(
            loaded_params,
            params,
            missing_regex=r".*(lora|force_tokenizer|future_query|future_force_decoder|student_query_).*",
        )


@dataclasses.dataclass(frozen=True)
class Pi0WithFutureTactileEncoderWeightLoader(WeightLoader):
    """Loads pi0 base weights plus an optional stage-1 future tactile encoder.

    The stage-1 pretrain checkpoint stores the Q-Former under `future_encoder`.
    The action-aware policy stores the same module under `target_force_tokenizer`.
    """

    pi0_params_path: str = "checkpoints/pi0_base/params"
    encoder_params_path: str | None = None

    def load(self, params: at.Params) -> at.Params:
        base_params = _model.restore_params(download.maybe_download(self.pi0_params_path), restore_type=np.ndarray)
        base_params = _augment_with_moe_shared_ffn_weights(base_params, params)
        base_params = _augment_with_mor_action_expert_weights(base_params, params)
        merged = _merge_params(
            base_params,
            params,
            missing_regex=r".*",
        )

        if self.encoder_params_path is None:
            return merged

        encoder_checkpoint = _model.restore_params(
            download.maybe_download(self.encoder_params_path), restore_type=np.ndarray
        )
        flat_ref = _flatten_params(params)
        flat_merged = _flatten_params(merged)
        flat_encoder = _flatten_params(encoder_checkpoint)

        copied = 0
        for key, value in flat_encoder.items():
            if not key or key[0] != "future_encoder":
                continue
            mapped_key = ("target_force_tokenizer", *key[1:])
            if mapped_key not in flat_ref:
                continue
            if hasattr(value, "shape") and hasattr(flat_ref[mapped_key], "shape") and value.shape != flat_ref[mapped_key].shape:
                logger.warning(
                    "Skipping future tactile encoder param %s -> %s due to shape mismatch: %s vs %s",
                    _path_string(key),
                    _path_string(mapped_key),
                    value.shape,
                    flat_ref[mapped_key].shape,
                )
                continue
            flat_merged[mapped_key] = value.astype(flat_ref[mapped_key].dtype)
            copied += 1

        logger.info("Loaded %d future tactile encoder tensors from %s.", copied, self.encoder_params_path)
        return _unflatten_params(flat_merged)


@dataclasses.dataclass(frozen=True)
class Pi0WithPatchTactileEncoderWeightLoader(WeightLoader):
    """Loads pi0 base weights plus a pretrained patch-informed tactile encoder.

    The stage-1 checkpoint stores the encoder under `patch_encoder`. The dual-AE
    policy has separate student/teacher tactile encoders, so the same pretrained
    tensors are copied into both `student_force_tokenizer` and
    `teacher_force_tokenizer` when shapes match.
    """

    pi0_params_path: str = "checkpoints/pi0_base/params"
    encoder_params_path: str | None = None

    def load(self, params: at.Params) -> at.Params:
        base_params = _model.restore_params(download.maybe_download(self.pi0_params_path), restore_type=np.ndarray)
        base_params = _augment_with_moe_shared_ffn_weights(base_params, params)
        base_params = _augment_with_mor_action_expert_weights(base_params, params)
        merged = _merge_params(base_params, params, missing_regex=r".*")

        if self.encoder_params_path is None:
            return merged

        encoder_checkpoint = _model.restore_params(
            download.maybe_download(self.encoder_params_path), restore_type=np.ndarray
        )
        flat_ref = _flatten_params(params)
        flat_merged = _flatten_params(merged)
        flat_encoder = _flatten_params(encoder_checkpoint)

        copied = 0
        for key, value in flat_encoder.items():
            if not key or key[0] != "patch_encoder":
                continue
            for target_root in ("student_force_tokenizer", "teacher_force_tokenizer"):
                mapped_key = (target_root, *key[1:])
                if mapped_key not in flat_ref:
                    continue
                if (
                    hasattr(value, "shape")
                    and hasattr(flat_ref[mapped_key], "shape")
                    and value.shape != flat_ref[mapped_key].shape
                ):
                    logger.warning(
                        "Skipping patch tactile encoder param %s -> %s due to shape mismatch: %s vs %s",
                        _path_string(key),
                        _path_string(mapped_key),
                        value.shape,
                        flat_ref[mapped_key].shape,
                    )
                    continue
                flat_merged[mapped_key] = value.astype(flat_ref[mapped_key].dtype)
                copied += 1

        logger.info("Loaded %d patch tactile encoder tensors from %s.", copied, self.encoder_params_path)
        return _unflatten_params(flat_merged)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = _flatten_params(params)
    flat_loaded = _flatten_params(loaded_params)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(_path_string(k))}:
        if k not in result:
            result[k] = flat_ref[k]

    return _unflatten_params(result)


def _augment_with_moe_shared_ffn_weights(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Copies base FFN weights into MOE shared expert keys when shapes are compatible.

    Mapping examples:
      .../mlp/...   -> .../moe/expert_0/...
      .../mlp_1/... -> .../moe_1/expert_0/...
    """
    flat_loaded = _flatten_params(loaded_params)
    flat_ref = _flatten_params(params)
    augmented = dict(flat_loaded)

    mlp_pattern = re.compile(r"mlp(?:_\d+)?")

    copied = 0
    for k, v in flat_loaded.items():
        mlp_index = next(
            (index for index, part in enumerate(k) if isinstance(part, str) and mlp_pattern.fullmatch(part)),
            None,
        )
        if mlp_index is None:
            continue
        mlp_name = k[mlp_index]
        moe_name = mlp_name.replace("mlp", "moe", 1)
        mapped_key = (*k[:mlp_index], moe_name, "expert_0", *k[mlp_index + 1 :])
        if mapped_key in augmented:
            continue
        if mapped_key not in flat_ref:
            continue
        if hasattr(v, "shape") and hasattr(flat_ref[mapped_key], "shape") and v.shape != flat_ref[mapped_key].shape:
            continue
        augmented[mapped_key] = v
        copied += 1

    if copied > 0:
        logger.info("Mapped %d FFN tensors from base MLP to MOE shared expert.", copied)

    return _unflatten_params(augmented)


def _augment_with_mor_action_expert_weights(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Copies action-expert (_1) checkpoint tensors into MOR refine-expert (_2) slots.

    This is useful when loading a 2-expert pi0 checkpoint into a 3-expert MOR model:
      .../attn_1/...       -> .../attn_2/...
      .../mlp_1/...        -> .../mlp_2/...
      .../final_norm_1/... -> .../final_norm_2/...
    """
    flat_loaded = _flatten_params(loaded_params)
    flat_ref = _flatten_params(params)
    augmented = dict(flat_loaded)

    copied = 0
    for k, v in flat_loaded.items():
        mapped_key = tuple(
            part[:-2] + "_2" if isinstance(part, str) and part.endswith("_1") else part for part in k
        )
        if mapped_key == k:
            continue
        if mapped_key in augmented:
            continue
        if mapped_key not in flat_ref:
            continue
        if hasattr(v, "shape") and hasattr(flat_ref[mapped_key], "shape") and v.shape != flat_ref[mapped_key].shape:
            continue
        augmented[mapped_key] = v.astype(flat_ref[mapped_key].dtype) if v.dtype != flat_ref[mapped_key].dtype else v
        copied += 1

    if copied > 0:
        logger.info("Mapped %d tensors from action expert (_1) to MOR refine expert (_2).", copied)

    return _unflatten_params(augmented)
