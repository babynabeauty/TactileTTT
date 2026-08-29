import dataclasses
import pathlib
from typing import Literal

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


TACTILE_SENSOR_COUNT = 5
TACTILE_BLOCK_SIZE = 384
TACTILE_BLOCK_START = 52
TACTILE_CALC_FORCE_OFFSET = 0
TACTILE_RAW_FORCE_OFFSET = 24
TACTILE_RAW_FORCE_POINTS = 120
XHAND_JOINT_COUNT = 12
SCENE_FLOW_FEATURE_DIM = 10

STATE_KEY_ALIASES = (
    "observation.state",
    "observation/state",
    "state",
    "observation_state",
    "robot_state",
)

ACTION_KEY_ALIASES = ("action", "actions")
TACTILE_KEY_ALIASES = ("tactile", "observation.tactile", "observation/tactile")

PROMPT_KEY_ALIASES = ("prompt", "task")

EPISODE_INDEX_KEY_ALIASES = ("episode_index", "episode/index", "episode.index")
FRAME_INDEX_KEY_ALIASES = ("frame_index", "frame/index", "frame.index", "index")

IMAGE_KEY_ALIASES = {
    "cam_front": (
        "observation.images.cam_front",
        "observation/cam_front_image",
        "observation/cam_front",
        "observation.image.cam_front",
        "observation.cam_front",
        "images.cam_front",
        "image.cam_front",
        "cam_front",
        "front",
        "base_0_rgb",
    ),
    "cam_right": (
        "observation.images.cam_right",
        "observation/cam_right_image",
        "observation/cam_right",
        "observation.image.cam_right",
        "observation.cam_right",
        "images.cam_right",
        "image.cam_right",
        "cam_right",
        "right",
        "left_wrist_0_rgb",
    ),
    "cam_left": (
        "observation.images.cam_left",
        "observation/cam_left_image",
        "observation/cam_left",
        "observation.image.cam_left",
        "observation.cam_left",
        "images.cam_left",
        "image.cam_left",
        "cam_left",
        "left",
        "right_wrist_0_rgb",
    ),
}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _as_state_sequence(state) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        return state[None, :]
    if state.ndim == 2:
        return state
    raise ValueError(f"Expected xhand observation.state with shape [D] or [T, D], got {state.shape}.")


def _first_present(data: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key and key in data:
            return key
    return None


def _lookup_nested_image(data: dict, key: str):
    image_dict = data.get("image")
    if isinstance(image_dict, dict) and key in image_dict:
        return image_dict[key]

    images_dict = data.get("images")
    if isinstance(images_dict, dict) and key in images_dict:
        return images_dict[key]

    observation_dict = data.get("observation")
    if isinstance(observation_dict, dict):
        obs_images = observation_dict.get("images")
        if isinstance(obs_images, dict) and key in obs_images:
            return obs_images[key]
        if key in observation_dict:
            return observation_dict[key]

    return None


def _get_required(data: dict, keys: tuple[str, ...], field_name: str):
    key = _first_present(data, keys)
    if key is not None:
        return data[key]

    for key in keys:
        value = _lookup_nested_image(data, key)
        if value is not None:
            return value

    available = ", ".join(sorted(map(str, data.keys())))
    raise KeyError(f"Missing `{field_name}`. Tried keys={keys}. Available top-level keys: [{available}]")


def _get_optional(data: dict, keys: tuple[str, ...]):
    key = _first_present(data, keys)
    if key is not None:
        return data[key]

    for key in keys:
        value = _lookup_nested_image(data, key)
        if value is not None:
            return value

    return None


def _image_keys(configured_key: str | None, camera: str) -> tuple[str, ...]:
    keys = []
    if configured_key:
        keys.append(configured_key)
    keys.extend(IMAGE_KEY_ALIASES[camera])
    return tuple(dict.fromkeys(keys))


def _to_scalar_int(value) -> int:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return int(arr)
    return int(arr.reshape(-1)[0])


def _extract_current_calc_force(state: np.ndarray) -> np.ndarray:
    required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    if state.shape[-1] < required_dim:
        raise ValueError(
            "XHand calc-force tactile extraction requires the full raw observation.state "
            f"with at least {required_dim} values, got {state.shape[-1]}."
        )
    return np.stack(
        [
            state[
                TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_CALC_FORCE_OFFSET : TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_CALC_FORCE_OFFSET
                + 3
            ]
            for sensor_id in range(TACTILE_SENSOR_COUNT)
        ],
        axis=0,
    ).astype(np.float32)


def _extract_current_raw_force(state: np.ndarray) -> np.ndarray:
    required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    if state.shape[-1] < required_dim:
        raise ValueError(
            "XHand raw-force tactile extraction requires the full raw observation.state "
            f"with at least {required_dim} values, got {state.shape[-1]}."
        )
    return np.stack(
        [
            state[
                TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_RAW_FORCE_OFFSET : TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_RAW_FORCE_OFFSET
                + TACTILE_RAW_FORCE_POINTS * 3
            ].reshape(TACTILE_RAW_FORCE_POINTS, 3)
            for sensor_id in range(TACTILE_SENSOR_COUNT)
        ],
        axis=0,
    ).astype(np.float32)


def _load_scene_flow_npz(
    *,
    root: pathlib.Path,
    episode_index: int,
    frame_index: int,
    future_step: int,
    num_points: int,
    required: bool,
) -> np.ndarray:
    target_frame = frame_index + future_step
    path = (
        root
        / f"episode_{episode_index:06d}"
        / "pairs"
        / f"frame_{frame_index:06d}_to_{target_frame:06d}"
        / "nn_scene_flow_robot_object.npz"
    )
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing 3D scene-flow npz: {path}")
        return np.zeros((num_points, SCENE_FLOW_FEATURE_DIM), dtype=np.float32)

    data = np.load(path)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    target_xyz = np.asarray(data["target_xyz"], dtype=np.float32)
    flow_xyz = np.asarray(data["flow_xyz"], dtype=np.float32)
    class_id = np.asarray(data["class_id"], dtype=np.float32).reshape(-1, 1) / 2.0
    features = np.concatenate([xyz, target_xyz, flow_xyz, class_id], axis=-1)

    if features.shape[0] > num_points:
        rng = np.random.default_rng(episode_index * 1_000_003 + frame_index)
        indices = rng.choice(features.shape[0], size=num_points, replace=False)
        features = features[np.sort(indices)]
    elif features.shape[0] < num_points:
        pad = np.zeros((num_points - features.shape[0], features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=0)
    return features.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class XHandPi0Inputs(transforms.DataTransformFn):
    """Converts ur7e_xhand LeRobot samples into vanilla pi0 inputs."""

    model_type: _model.ModelType
    state_dim: int = 18
    primary_image_key: str = "observation.images.cam_front"
    wrist_image_key: str = "observation.images.cam_right"
    extra_image_key: str | None = "observation.images.cam_left"
    prompt_key: str = "prompt"
    task_key: str = "task"

    def __call__(self, data: dict) -> dict:
        state_seq = _as_state_sequence(_get_required(data, STATE_KEY_ALIASES, "observation.state"))
        current_state = state_seq[-1]

        primary_image = _get_required(data, _image_keys(self.primary_image_key, "cam_front"), self.primary_image_key)
        wrist_image = _get_required(data, _image_keys(self.wrist_image_key, "cam_right"), self.wrist_image_key)

        inputs = {
            "state": self._extract_proprio(current_state),
            "image": {
                "base_0_rgb": _parse_image(primary_image),
                "left_wrist_0_rgb": _parse_image(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
        }

        extra_image = _get_optional(data, _image_keys(self.extra_image_key, "cam_left")) if self.extra_image_key else None
        if extra_image is not None:
            inputs["image"]["right_wrist_0_rgb"] = _parse_image(extra_image)
            inputs["image_mask"]["right_wrist_0_rgb"] = np.True_
        elif self.model_type == _model.ModelType.PI0_FAST:
            inputs["image"]["right_wrist_0_rgb"] = np.zeros_like(inputs["image"]["left_wrist_0_rgb"])
            inputs["image_mask"]["right_wrist_0_rgb"] = np.False_

        action_key = _first_present(data, ACTION_KEY_ALIASES)
        if action_key is not None:
            inputs["actions"] = np.asarray(data[action_key], dtype=np.float32)

        prompt_key = _first_present(data, tuple(dict.fromkeys((self.prompt_key, self.task_key, *PROMPT_KEY_ALIASES))))
        if prompt_key is not None:
            inputs["prompt"] = data[prompt_key]

        return inputs

    def _extract_proprio(self, state: np.ndarray) -> np.ndarray:
        if state.shape[-1] == self.state_dim:
            return state.astype(np.float32)

        hand_joint_pos_indices = 28 + 2 * np.arange(XHAND_JOINT_COUNT)
        proprio = np.concatenate([state[:6], state[hand_joint_pos_indices]], axis=0).astype(np.float32)
        return proprio[: self.state_dim]


@dataclasses.dataclass(frozen=True)
class XHandPi0StateTactileInputs(XHandPi0Inputs):
    """Concatenates the current XHand tactile observation into the pi0/pi0.5 state input."""

    tactile_mode: Literal["calc_force", "raw_force"] = "calc_force"

    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        tactile_key = _first_present(data, TACTILE_KEY_ALIASES)
        if tactile_key is not None:
            tactile = np.asarray(data[tactile_key], dtype=np.float32)
            if self.tactile_mode == "calc_force" and tactile.shape == (TACTILE_SENSOR_COUNT * 3,):
                tactile = tactile.reshape(TACTILE_SENSOR_COUNT, 3)
            elif self.tactile_mode == "raw_force" and tactile.shape == (
                TACTILE_SENSOR_COUNT * TACTILE_RAW_FORCE_POINTS * 3,
            ):
                tactile = tactile.reshape(TACTILE_SENSOR_COUNT, TACTILE_RAW_FORCE_POINTS, 3)

            expected_shape = (
                (TACTILE_SENSOR_COUNT, 3)
                if self.tactile_mode == "calc_force"
                else (TACTILE_SENSOR_COUNT, TACTILE_RAW_FORCE_POINTS, 3)
            )
            if tactile.shape != expected_shape:
                raise ValueError(
                    f"Expected explicit XHand tactile observation for mode={self.tactile_mode!r} "
                    f"with shape {expected_shape}, got {tactile.shape}."
                )
        else:
            state_seq = _as_state_sequence(_get_required(data, STATE_KEY_ALIASES, "observation.state"))
            if self.tactile_mode == "calc_force":
                tactile = _extract_current_calc_force(state_seq[-1])
            elif self.tactile_mode == "raw_force":
                tactile = _extract_current_raw_force(state_seq[-1])
            else:
                raise ValueError(f"Unsupported tactile_mode={self.tactile_mode!r}.")

        inputs["state"] = np.concatenate(
            [np.asarray(inputs["state"], dtype=np.float32), tactile.reshape(-1).astype(np.float32)],
            axis=-1,
        )
        return inputs


@dataclasses.dataclass(frozen=True)
class XHandTactileObsInputs(XHandPi0Inputs):
    """Adds the current five-finger resultant force under the shared effort key."""

    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)
        tactile_key = _first_present(data, TACTILE_KEY_ALIASES)
        if tactile_key is not None:
            tactile = np.asarray(data[tactile_key], dtype=np.float32)
            if tactile.shape == (TACTILE_SENSOR_COUNT * 3,):
                tactile = tactile.reshape(TACTILE_SENSOR_COUNT, 3)
            if tactile.shape != (TACTILE_SENSOR_COUNT, 3):
                raise ValueError(
                    "Expected explicit XHand tactile observation with shape "
                    f"({TACTILE_SENSOR_COUNT}, 3), got {tactile.shape}."
                )
        else:
            state_seq = _as_state_sequence(_get_required(data, STATE_KEY_ALIASES, "observation.state"))
            tactile = _extract_current_calc_force(state_seq[-1])
        inputs["effort"] = tactile
        return inputs


@dataclasses.dataclass(frozen=True)
class XHandTactileFlowInputs(transforms.DataTransformFn):
    """Converts ur7e_xhand LeRobot samples into pi0 latent-flow inputs.

    The existing ForceWAM model names this stream `effort`; for this xhand
    config it carries tactile history/future values instead of a 6D force sensor.
    """

    model_type: _model.ModelType
    tactile_mode: Literal["calc_force", "raw_force"] = "calc_force"
    structured_tactile: bool = False
    tactile_history_frames: int = 10
    state_dim: int = 18
    primary_image_key: str = "observation.images.cam_front"
    wrist_image_key: str = "observation.images.cam_left"
    extra_image_key: str | None = "observation.images.cam_right"
    future_flow_key: str | None = "observation.future_flow.cam_front"
    future_wrist_flow_key: str | None = "observation.future_flow.cam_left"
    scene_flow_root: str | None = None
    scene_flow_future_step: int = 32
    scene_flow_num_points: int = 4096
    scene_flow_required: bool = False
    prompt_key: str = "prompt"
    task_key: str = "task"

    def __call__(self, data: dict) -> dict:
        state_seq = _as_state_sequence(_get_required(data, STATE_KEY_ALIASES, "observation.state"))
        current_state = state_seq[min(max(self.tactile_history_frames - 1, 0), state_seq.shape[0] - 1)]

        primary_image = _get_required(data, _image_keys(self.primary_image_key, "cam_front"), self.primary_image_key)
        wrist_camera = "cam_left" if self.wrist_image_key and "cam_left" in self.wrist_image_key else "cam_right"
        wrist_image = _get_required(data, _image_keys(self.wrist_image_key, wrist_camera), self.wrist_image_key)

        inputs = {
            "state": self._extract_proprio(current_state),
            "effort": self._extract_tactile(state_seq),
            "image": {
                "base_0_rgb": _parse_image(primary_image),
                "left_wrist_0_rgb": _parse_image(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
        }

        extra_camera = "cam_right" if self.extra_image_key and "cam_right" in self.extra_image_key else "cam_left"
        extra_image = _get_optional(data, _image_keys(self.extra_image_key, extra_camera)) if self.extra_image_key else None
        if extra_image is not None:
            inputs["image"]["right_wrist_0_rgb"] = _parse_image(extra_image)
            inputs["image_mask"]["right_wrist_0_rgb"] = np.True_
        elif self.model_type == _model.ModelType.PI0_FAST:
            inputs["image"]["right_wrist_0_rgb"] = np.zeros_like(inputs["image"]["left_wrist_0_rgb"])
            inputs["image_mask"]["right_wrist_0_rgb"] = np.False_

        if self.future_flow_key and self.future_flow_key in data:
            inputs["flow_img"] = _parse_image(data[self.future_flow_key])
        if self.future_wrist_flow_key and self.future_wrist_flow_key in data:
            inputs["wrist_flow_img"] = _parse_image(data[self.future_wrist_flow_key])
        if self.scene_flow_root is not None:
            episode_key = _first_present(data, EPISODE_INDEX_KEY_ALIASES)
            frame_key = _first_present(data, FRAME_INDEX_KEY_ALIASES)
            if episode_key is not None and frame_key is not None:
                inputs["scene_flow"] = _load_scene_flow_npz(
                    root=pathlib.Path(self.scene_flow_root),
                    episode_index=_to_scalar_int(data[episode_key]),
                    frame_index=_to_scalar_int(data[frame_key]),
                    future_step=self.scene_flow_future_step,
                    num_points=self.scene_flow_num_points,
                    required=self.scene_flow_required,
                )
            elif self.scene_flow_required:
                available = ", ".join(sorted(map(str, data.keys())))
                raise KeyError(
                    "Missing episode/frame index for required 3D scene-flow lookup. "
                    f"Available top-level keys: [{available}]"
                )

        action_key = _first_present(data, ACTION_KEY_ALIASES)
        if action_key is not None:
            inputs["actions"] = np.asarray(data[action_key], dtype=np.float32)

        prompt_key = _first_present(data, tuple(dict.fromkeys((self.prompt_key, self.task_key, *PROMPT_KEY_ALIASES))))
        if prompt_key is not None:
            inputs["prompt"] = data[prompt_key]

        # Preserve identifiers for episode-level offline analysis. These keys are
        # ignored by Observation.from_dict and therefore never enter the policy.
        for metadata_key in ("episode_index", "frame_index", "task_index", "timestamp"):
            if metadata_key in data:
                inputs[metadata_key] = data[metadata_key]

        return inputs

    def _extract_proprio(self, state: np.ndarray) -> np.ndarray:
        if state.shape[-1] == self.state_dim:
            return state.astype(np.float32)

        hand_joint_pos_indices = 28 + 2 * np.arange(XHAND_JOINT_COUNT)
        proprio = np.concatenate([state[:6], state[hand_joint_pos_indices]], axis=0).astype(np.float32)
        return proprio[: self.state_dim]

    def _extract_tactile(self, state_seq: np.ndarray) -> np.ndarray:
        if self.tactile_mode == "calc_force":
            sensor_chunks = [
                state_seq[
                    :,
                    TACTILE_BLOCK_START
                    + sensor_id * TACTILE_BLOCK_SIZE
                    + TACTILE_CALC_FORCE_OFFSET : TACTILE_BLOCK_START
                    + sensor_id * TACTILE_BLOCK_SIZE
                    + TACTILE_CALC_FORCE_OFFSET
                    + 3,
                ]
                for sensor_id in range(TACTILE_SENSOR_COUNT)
            ]
            tactile = (
                np.stack(sensor_chunks, axis=-2)
                if self.structured_tactile
                else np.concatenate(sensor_chunks, axis=-1)
            )
        elif self.tactile_mode == "raw_force":
            sensor_chunks = [
                state_seq[
                    :,
                    TACTILE_BLOCK_START
                    + sensor_id * TACTILE_BLOCK_SIZE
                    + TACTILE_RAW_FORCE_OFFSET : TACTILE_BLOCK_START
                    + sensor_id * TACTILE_BLOCK_SIZE
                    + TACTILE_RAW_FORCE_OFFSET
                    + TACTILE_RAW_FORCE_POINTS * 3,
                ]
                for sensor_id in range(TACTILE_SENSOR_COUNT)
            ]
            if self.structured_tactile:
                tactile = np.stack(
                    [
                        chunk.reshape(state_seq.shape[0], TACTILE_RAW_FORCE_POINTS, 3)
                        for chunk in sensor_chunks
                    ],
                    axis=1,
                )
            else:
                tactile = np.concatenate(sensor_chunks, axis=-1)
        else:
            raise ValueError(f"Unsupported tactile_mode={self.tactile_mode!r}.")

        return tactile.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class XHandPatchTactilePretrainInputs(transforms.DataTransformFn):
    """Lightweight raw-tactile inputs for patch encoder pretraining.

    Stage-1 patch encoder pretraining does not use RGB or language, so this
    transform emits only proprioception, tactile input, and actions.
    """

    tactile_mode: Literal["raw_force"] = "raw_force"
    structured_tactile: bool = True
    tactile_history_frames: int = 10
    state_dim: int = 18

    def __call__(self, data: dict) -> dict:
        state_seq = _as_state_sequence(_get_required(data, STATE_KEY_ALIASES, "observation.state"))
        current_state = state_seq[min(max(self.tactile_history_frames - 1, 0), state_seq.shape[0] - 1)]

        inputs = {
            "state": self._extract_proprio(current_state),
            "effort": self._extract_tactile(state_seq),
        }

        action_key = _first_present(data, ACTION_KEY_ALIASES)
        if action_key is not None:
            inputs["actions"] = np.asarray(data[action_key], dtype=np.float32)

        return inputs

    def _extract_proprio(self, state: np.ndarray) -> np.ndarray:
        if state.shape[-1] == self.state_dim:
            return state.astype(np.float32)

        hand_joint_pos_indices = 28 + 2 * np.arange(XHAND_JOINT_COUNT)
        proprio = np.concatenate([state[:6], state[hand_joint_pos_indices]], axis=0).astype(np.float32)
        return proprio[: self.state_dim]

    def _extract_tactile(self, state_seq: np.ndarray) -> np.ndarray:
        sensor_chunks = [
            state_seq[
                :,
                TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_RAW_FORCE_OFFSET : TACTILE_BLOCK_START
                + sensor_id * TACTILE_BLOCK_SIZE
                + TACTILE_RAW_FORCE_OFFSET
                + TACTILE_RAW_FORCE_POINTS * 3,
            ]
            for sensor_id in range(TACTILE_SENSOR_COUNT)
        ]
        tactile = np.stack(
            [
                chunk.reshape(state_seq.shape[0], TACTILE_RAW_FORCE_POINTS, 3)
                for chunk in sensor_chunks
            ],
            axis=1,
        )
        return tactile.astype(np.float32)


@dataclasses.dataclass(frozen=True)
class XHandTactileFlowOutputs(transforms.DataTransformFn):
    action_dim: int = 18

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
