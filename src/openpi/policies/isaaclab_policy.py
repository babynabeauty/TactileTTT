import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class IsaacLabInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    state_dim: int = 7
    use_right_wrist_image: bool = False

    def __call__(self, data: dict) -> dict:
        front_camera_image = _parse_image(data["observation.images.front"])
        left_wrist_camera_image = _parse_image(data["observation.images.left_wrist"])

        inputs = {
            "state": data["observation.state"][: self.state_dim],
            "image": {
                "base_0_rgb": front_camera_image,
                "left_wrist_0_rgb": left_wrist_camera_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
        }
        if self.use_right_wrist_image and "observation.images.right_wrist" in data:
            inputs["image"]["right_wrist_0_rgb"] = _parse_image(data["observation.images.right_wrist"])
            inputs["image_mask"]["right_wrist_0_rgb"] = np.True_

        if "action" in data:
            inputs["actions"] = data["action"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            raise ValueError("No task prompt found!")

        return inputs


@dataclasses.dataclass(frozen=True)
class IsaacLabOutputs(transforms.DataTransformFn):
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class IsaacLabTaVLALatentFlowInputs(IsaacLabInputs):
    effort_dim: int = 6

    def __call__(self, data: dict) -> dict:
        inputs = super().__call__(data)

        effort = data["observation.effort"]
        if effort.ndim == 1:
            inputs["effort"] = effort[: self.effort_dim]
        elif effort.ndim == 2:
            inputs["effort"] = effort[:, : self.effort_dim]
        else:
            inputs["effort"] = effort[..., : self.effort_dim]

        if "observation.future_flow.base_0_rgb" in data:
            inputs["flow_img"] = _parse_image(data["observation.future_flow.base_0_rgb"])
        if "observation.future_flow.left_wrist_0_rgb" in data:
            inputs["wrist_flow_img"] = _parse_image(data["observation.future_flow.left_wrist_0_rgb"])

        return inputs


@dataclasses.dataclass(frozen=True)
class IsaacLabTaVLAOutputs(IsaacLabOutputs):
    pass
