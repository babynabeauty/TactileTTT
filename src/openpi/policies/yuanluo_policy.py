import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_yuanluo_example() -> dict:
    """Creates a random input example for the Reactive Diffusion Policy."""
    return {
        "observation/front_camera": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "observation/left_wrist_camera": np.random.randint(256, size=(720, 1280, 3), dtype=np.uint8),
        "observation/gelsight_left": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        # "observation/gelsight_right": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/left_wrench": np.random.rand(6),
        "observation/left_state": np.random.rand(8),
        "actions": np.random.rand(8),
        "prompt": "Please Insert the USB hub into the USB slot on the board.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class YuanluoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format for the Reactive Diffusion Policy dataset.
    It is used for both training and inference.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    state_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Parse images to uint8 (H,W,C)
        front_camera_image = _parse_image(data["observation.images.head_camera"])
        left_wrist_camera_image = _parse_image(data["observation.images.wrist_left_camera"])
        # gelsight_left_image = _parse_image(data["observation.images.gelsight_left"])
        # gelsight_right_image = _parse_image(data["observation/gelsight_right"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation.state"][:self.state_dim],
            "image": {
                "base_0_rgb": front_camera_image,
                "left_wrist_0_rgb": left_wrist_camera_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                # "right_wrist_0_rgb": gelsight_left_image, #np.zeros_like(left_wrist_camera_image),
                # "right_wrist_0_rgb": np.zeros_like(left_wrist_camera_image), # without Gelsight
                # "right_wrist_0_rgb": gelsight_left_image, # with gelsight
                # "gelsight_left_rgb": gelsight_left_image,
                # "gelsight_right_rgb": gelsight_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                # "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_, # without gelsight
                # "right_wrist_0_rgb": np.True_, # # with gelsight
                # "gelsight_right_rgb": np.True_,
            },
        }
        if "observation.images.future_flow" in data:
            inputs["flow_img"] = _parse_image(data["observation.images.future_flow"])

        # Pad actions to the model action dimension. Actions are only available during training.
        if "action" in data:
            inputs["actions"] = data["action"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            raise ValueError("No task prompt found!")

        return inputs


@dataclasses.dataclass(frozen=True)
class YuanluoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the dataset specific format for Reactive Diffusion Policy.
    It is used for inference only.
    """
    action_dim = 7 

    def __call__(self, data: dict) -> dict:
        # Return the actions with the correct dimension.
        # Assuming 7 actions (e.g., 3 for position, 4 for quaternion).
        # This needs to match the actual action dimension of your robot.
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
    

@dataclasses.dataclass(frozen=True)
class YuanluoTaVLAInputs(YuanluoInputs):
    use_future_rgb_instead_of_flow: bool = False

    def _split_camera_stream(
        self,
        camera,
        *,
        camera_name: str,
        require_future_frame: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        camera = np.asarray(camera)
        if not self.use_future_rgb_instead_of_flow:
            return _parse_image(camera), None

        if camera.ndim < 4:
            if require_future_frame:
                raise ValueError(
                    f"YuanluoTaVLAInputs expected `{camera_name}` to contain "
                    "current and future frames when `use_future_rgb_instead_of_flow=True`."
                )
            return _parse_image(camera), None

        if camera.shape[0] < 2:
            if require_future_frame:
                raise ValueError(
                    f"YuanluoTaVLAInputs expected `{camera_name}` to contain "
                    "current and future frames when `use_future_rgb_instead_of_flow=True`."
                )
            return _parse_image(camera[0]), None

        current_frame = _parse_image(camera[0])
        future_frame = _parse_image(camera[1])
        return current_frame, future_frame

    def __call__(self, data: dict) -> dict:
        current_head_camera, future_rgb_img = self._split_camera_stream(
            data["observation.images.head_camera"],
            camera_name="observation.images.head_camera",
            require_future_frame="action" in data,
        )
        current_wrist_camera, future_wrist_rgb_img = self._split_camera_stream(
            data["observation.images.wrist_left_camera"],
            camera_name="observation.images.wrist_left_camera",
            require_future_frame="action" in data,
        )
        data = dict(data)
        data["observation.images.head_camera"] = current_head_camera
        data["observation.images.wrist_left_camera"] = current_wrist_camera
        inputs = super().__call__(data)
        inputs["effort"] = data["observation.effort"] # Append 6-axis force sensor data 
        if future_rgb_img is not None:
            inputs["future_rgb_img"] = future_rgb_img
        if future_wrist_rgb_img is not None:
            inputs["future_wrist_rgb_img"] = future_wrist_rgb_img
        if "observation.images.future_flow" in data:
            inputs["flow_img"] = _parse_image(data["observation.images.future_flow"])
        if "observation.images.future_wrist_flow" in data:
            inputs["wrist_flow_img"] = _parse_image(data["observation.images.future_wrist_flow"])
        
        # Optionally pass through contact-related supervision signals if present
        if "observation.is_contact" in data:
            inputs["is_contact"] = data["observation.is_contact"]
        return inputs

@dataclasses.dataclass(frozen=True)
class YuanluoTaVLAOutputs(YuanluoOutputs):
    pass
