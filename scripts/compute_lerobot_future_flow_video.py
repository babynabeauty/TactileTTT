"""Compute future-frame RAFT optical flow for a LeRobot dataset and render videos."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

import numpy as np

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError:
    lerobot_dataset = None

try:
    import openpi.transforms as transforms
except ImportError:
    transforms = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tqdm
except ImportError:
    tqdm = None


DEFAULT_IMAGE_KEY_CANDIDATES = (
    ("observation.images.head_camera", "base_0_rgb"),
    ("observation.images.front", "base_0_rgb"),
    ("observation.images.wrist_left_camera", "left_wrist_0_rgb"),
    ("observation.images.left_wrist", "left_wrist_0_rgb"),
    ("observation.images.wrist_camera", "left_wrist_0_rgb"),
    ("observation.images.fixed_camera", "base_0_rgb"),
)


def require_cv2():
    if cv2 is None:
        raise ImportError("This script requires opencv-python. Run it in the project environment with cv2 installed.")
    return cv2


def require_lerobot_dataset():
    if lerobot_dataset is None:
        raise ImportError("This script requires lerobot. Run it in the project environment with LeRobot installed.")
    return lerobot_dataset


def require_transforms():
    if transforms is None:
        raise ImportError("This script requires the local openpi package to be importable.")
    return transforms


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm.tqdm(iterable, **kwargs)


def videojam_magnitude_scale(flow: np.ndarray, sigma: float = 0.08) -> float:
    height, width = flow.shape[:2]
    return float(sigma * np.sqrt(height * height + width * width))


def flow_to_rgb_image(
    flow: np.ndarray,
    sigma: float = 0.08,
    clip_flow: float | None = None,
    background: str = "white",
) -> np.ndarray:
    cv = require_cv2()
    dx = flow[..., 0].astype(np.float32)
    dy = flow[..., 1].astype(np.float32)
    mag, ang = cv.cartToPolar(dx, dy, angleInDegrees=False)

    if clip_flow is not None:
        mag = np.clip(mag, 0, clip_flow)

    magnitude_scale = videojam_magnitude_scale(flow, sigma=sigma)
    mag_norm = np.clip(mag / magnitude_scale, 0, 1)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang / (2 * np.pi)) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = 255
    color = cv.cvtColor(hsv, cv.COLOR_HSV2BGR).astype(np.float32)

    if background == "black":
        return (color * mag_norm[..., None]).astype(np.uint8)
    if background == "white":
        white = np.ones_like(color) * 255
        return (white * (1 - mag_norm[..., None]) + color * mag_norm[..., None]).astype(np.uint8)
    raise ValueError(f"Unsupported background: {background}")


def pad_to_even(image: np.ndarray) -> np.ndarray:
    cv = require_cv2()
    height, width = image.shape[:2]
    pad_bottom = height % 2
    pad_right = width % 2
    if pad_bottom == 0 and pad_right == 0:
        return image
    return cv.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv.BORDER_CONSTANT, value=(0, 0, 0))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def default_stream_name(image_key: str) -> str:
    return safe_name(image_key.split(".")[-1])


def infer_image_keys_and_streams(metadata, requested_image_keys: list[str] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if requested_image_keys:
        return tuple(requested_image_keys), tuple(default_stream_name(key) for key in requested_image_keys)

    features = getattr(metadata, "features", None)
    if features is None:
        features = {}
    inferred = []
    used_streams = set()
    for image_key, stream_name in DEFAULT_IMAGE_KEY_CANDIDATES:
        feature = features.get(image_key)
        if feature is None or feature.get("dtype") != "video" or stream_name in used_streams:
            continue
        inferred.append((image_key, stream_name))
        used_streams.add(stream_name)

    if inferred:
        return tuple(image_key for image_key, _ in inferred), tuple(stream_name for _, stream_name in inferred)

    video_keys = [
        key for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    if video_keys:
        return tuple(video_keys), tuple(default_stream_name(key) for key in video_keys)

    raise ValueError("Could not infer image keys from dataset metadata. Please pass --image-keys explicitly.")


def parse_episode_selector(items: list[str] | None) -> set[int] | None:
    if not items:
        return None
    episodes = set()
    for item in items:
        match = re.fullmatch(r"episode_(\d+)", item)
        episodes.add(int(match.group(1)) if match else int(item))
    return episodes


def to_numpy_image(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def get_episode_ranges(dataset) -> list[tuple[int, int]]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is None:
        raise ValueError("LeRobotDataset does not expose episode_data_index.")
    starts = np.asarray(episode_data_index["from"], dtype=np.int64)
    ends = np.asarray(episode_data_index["to"], dtype=np.int64)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def make_dataset(repo_id: str, video_backend: str):
    lr_dataset = require_lerobot_dataset()
    try:
        return lr_dataset.LeRobotDataset(repo_id, video_backend=video_backend)
    except TypeError:
        return lr_dataset.LeRobotDataset(repo_id)


def flow_path(output_dir: pathlib.Path, episode_index: int, stream_name: str, local_frame_index: int) -> pathlib.Path:
    return (
        output_dir
        / "flows"
        / f"episode_{episode_index:06d}"
        / stream_name
        / f"frame_{local_frame_index:06d}.npy"
    )


def load_or_compute_flows(
    dataset,
    flow_transform,
    output_dir: pathlib.Path,
    image_keys: tuple[str, ...],
    stream_names: tuple[str, ...],
    current_index: int,
    future_index: int,
    episode_index: int,
    local_frame_index: int,
    overwrite: bool,
    save_npy: bool,
) -> dict[str, np.ndarray]:
    paths = {
        stream_name: flow_path(output_dir, episode_index, stream_name, local_frame_index)
        for stream_name in stream_names
    }
    if save_npy and not overwrite and all(path.exists() for path in paths.values()):
        return {stream_name: np.load(path).astype(np.float32) for stream_name, path in paths.items()}

    current_sample = dataset[current_index]
    future_sample = dataset[future_index]
    data = {
        "image": {},
        "future_image": {},
        "future_image_mask": {},
    }
    for image_key, stream_name in zip(image_keys, stream_names, strict=True):
        if image_key not in current_sample:
            raise KeyError(f"Image key {image_key!r} not found in current sample.")
        if image_key not in future_sample:
            raise KeyError(f"Image key {image_key!r} not found in future sample.")
        data["image"][stream_name] = to_numpy_image(current_sample[image_key])
        data["future_image"][stream_name] = to_numpy_image(future_sample[image_key])
        data["future_image_mask"][stream_name] = np.True_

    data = flow_transform(data)
    flows = {}
    for stream_name in stream_names:
        flow = np.asarray(data["future_image"][stream_name], dtype=np.float32)
        if save_npy:
            path = paths[stream_name]
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("wb") as file:
                np.save(file, flow.astype(np.float16))
            tmp_path.replace(path)
        flows[stream_name] = flow
    return flows


def render_episode(
    dataset,
    flow_transform,
    output_dir: pathlib.Path,
    image_keys: tuple[str, ...],
    stream_names: tuple[str, ...],
    episode_index: int,
    start: int,
    end: int,
    future_step: int,
    fps: float,
    sigma: float,
    clip_flow: float | None,
    background: str,
    overwrite: bool,
    save_npy: bool,
    limit_frames: int | None,
) -> int:
    if end <= start:
        return 0

    frame_count = end - start
    if limit_frames is not None:
        frame_count = min(frame_count, limit_frames)
    if frame_count <= 0:
        return 0

    cv = require_cv2()
    writers: dict[str, Any] = {}
    video_paths: dict[str, pathlib.Path] = {}
    try:
        frame_iter = range(frame_count)
        frame_iter = progress(frame_iter, desc=f"episode_{episode_index:06d}", unit="frame", leave=False)
        for local_frame_index in frame_iter:
            current_index = start + local_frame_index
            future_index = min(current_index + future_step, end - 1)
            flows = load_or_compute_flows(
                dataset=dataset,
                flow_transform=flow_transform,
                output_dir=output_dir,
                image_keys=image_keys,
                stream_names=stream_names,
                current_index=current_index,
                future_index=future_index,
                episode_index=episode_index,
                local_frame_index=local_frame_index,
                overwrite=overwrite,
                save_npy=save_npy,
            )

            for stream_name, flow in flows.items():
                frame = pad_to_even(
                    flow_to_rgb_image(flow, sigma=sigma, clip_flow=clip_flow, background=background)
                )
                if stream_name not in writers:
                    stream_video_dir = output_dir / "videos" / stream_name
                    stream_video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = stream_video_dir / f"episode_{episode_index:06d}.mp4"
                    if video_path.exists() and overwrite:
                        video_path.unlink()
                    height, width = frame.shape[:2]
                    writer = cv.VideoWriter(
                        str(video_path),
                        cv.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {video_path}")
                    writers[stream_name] = writer
                    video_paths[stream_name] = video_path
                writers[stream_name].write(frame)
    finally:
        for writer in writers.values():
            writer.release()

    for video_path in video_paths.values():
        print(f"Saved {video_path}")
    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output root directory.")
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=None,
        help="RGB feature keys to read. Defaults to auto-detecting video keys from dataset metadata.",
    )
    parser.add_argument("--stream-names", nargs="+", default=None, help="Output stream names. Defaults to image key suffixes.")
    parser.add_argument("--future-step", type=int, default=32, help="Future frame offset in episode-local frames.")
    parser.add_argument("--episodes", nargs="*", default=None, help="Episode ids or names, e.g. 0 episode_000001.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional maximum number of selected episodes.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Optional maximum frames per episode.")
    parser.add_argument("--raft-model", choices=("large", "small"), default="large", help="Torchvision RAFT model size.")
    parser.add_argument("--raft-weights", default="DEFAULT", help="Torchvision RAFT weights enum name, DEFAULT, or none.")
    parser.add_argument("--device", default=None, help="Torch device for RAFT. Defaults to cuda if available.")
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS. Defaults to dataset metadata fps.")
    parser.add_argument("--sigma", type=float, default=0.08, help="VideoJAM magnitude scale coefficient.")
    parser.add_argument("--clip-flow", type=float, default=None, help="Optional dx/dy clip before saving and rendering.")
    parser.add_argument("--background", choices=("white", "black"), default="white", help="Zero-motion background color.")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw H x W x 2 flow arrays under OUTPUT_DIR/flows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing flow files and videos.")
    parser.add_argument("--video-backend", default="torchcodec", help="LeRobot video backend.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.future_step < 0:
        raise ValueError(f"Expected --future-step >= 0, got {args.future_step}.")
    if args.sigma <= 0:
        raise ValueError(f"Expected --sigma > 0, got {args.sigma}.")
    if args.fps is not None and args.fps <= 0:
        raise ValueError(f"Expected --fps > 0, got {args.fps}.")

    lr_dataset = require_lerobot_dataset()
    openpi_transforms = require_transforms()
    metadata = lr_dataset.LeRobotDatasetMetadata(args.repo_id)

    image_keys, inferred_stream_names = infer_image_keys_and_streams(metadata, args.image_keys)
    stream_names = tuple(args.stream_names or inferred_stream_names)
    if len(stream_names) != len(image_keys):
        raise ValueError("--stream-names must have the same length as --image-keys.")
    stream_names = tuple(safe_name(name) for name in stream_names)

    raft_weights = None if str(args.raft_weights).lower() == "none" else args.raft_weights
    dataset = make_dataset(args.repo_id, args.video_backend)
    episode_ranges = get_episode_ranges(dataset)

    selected_episodes = parse_episode_selector(args.episodes)
    indexed_ranges = [
        (episode_index, start, end)
        for episode_index, (start, end) in enumerate(episode_ranges)
        if selected_episodes is None or episode_index in selected_episodes
    ]
    if args.max_episodes is not None:
        indexed_ranges = indexed_ranges[: args.max_episodes]
    if not indexed_ranges:
        raise ValueError("No episodes selected.")

    fps = float(args.fps or metadata.fps)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    flow_transform = openpi_transforms.ComputeFutureOpticalFlowImages(
        image_keys=stream_names,
        clip_flow=args.clip_flow,
        flow_method="raft",
        raft_model=args.raft_model,
        raft_weights=raft_weights,
        raft_device=args.device,
    )

    total_frames = 0
    episode_iter = progress(indexed_ranges, desc="Episodes", unit="episode")
    for episode_index, start, end in episode_iter:
        video_targets = [output_dir / "videos" / stream / f"episode_{episode_index:06d}.mp4" for stream in stream_names]
        if not args.overwrite and all(path.exists() for path in video_targets):
            print(f"Skip episode_{episode_index:06d}: videos already exist.")
            continue
        total_frames += render_episode(
            dataset=dataset,
            flow_transform=flow_transform,
            output_dir=output_dir,
            image_keys=image_keys,
            stream_names=stream_names,
            episode_index=episode_index,
            start=start,
            end=end,
            future_step=args.future_step,
            fps=fps,
            sigma=args.sigma,
            clip_flow=args.clip_flow,
            background=args.background,
            overwrite=args.overwrite,
            save_npy=args.save_npy,
            limit_frames=args.limit_frames,
        )

    metadata_path = output_dir / "metadata.json"
    metadata_payload = {
        "repo_id": args.repo_id,
        "image_keys": list(image_keys),
        "stream_names": list(stream_names),
        "future_step": args.future_step,
        "fps": fps,
        "raft_model": args.raft_model,
        "raft_weights": raft_weights,
        "device": args.device,
        "sigma": args.sigma,
        "clip_flow": args.clip_flow,
        "background": args.background,
        "save_npy": args.save_npy,
        "episodes": [episode_index for episode_index, _, _ in indexed_ranges],
        "processed_frames_this_run": total_frames,
        "complete": True,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
