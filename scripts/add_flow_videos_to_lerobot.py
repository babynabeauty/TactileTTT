"""Add pre-rendered optical-flow videos to an existing LeRobot v2 dataset.

Expected source layout, produced by scripts/flow_cache_to_videos.py:
    FLOW_VIDEOS_DIR/base_0_rgb/episode_000072.mp4
    FLOW_VIDEOS_DIR/left_wrist_0_rgb/episode_000072.mp4

Target LeRobot layout:
    ~/.cache/huggingface/lerobot/REPO_ID/videos/chunk-000/observation.future_flow.base_0_rgb/episode_000072.mp4
    ~/.cache/huggingface/lerobot/REPO_ID/videos/chunk-000/observation.future_flow.left_wrist_0_rgb/episode_000072.mp4

The script updates meta/info.json so LeRobot can discover the new video keys.
It does not rewrite parquet files: LeRobot v2 decodes video features from
meta.video_keys, timestamps, and the video_path template.
"""

from __future__ import annotations

import argparse
import os
import json
import pathlib
import re
import shutil
import subprocess
from typing import Any


DEFAULT_STREAM_TO_FEATURE = {
    "base_0_rgb": "observation.future_flow.base_0_rgb",
    "left_wrist_0_rgb": "observation.future_flow.left_wrist_0_rgb",
}


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def episode_index_from_name(path: pathlib.Path) -> int:
    match = re.fullmatch(r"episode_(\d{6})\.mp4", path.name)
    if match is None:
        raise ValueError(f"Expected video filename like episode_000000.mp4, got {path}")
    return int(match.group(1))


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def parse_stream_map(items: list[str] | None) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_STREAM_TO_FEATURE)

    stream_map = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --map item STREAM=FEATURE_KEY, got {item!r}")
        stream, feature_key = item.split("=", 1)
        stream = stream.strip()
        feature_key = feature_key.strip()
        if not stream or not feature_key:
            raise ValueError(f"Expected non-empty STREAM=FEATURE_KEY, got {item!r}")
        stream_map[stream] = feature_key
    return stream_map


def default_lerobot_home() -> pathlib.Path:
    return pathlib.Path(
        os.environ.get("HF_LEROBOT_HOME")
        or os.environ.get("LEROBOT_HOME")
        or pathlib.Path.home() / ".cache" / "huggingface" / "lerobot"
    )


def resolve_dataset_root(repo_id: str, dataset_root: pathlib.Path | None) -> pathlib.Path:
    if dataset_root is not None:
        return dataset_root.expanduser().resolve()
    return (default_lerobot_home() / repo_id).expanduser().resolve()


def probe_video_info(video_path: pathlib.Path, fallback_fps: int | float) -> tuple[list[int], dict[str, Any]]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,pix_fmt,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout).get("streams", [])
        if streams:
            stream = streams[0]
            width = int(stream["width"])
            height = int(stream["height"])
            fps = fallback_fps
            rate = stream.get("r_frame_rate")
            if rate and "/" in rate:
                num, den = rate.split("/", 1)
                if float(den) != 0:
                    fps = float(num) / float(den)
            info = {
                "video.height": height,
                "video.width": width,
                "video.codec": stream.get("codec_name", "unknown"),
                "video.pix_fmt": stream.get("pix_fmt", "unknown"),
                "video.is_depth_map": False,
                "video.fps": int(round(fps)),
                "video.channels": 3,
                "has_audio": False,
            }
            return [height, width, 3], info

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Need either ffprobe or cv2 to infer video shape/info.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for probing: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or fallback_fps
    cap.release()
    info = {
        "video.height": height,
        "video.width": width,
        "video.codec": "unknown",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": int(round(fps)),
        "video.channels": 3,
        "has_audio": False,
    }
    return [height, width, 3], info


def collect_episode_videos(stream_dir: pathlib.Path) -> dict[int, pathlib.Path]:
    videos = {}
    for path in sorted(stream_dir.glob("episode_*.mp4")):
        if path.name.endswith(".raw.mp4"):
            continue
        videos[episode_index_from_name(path)] = path
    return videos


def copy_video(src: pathlib.Path, dst: pathlib.Path, mode: str, overwrite: bool, dry_run: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        if not dry_run:
            dst.unlink()

    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def add_feature(info: dict[str, Any], feature_key: str, shape: list[int], video_info: dict[str, Any]) -> None:
    info.setdefault("features", {})[feature_key] = {
        "dtype": "video",
        "shape": shape,
        "names": ["height", "width", "channels"],
        "info": video_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", required=True, help="LeRobot dataset repo id.")
    parser.add_argument(
        "--dataset-root",
        type=pathlib.Path,
        default=None,
        help="Optional explicit dataset root. Defaults to $HF_LEROBOT_HOME/REPO_ID or ~/.cache/huggingface/lerobot/REPO_ID.",
    )
    parser.add_argument(
        "--flow-videos-dir",
        type=pathlib.Path,
        required=True,
        help="Flow video directory containing per-stream episode mp4 files.",
    )
    parser.add_argument(
        "--map",
        nargs="*",
        default=None,
        help=(
            "Stream to LeRobot feature mapping. Default: "
            "base_0_rgb=observation.future_flow.base_0_rgb "
            "left_wrist_0_rgb=observation.future_flow.left_wrist_0_rgb"
        ),
    )
    parser.add_argument("--mode", choices=("copy", "symlink", "hardlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target videos/features.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if some episode videos are missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without writing files.")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.repo_id, args.dataset_root)
    flow_videos_dir = args.flow_videos_dir.expanduser().resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot info.json: {info_path}")

    info = read_json(info_path)
    chunks_size = int(info["chunks_size"])
    total_episodes = int(info["total_episodes"])
    fps = info.get("fps", 30)
    video_path_template = info.get("video_path")
    if not video_path_template:
        raise ValueError("Dataset info.json does not define video_path.")

    stream_map = parse_stream_map(args.map)
    expected_episodes = set(range(total_episodes))
    copied = 0

    for stream_name, feature_key in stream_map.items():
        stream_dir = flow_videos_dir / stream_name
        if not stream_dir.exists():
            raise FileNotFoundError(f"Missing flow video stream directory: {stream_dir}")

        videos = collect_episode_videos(stream_dir)
        missing = sorted(expected_episodes - set(videos))
        if missing and not args.allow_missing:
            preview = ", ".join(f"episode_{idx:06d}" for idx in missing[:10])
            raise FileNotFoundError(
                f"{stream_dir} is missing {len(missing)} / {total_episodes} episodes, e.g. {preview}. "
                "Use --allow-missing if this is intentional."
            )

        first_video = videos[min(videos)]
        shape, video_info = probe_video_info(first_video, fallback_fps=fps)
        if feature_key in info.get("features", {}) and not args.overwrite:
            raise ValueError(f"Feature already exists in info.json: {feature_key}. Use --overwrite to replace it.")

        add_feature(info, feature_key, shape=shape, video_info=video_info)
        for episode_index, src in sorted(videos.items()):
            chunk = episode_chunk(episode_index, chunks_size)
            rel_dst = video_path_template.format(
                episode_chunk=chunk,
                video_key=feature_key,
                episode_index=episode_index,
            )
            dst = dataset_root / rel_dst
            copy_video(src, dst, mode=args.mode, overwrite=args.overwrite, dry_run=args.dry_run)
            copied += 1

        print(f"Added stream {stream_name} -> {feature_key}: {len(videos)} videos, shape={shape}")

    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    info["total_videos"] = total_episodes * len(video_keys)

    if args.dry_run:
        print(f"[dry-run] Would update {info_path}")
        print(f"[dry-run] Would add/copy {copied} videos under {dataset_root / 'videos'}")
        print(f"[dry-run] New video keys: {video_keys}")
        return

    backup_path = info_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy2(info_path, backup_path)
    write_json(info_path, info)
    print(f"Updated {info_path}")
    print(f"Backup: {backup_path}")
    print(f"Total video keys: {len(video_keys)}, total_videos={info['total_videos']}")


if __name__ == "__main__":
    main()
