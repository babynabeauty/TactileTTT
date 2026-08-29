"""Render LeRobot videos with synchronized XHand fingertip force plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from torchcodec.decoders import VideoDecoder


SENSOR_COUNT = 5
AXES = ("fx", "fy", "fz")
COLORS = [
    (66, 135, 245),
    (46, 204, 113),
    (245, 176, 65),
    (155, 89, 182),
    (231, 76, 60),
]


def _load_state_names(repo_id: Path) -> list[str]:
    info = json.loads((repo_id / "meta" / "info.json").read_text())
    return info["features"]["observation.state"]["names"]


def _calc_force_indices(names: list[str]) -> np.ndarray:
    indices = []
    for sensor_id in range(SENSOR_COUNT):
        sensor_indices = []
        for axis in AXES:
            name = f"hand_tactile_sensor_{sensor_id}.calc_force.{axis}"
            try:
                sensor_indices.append(names.index(name))
            except ValueError as exc:
                raise ValueError(f"Missing tactile state field: {name}") from exc
        indices.append(sensor_indices)
    return np.asarray(indices, dtype=np.int64)


def _read_episode_forces(repo_id: Path, episode_index: int, force_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parquet_path = repo_id / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path, columns=["observation.state", "timestamp"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    forces = states[:, force_indices.reshape(-1)].reshape(states.shape[0], SENSOR_COUNT, 3)
    return forces, timestamps


def _find_video(repo_id: Path, camera: str, episode_index: int) -> Path:
    path = repo_id / "videos" / "chunk-000" / camera / f"episode_{episode_index:06d}.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _draw_text(img: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.5, color=(235, 235, 235)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _normalize_plot_values(values: np.ndarray, lo: float, hi: float, top: int, bottom: int) -> np.ndarray:
    denom = max(hi - lo, 1e-6)
    y = bottom - (values - lo) / denom * (bottom - top)
    return np.clip(y, top, bottom).astype(np.int32)


def _draw_magnitude_plot(
    panel: np.ndarray,
    magnitudes: np.ndarray,
    frame_idx: int,
    plot_box: tuple[int, int, int, int],
    y_limit: float | None,
) -> None:
    x0, y0, x1, y1 = plot_box
    cv2.rectangle(panel, (x0, y0), (x1, y1), (55, 55, 55), 1)
    _draw_text(panel, "force magnitude |F| per fingertip", (x0, y0 - 8), 0.55)

    hi = float(y_limit) if y_limit and y_limit > 0 else float(np.nanpercentile(magnitudes, 99))
    hi = max(hi, 1.0)
    lo = 0.0
    for tick in np.linspace(lo, hi, 5):
        y = int(y1 - (tick - lo) / (hi - lo) * (y1 - y0))
        cv2.line(panel, (x0, y), (x1, y), (40, 40, 40), 1)
        _draw_text(panel, f"{tick:5.1f}", (x0 + 4, y - 3), 0.38, (170, 170, 170))

    n = magnitudes.shape[0]
    xs = np.linspace(x0, x1, n).astype(np.int32)
    for sensor_id in range(SENSOR_COUNT):
        ys = _normalize_plot_values(magnitudes[:, sensor_id], lo, hi, y0, y1)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, COLORS[sensor_id], 2, cv2.LINE_AA)

    cursor_x = int(x0 + frame_idx / max(n - 1, 1) * (x1 - x0))
    cv2.line(panel, (cursor_x, y0), (cursor_x, y1), (245, 245, 245), 1)


def _draw_current_force_bars(
    panel: np.ndarray,
    forces: np.ndarray,
    frame_idx: int,
    box: tuple[int, int, int, int],
    component_limit: float | None,
) -> None:
    x0, y0, x1, y1 = box
    cv2.rectangle(panel, (x0, y0), (x1, y1), (55, 55, 55), 1)
    _draw_text(panel, "current calc_force vector", (x0, y0 - 8), 0.55)
    cur = forces[frame_idx]
    abs_hi = float(component_limit) if component_limit and component_limit > 0 else float(np.nanpercentile(np.abs(forces), 99))
    abs_hi = max(abs_hi, 1.0)
    zero_x = x0 + 165
    max_bar = max(60, x1 - zero_x - 25)

    axis_colors = [(80, 180, 255), (80, 230, 150), (255, 190, 70)]
    row_h = (y1 - y0 - 18) // SENSOR_COUNT
    for sensor_id in range(SENSOR_COUNT):
        row_y = y0 + 18 + sensor_id * row_h
        color = COLORS[sensor_id]
        mag = float(np.linalg.norm(cur[sensor_id]))
        _draw_text(panel, f"finger {sensor_id} |F|={mag:6.2f}", (x0 + 10, row_y), 0.45, color)
        for axis_id, axis in enumerate(AXES):
            value = float(cur[sensor_id, axis_id])
            y = row_y + 13 + axis_id * 13
            _draw_text(panel, f"{axis}:{value:7.2f}", (x0 + 18, y), 0.35, axis_colors[axis_id])
            length = int(np.clip(value / abs_hi, -1, 1) * max_bar)
            cv2.line(panel, (zero_x, y - 3), (zero_x + length, y - 3), axis_colors[axis_id], 5, cv2.LINE_AA)
            cv2.line(panel, (zero_x, y - 8), (zero_x, y + 2), (150, 150, 150), 1)


def _render_episode(
    repo_id: Path,
    output_dir: Path,
    episode_index: int,
    camera: str,
    max_frames: int | None,
    y_limit: float | None,
    component_limit: float | None,
) -> dict[str, object]:
    names = _load_state_names(repo_id)
    force_indices = _calc_force_indices(names)
    forces, timestamps = _read_episode_forces(repo_id, episode_index, force_indices)
    magnitudes = np.linalg.norm(forces, axis=-1)
    video_path = _find_video(repo_id, camera, episode_index)
    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    video_fps = float(decoder.metadata.average_fps or 15)
    video_frames = len(decoder)
    total_frames = min(len(forces), video_frames)
    if max_frames:
        total_frames = min(total_frames, max_frames)

    out_path = output_dir / f"episode_{episode_index:06d}_{camera}_tactile_force.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 1280, 480
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {out_path}")

    for frame_idx in range(total_frames):
        frame_rgb = decoder[frame_idx].permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        panel = np.full((480, 640, 3), 22, dtype=np.uint8)
        _draw_text(panel, f"episode {episode_index:06d}  frame {frame_idx:04d}/{total_frames-1:04d}", (18, 28), 0.58)
        if frame_idx < len(timestamps):
            _draw_text(panel, f"t={timestamps[frame_idx]:.3f}s  camera={camera}", (18, 52), 0.5, (190, 190, 190))
        _draw_magnitude_plot(panel, magnitudes[:total_frames], frame_idx, (45, 90, 610, 265), y_limit)
        _draw_current_force_bars(panel, forces, frame_idx, (45, 315, 610, 462), component_limit)
        for sensor_id in range(SENSOR_COUNT):
            _draw_text(panel, f"finger {sensor_id}", (470, 292 + sensor_id * 16), 0.42, COLORS[sensor_id])
        writer.write(np.concatenate([frame, panel], axis=1))

    writer.release()

    stats = {
        "episode_index": episode_index,
        "output": str(out_path),
        "frames": int(total_frames),
        "camera": camera,
        "force_min": forces[:total_frames].min(axis=(0, 2)).round(4).tolist(),
        "force_max": forces[:total_frames].max(axis=(0, 2)).round(4).tolist(),
        "mag_mean": magnitudes[:total_frames].mean(axis=0).round(4).tolist(),
        "mag_max": magnitudes[:total_frames].max(axis=0).round(4).tolist(),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=Path, default=Path("grasp_pipette_and_press_button"))
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 16, 32, 48])
    parser.add_argument("--camera", default="observation.images.cam_front")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tactile_force_debug_videos"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--y-limit", type=float, default=None, help="Fixed |F| plot max. Auto if omitted.")
    parser.add_argument("--component-limit", type=float, default=None, help="Fixed fx/fy/fz bar abs max. Auto if omitted.")
    args = parser.parse_args()

    all_stats = []
    for episode_index in args.episodes:
        stats = _render_episode(
            args.repo_id,
            args.output_dir,
            episode_index,
            args.camera,
            args.max_frames,
            args.y_limit,
            args.component_limit,
        )
        all_stats.append(stats)
        print(json.dumps(stats, ensure_ascii=False))

    (args.output_dir / "summary.json").write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
