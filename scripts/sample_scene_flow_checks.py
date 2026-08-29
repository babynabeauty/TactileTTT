"""
检验生成的npz文件
cd /data/workspace/zhangshiqi/forceWAM

env/.venv/bin/python scripts/sample_scene_flow_checks.py \
  --flow-root /data/workspace/zhangshiqi/forceWAM/outputs/front_scene_flow_grasp_pipette_sam3_tracked_npz \
  --num-samples 12 \
  --seed 42 \
  --output-dir /data/workspace/zhangshiqi/forceWAM/outputs/scene_flow_random_checks_seed42 \
  --max-lines 2500
""""
from __future__ import annotations

import argparse
import pathlib
import random
import re

import numpy as np

from visualize_front_scene_flow_episode import write_line_ply, write_point_ply


YELLOW = np.array([245, 191, 35], dtype=np.uint8)
CYAN = np.array([0, 210, 210], dtype=np.uint8)


def _magnitude_colors(mag: np.ndarray) -> np.ndarray:
    if mag.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(mag, [5, 95])
    if hi <= lo:
        scaled = np.zeros_like(mag)
    else:
        scaled = np.clip((mag - lo) / (hi - lo), 0.0, 1.0)
    colors = np.zeros((len(mag), 3), dtype=np.uint8)
    colors[:, 0] = (255 * scaled).astype(np.uint8)
    colors[:, 1] = (255 * (1.0 - np.abs(scaled - 0.5) * 2.0)).astype(np.uint8)
    colors[:, 2] = (255 * (1.0 - scaled)).astype(np.uint8)
    return colors


def _pair_name(path: pathlib.Path) -> str:
    episode = next((part for part in path.parts if re.fullmatch(r"episode_\d{6}", part)), "episode_unknown")
    return f"{episode}_{path.parent.name}"


def _summarize_class(name: str, mag: np.ndarray) -> str:
    if len(mag) == 0:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(mag)} "
        f"mean={mag.mean():.4f}m median={np.median(mag):.4f}m "
        f"p90={np.percentile(mag, 90):.4f}m max={mag.max():.4f}m"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow-root",
        type=pathlib.Path,
        default=pathlib.Path("outputs/front_scene_flow_grasp_pipette_sam3_tracked_npz"),
    )
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("outputs/scene_flow_random_checks"))
    parser.add_argument("--max-lines", type=int, default=3000)
    args = parser.parse_args()

    npz_paths = sorted(args.flow_root.glob("episode_*/pairs/frame_*_to_*/nn_scene_flow_robot_object.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No scene-flow npz files found under {args.flow_root}")

    rng = random.Random(args.seed)
    samples = rng.sample(npz_paths, min(args.num_samples, len(npz_paths)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        "sample,path,source_frame,target_frame,total,robot_n,object_n,"
        "mean_m,median_m,p90_m,max_m,robot_mean_m,object_mean_m"
    ]
    for path in samples:
        data = np.load(path)
        xyz = data["xyz"].astype(np.float32)
        target_xyz = data["target_xyz"].astype(np.float32)
        flow = data["flow_xyz"].astype(np.float32)
        class_id = data["class_id"].astype(np.int8)
        source_frame = int(data["source_frame"])
        target_frame = int(data["target_frame"])

        mag = np.linalg.norm(flow, axis=-1)
        robot_mag = mag[class_id == 1]
        object_mag = mag[class_id == 2]

        order = np.arange(len(xyz))
        if len(order) > args.max_lines:
            order = rng.sample(list(order), args.max_lines)
            order = np.asarray(order, dtype=np.int64)

        sample_dir = args.output_dir / _pair_name(path)
        sample_dir.mkdir(parents=True, exist_ok=True)
        line_colors = data["color"].astype(np.uint8)[order] if "color" in data.files else _magnitude_colors(mag[order])
        write_line_ply(sample_dir / "flow_lines_robot_yellow_object_cyan.ply", xyz[order], target_xyz[order], line_colors)
        write_point_ply(sample_dir / "source_points_colored_by_flow_magnitude.ply", xyz[order], _magnitude_colors(mag[order]))
        write_point_ply(sample_dir / "target_points_colored_by_flow_magnitude.ply", target_xyz[order], _magnitude_colors(mag[order]))

        summary = "\n".join(
            [
                f"path={path}",
                f"source_frame={source_frame}",
                f"target_frame={target_frame}",
                _summarize_class("all", mag),
                _summarize_class("robot", robot_mag),
                _summarize_class("object", object_mag),
                "colors: robot=yellow, object=cyan; magnitude: blue=small, red=large",
            ]
        )
        (sample_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
        print(f"\n[{sample_dir.name}]\n{summary}")

        rows.append(
            ",".join(
                [
                    sample_dir.name,
                    str(path),
                    str(source_frame),
                    str(target_frame),
                    str(len(mag)),
                    str(len(robot_mag)),
                    str(len(object_mag)),
                    f"{mag.mean() if len(mag) else 0.0:.6f}",
                    f"{np.median(mag) if len(mag) else 0.0:.6f}",
                    f"{np.percentile(mag, 90) if len(mag) else 0.0:.6f}",
                    f"{mag.max() if len(mag) else 0.0:.6f}",
                    f"{robot_mag.mean() if len(robot_mag) else 0.0:.6f}",
                    f"{object_mag.mean() if len(object_mag) else 0.0:.6f}",
                ]
            )
        )

    (args.output_dir / "random_check_summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nWrote {len(samples)} sampled checks to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
