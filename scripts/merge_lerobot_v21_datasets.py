#!/usr/bin/env python3
"""


python scripts/merge_lerobot_v21_datasets.py \
  --sources \
    /workspace/mnt/sqzhang26/FactileLDM/data/47ep \
    /workspace/mnt/sqzhang26/FactileLDM/data/grasp_pipette_and_press_button_0616_59ep \
  --output /workspace/mnt/sqzhang26/FactileLDM/data/grasp_pipette_and_press_button_106ep \
  --overwrite


"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CORE_META_FILES = {"info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=4)
        file.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalized_features(info: dict[str, Any]) -> dict[str, Any]:
    return info.get("features", {})


def validate_sources(sources: list[Path]) -> list[dict[str, Any]]:
    infos = []
    for source in sources:
        info_path = source / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
        info = read_json(info_path)
        if not str(info.get("codebase_version", "")).startswith("v2.1"):
            raise ValueError(f"{source} is not LeRobot v2.1: {info.get('codebase_version')}")
        infos.append(info)

    reference = infos[0]
    for source, info in zip(sources[1:], infos[1:], strict=True):
        for key in ("fps", "robot_type"):
            if info.get(key) != reference.get(key):
                raise ValueError(
                    f"Incompatible {key}: {sources[0]}={reference.get(key)!r}, "
                    f"{source}={info.get(key)!r}"
                )
        if normalized_features(info) != normalized_features(reference):
            raise ValueError(f"Incompatible feature schema between {sources[0]} and {source}")
    return infos


def episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = int(info["chunks_size"])
    return root / str(info["data_path"]).format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def output_episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    return episode_path(root, info, episode_index)


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    position = table.schema.get_field_index(name)
    if position < 0:
        raise KeyError(f"Required parquet column is missing: {name}")
    field = table.schema.field(position)
    return table.set_column(position, field, pa.array(values, type=field.type))


def scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    return {
        "min": [values.min().item()],
        "max": [values.max().item()],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(len(values))],
    }


def update_index_stats(
    source_stats: dict[str, Any] | None,
    *,
    episode_indices: np.ndarray,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
) -> dict[str, Any] | None:
    if source_stats is None:
        return None
    output = copy.deepcopy(source_stats)
    output["episode_index"] = scalar_stats(episode_indices)
    output["index"] = scalar_stats(global_indices)
    output["task_index"] = scalar_stats(task_indices)
    return output


def copy_video_streams(
    source: Path,
    output: Path,
    *,
    old_episode_index: int,
    new_episode_index: int,
    output_chunk_size: int,
) -> int:
    videos_root = source / "videos"
    if not videos_root.exists():
        return 0

    matches = list(videos_root.glob(f"chunk-*/*/episode_{old_episode_index:06d}.mp4"))
    for source_video in matches:
        stream_name = source_video.parent.name
        destination = (
            output
            / "videos"
            / f"chunk-{new_episode_index // output_chunk_size:03d}"
            / stream_name
            / f"episode_{new_episode_index:06d}.mp4"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_video, destination)
    return len(matches)


def copy_ancillary_files(first_source: Path, output: Path) -> None:
    for path in first_source.iterdir():
        if path.name in {"data", "videos", "meta"}:
            continue
        destination = output / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)

    source_meta = first_source / "meta"
    for path in source_meta.iterdir():
        if path.name in CORE_META_FILES:
            continue
        destination = output / "meta" / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True, help="Input LeRobot dataset roots.")
    parser.add_argument("--output", type=Path, required=True, help="Output LeRobot dataset root.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output directory.")
    args = parser.parse_args()

    sources = [path.expanduser().resolve() for path in args.sources]
    output = args.output.expanduser().resolve()
    if output in sources:
        raise ValueError("Output must be different from every input dataset.")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)

    infos = validate_sources(sources)
    output.mkdir(parents=True)
    copy_ancillary_files(sources[0], output)

    output_info = copy.deepcopy(infos[0])
    output_chunk_size = int(output_info["chunks_size"])

    task_to_index: dict[str, int] = {}
    output_tasks: list[dict[str, Any]] = []
    output_episodes: list[dict[str, Any]] = []
    output_episode_stats: list[dict[str, Any]] = []
    next_episode_index = 0
    next_global_index = 0
    total_videos = 0

    for source, info in zip(sources, infos, strict=True):
        source_tasks = {
            int(row["task_index"]): str(row["task"])
            for row in read_jsonl(source / "meta" / "tasks.jsonl")
        }
        source_episodes = {
            int(row["episode_index"]): row
            for row in read_jsonl(source / "meta" / "episodes.jsonl")
        }
        source_stats = {
            int(row["episode_index"]): row.get("stats")
            for row in read_jsonl(source / "meta" / "episodes_stats.jsonl")
        }

        expected_episodes = int(info["total_episodes"])
        for old_episode_index in range(expected_episodes):
            source_parquet = episode_path(source, info, old_episode_index)
            if not source_parquet.exists():
                raise FileNotFoundError(f"Missing episode parquet: {source_parquet}")

            table = pq.read_table(source_parquet)
            length = table.num_rows
            old_task_indices = table["task_index"].to_numpy(zero_copy_only=False)
            new_task_indices = np.empty(length, dtype=np.int64)
            for old_task_index in np.unique(old_task_indices):
                task = source_tasks[int(old_task_index)]
                if task not in task_to_index:
                    new_index = len(task_to_index)
                    task_to_index[task] = new_index
                    output_tasks.append({"task_index": new_index, "task": task})
                new_task_indices[old_task_indices == old_task_index] = task_to_index[task]

            episode_indices = np.full(length, next_episode_index, dtype=np.int64)
            global_indices = np.arange(next_global_index, next_global_index + length, dtype=np.int64)
            table = replace_column(table, "episode_index", episode_indices)
            table = replace_column(table, "index", global_indices)
            table = replace_column(table, "task_index", new_task_indices)

            destination_parquet = output_episode_path(output, output_info, next_episode_index)
            destination_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination_parquet)

            episode_meta = copy.deepcopy(source_episodes.get(old_episode_index, {}))
            episode_meta["episode_index"] = next_episode_index
            episode_meta["length"] = length
            episode_meta["tasks"] = [
                source_tasks[int(index)] for index in dict.fromkeys(old_task_indices.tolist())
            ]
            output_episodes.append(episode_meta)

            updated_stats = update_index_stats(
                source_stats.get(old_episode_index),
                episode_indices=episode_indices,
                global_indices=global_indices,
                task_indices=new_task_indices,
            )
            if updated_stats is not None:
                output_episode_stats.append(
                    {"episode_index": next_episode_index, "stats": updated_stats}
                )

            total_videos += copy_video_streams(
                source,
                output,
                old_episode_index=old_episode_index,
                new_episode_index=next_episode_index,
                output_chunk_size=output_chunk_size,
            )
            print(
                f"{source.name}: episode {old_episode_index} -> {next_episode_index} "
                f"({length} frames)"
            )
            next_episode_index += 1
            next_global_index += length

    output_info["total_episodes"] = next_episode_index
    output_info["total_frames"] = next_global_index
    output_info["total_tasks"] = len(output_tasks)
    output_info["total_videos"] = total_videos
    output_info["total_chunks"] = math.ceil(next_episode_index / output_chunk_size)
    output_info["splits"] = {"train": f"0:{next_episode_index}"}

    write_json(output / "meta" / "info.json", output_info)
    write_jsonl(output / "meta" / "tasks.jsonl", output_tasks)
    write_jsonl(output / "meta" / "episodes.jsonl", output_episodes)
    if output_episode_stats:
        write_jsonl(output / "meta" / "episodes_stats.jsonl", output_episode_stats)

    print(
        f"Merged {len(sources)} datasets into {output}: "
        f"{next_episode_index} episodes, {next_global_index} frames, "
        f"{len(output_tasks)} tasks, {total_videos} videos."
    )


if __name__ == "__main__":
    main()
