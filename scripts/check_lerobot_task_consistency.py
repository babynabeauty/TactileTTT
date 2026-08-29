#!/usr/bin/env python3
"""Check LeRobot task metadata against per-episode parquet task_index values.

env/.venv/bin/python scripts/check_lerobot_task_consistency.py \
  --dataset data/task12345-2 \
  --episodes-jsonl data/task12345-2/meta/episodes.jsonl \
  --expect-range 0:30:0 \
  --expect-range 31:139:1 \
  --expect-range 140:245:2 \
  --expect-range 246:345:0 \
  --expect-range 346:455:3 \
  --expect-range 456:558:4

"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict
from typing import Any

import pyarrow.parquet as pq


def _read_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _episode_chunk(episode_index: int, chunks_size: int) -> int:
    return episode_index // chunks_size


def _episode_path(dataset: pathlib.Path, info: dict[str, Any], episode_index: int) -> pathlib.Path:
    chunks_size = int(info.get("chunks_size", 1000))
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    rel = template.format(
        episode_chunk=_episode_chunk(episode_index, chunks_size),
        episode_index=episode_index,
    )
    return dataset / rel


def _parse_expect_range(spec: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+):(\d+):(\d+)", spec)
    if not match:
        raise argparse.ArgumentTypeError("Expected format START:END:TASK_INDEX, e.g. 0:30:0")
    start, end, task_index = map(int, match.groups())
    if end < start:
        raise argparse.ArgumentTypeError("END must be >= START")
    return start, end, task_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="LeRobot dataset root, e.g. data/task12345-2")
    parser.add_argument(
        "--episodes-jsonl",
        default=None,
        help="Optional replacement episodes.jsonl to compare against, e.g. tmp/episodes-task5.jsonl",
    )
    parser.add_argument(
        "--expect-range",
        action="append",
        type=_parse_expect_range,
        default=[],
        help="Assert episode range has one task_index. Format START:END:TASK_INDEX. Can repeat.",
    )
    parser.add_argument("--max-errors", type=int, default=30)
    args = parser.parse_args()

    dataset = pathlib.Path(args.dataset)
    meta = dataset / "meta"
    info_path = meta / "info.json"
    tasks_path = meta / "tasks.jsonl"
    episodes_path = pathlib.Path(args.episodes_jsonl) if args.episodes_jsonl else meta / "episodes.jsonl"

    for path in (info_path, tasks_path, episodes_path):
        if not path.exists():
            print(f"ERROR: missing file: {path}", file=sys.stderr)
            return 2

    info = _read_json(info_path)
    tasks_rows = _read_jsonl(tasks_path)
    episode_rows = _read_jsonl(episodes_path)
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks_rows}
    episode_by_index = {int(row["episode_index"]): row for row in episode_rows}

    errors: list[str] = []
    parquet_task_counter: Counter[int] = Counter()
    episode_task_counter: Counter[str] = Counter()
    range_actual: dict[tuple[int, int, int], Counter[int]] = {r: Counter() for r in args.expect_range}

    for episode_index in sorted(episode_by_index):
        episode_meta = episode_by_index[episode_index]
        parquet_path = _episode_path(dataset, info, episode_index)
        if not parquet_path.exists():
            errors.append(f"episode {episode_index}: missing parquet {parquet_path}")
            continue

        table = pq.read_table(parquet_path, columns=["task_index"])
        values = table.column("task_index").to_numpy(zero_copy_only=False)
        unique_task_indices = sorted({int(v) for v in values.tolist()})
        if len(unique_task_indices) != 1:
            errors.append(f"episode {episode_index}: parquet has multiple task_index values {unique_task_indices}")
            continue

        task_index = unique_task_indices[0]
        parquet_task_counter[task_index] += 1
        expected_task = task_by_index.get(task_index)
        if expected_task is None:
            errors.append(f"episode {episode_index}: parquet task_index={task_index} not found in tasks.jsonl")
            continue

        episode_tasks = [str(t) for t in episode_meta.get("tasks", [])]
        if expected_task not in episode_tasks:
            errors.append(
                f"episode {episode_index}: parquet task_index={task_index} -> {expected_task!r}, "
                f"but episodes jsonl has tasks={episode_tasks!r}"
            )
        if "length" in episode_meta and int(episode_meta["length"]) != len(values):
            errors.append(
                f"episode {episode_index}: episodes length={episode_meta['length']} but parquet rows={len(values)}"
            )
        episode_task_counter[expected_task] += 1

        for range_spec in args.expect_range:
            start, end, _expected_index = range_spec
            if start <= episode_index <= end:
                range_actual[range_spec][task_index] += 1

    info_total_episodes = int(info.get("total_episodes", -1))
    if info_total_episodes != len(episode_rows):
        errors.append(f"info total_episodes={info_total_episodes} but episodes jsonl rows={len(episode_rows)}")
    info_total_tasks = int(info.get("total_tasks", -1))
    if info_total_tasks != len(tasks_rows):
        errors.append(f"info total_tasks={info_total_tasks} but tasks jsonl rows={len(tasks_rows)}")

    for start, end, expected_index in args.expect_range:
        actual = range_actual[(start, end, expected_index)]
        if actual != Counter({expected_index: end - start + 1}):
            errors.append(
                f"expect-range {start}:{end}:{expected_index} failed; parquet task_index counts={dict(actual)}"
            )

    print(f"dataset: {dataset}")
    print(f"episodes source: {episodes_path}")
    print(f"tasks: {len(tasks_rows)}")
    for task_index in sorted(task_by_index):
        print(f"  task_index={task_index}: {task_by_index[task_index]}")
    print("parquet episode counts by task_index:")
    for task_index in sorted(parquet_task_counter):
        print(f"  {task_index}: {parquet_task_counter[task_index]} episodes")
    print("episode counts by task string:")
    for task, count in episode_task_counter.items():
        print(f"  {count}: {task}")

    if errors:
        print(f"\nFAILED with {len(errors)} issue(s):")
        for error in errors[: args.max_errors]:
            print(f"  - {error}")
        if len(errors) > args.max_errors:
            print(f"  ... {len(errors) - args.max_errors} more")
        return 1

    print("\nOK: tasks.jsonl, episodes jsonl, and parquet task_index are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
