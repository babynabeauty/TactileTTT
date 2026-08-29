#!/usr/bin/env python3
"""Create task-stratified train/val episode splits for local LeRobot datasets.
划分测试集和val集
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _primary_task(row: dict) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    task = row.get("task")
    if task is not None:
        return str(task)
    raise KeyError(f"Episode row has no task/tasks field: {row}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--val-ratio", default=0.10, type=float)
    parser.add_argument("--min-val-per-task", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--use-tail",
        action="store_true",
        help="Use the last N episodes per task as val instead of random sampling.",
    )
    args = parser.parse_args()

    repo = args.repo_id
    episodes_path = repo / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Cannot find {episodes_path}")

    rows = _read_jsonl(episodes_path)
    by_task: dict[str, list[int]] = collections.defaultdict(list)
    lengths_by_episode = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        by_task[_primary_task(row)].append(episode_index)
        lengths_by_episode[episode_index] = int(row.get("length", 0))

    rng = random.Random(args.seed)
    train_episodes: list[int] = []
    val_episodes: list[int] = []
    summary = {
        "repo_id": str(repo),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "tasks": {},
    }

    for task, episodes in sorted(by_task.items()):
        episodes = sorted(episodes)
        val_count = max(args.min_val_per_task, round(len(episodes) * args.val_ratio))
        val_count = min(val_count, max(len(episodes) - 1, 1))
        if args.use_tail:
            task_val = episodes[-val_count:]
        else:
            task_val = sorted(rng.sample(episodes, val_count))
        task_train = [episode for episode in episodes if episode not in set(task_val)]
        train_episodes.extend(task_train)
        val_episodes.extend(task_val)
        summary["tasks"][task] = {
            "total_episodes": len(episodes),
            "train_episodes": len(task_train),
            "val_episodes": len(task_val),
            "val_ids": task_val,
            "val_frames": sum(lengths_by_episode[i] for i in task_val),
        }

    train_episodes = sorted(train_episodes)
    val_episodes = sorted(val_episodes)
    output_dir = args.output_dir or Path("outputs/episode_splits") / repo.name
    _write_json(output_dir / "train_episodes.json", {"episodes": train_episodes})
    _write_json(output_dir / "val_episodes.json", {"episodes": val_episodes})
    summary.update(
        {
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
            "train_ids": train_episodes,
            "val_ids": val_episodes,
        }
    )
    _write_json(output_dir / "summary.json", summary)

    print(f"Saved split to {output_dir}")
    print(f"train episodes: {len(train_episodes)}")
    print(f"val episodes:   {len(val_episodes)}")
    for task, info in summary["tasks"].items():
        print(f"- {task}: train={info['train_episodes']} val={info['val_episodes']} val_ids={info['val_ids']}")


if __name__ == "__main__":
    main()
