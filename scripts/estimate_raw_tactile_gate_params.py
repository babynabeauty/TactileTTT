#!/usr/bin/env python3
"""Estimate raw tactile contact gate parameters from local LeRobot data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from openpi.policies import xhand_policy


def _episode_files(repo: Path, max_episodes: int | None) -> list[Path]:
    files = sorted(repo.glob("data/**/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {repo}")
    if max_episodes is not None:
        files = files[:max_episodes]
    return files


def _extract_raw_tactile_magnitude(state: np.ndarray) -> np.ndarray:
    chunks = []
    for sensor_id in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + sensor_id * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        chunks.append(state[start:end].reshape(xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    tactile = np.stack(chunks, axis=0).astype(np.float32)
    return np.linalg.norm(tactile, axis=-1)


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p}.")
    return math.log(p / (1.0 - p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Local LeRobot dataset path, e.g. data/task1_2_206ep")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="If omitted, use the minimum non-zero magnitude.")
    parser.add_argument("--zero-gate", type=float, default=0.1, help="Desired gate value for zero force.")
    parser.add_argument("--positive-gate", type=float, default=0.8, help="Desired gate value for low positive force.")
    parser.add_argument(
        "--positive-percentile",
        type=float,
        default=5.0,
        help="Positive magnitude percentile used as a low-contact anchor.",
    )
    args = parser.parse_args()

    repo = Path(args.repo_id)
    files = _episode_files(repo, args.max_episodes)

    magnitudes = []
    per_finger = [[] for _ in range(xhand_policy.TACTILE_SENSOR_COUNT)]
    for episode_file in files:
        df = pd.read_parquet(episode_file, columns=["observation.state"])
        if args.max_frames_per_episode is not None:
            df = df.iloc[: args.max_frames_per_episode]
        for state in df["observation.state"]:
            mag = _extract_raw_tactile_magnitude(np.asarray(state, dtype=np.float32))
            magnitudes.append(mag.reshape(-1))
            for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
                per_finger[finger].append(mag[finger].reshape(-1))

    all_mag = np.concatenate(magnitudes)
    positive = all_mag[all_mag > 0]
    if positive.size == 0:
        raise ValueError("No positive raw tactile magnitudes found.")

    threshold = float(args.threshold) if args.threshold is not None else float(positive.min())
    low_positive = float(np.percentile(positive, args.positive_percentile))
    tau_zero = (0.0 - threshold) / _logit(args.zero_gate)
    tau_positive = (low_positive - threshold) / _logit(args.positive_gate)
    recommended_temperature = float(np.median([tau_zero, tau_positive]))

    print(f"repo_id: {repo}")
    print(f"episodes: {len(files)}")
    print(f"total_taxel_values: {all_mag.size}")
    print(f"positive_ratio: {(all_mag > 0).mean():.6f}")
    print()
    print("all magnitude percentiles:")
    for q in [0, 50, 75, 90, 95, 97, 99, 99.5, 99.9, 100]:
        print(f"  p{q:>5}: {np.percentile(all_mag, q):.4f}")
    print()
    print("positive magnitude percentiles:")
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{q:>5}: {np.percentile(positive, q):.4f}")
    print()
    print("per-finger positive stats:")
    for finger, chunks in enumerate(per_finger):
        values = np.concatenate(chunks)
        pos = values[values > 0]
        if pos.size == 0:
            print(f"  finger {finger}: positive_ratio=0.000000")
            continue
        qs = np.percentile(pos, [5, 50, 95])
        print(
            f"  finger {finger}: positive_ratio={(values > 0).mean():.6f}, "
            f"p5={qs[0]:.4f}, p50={qs[1]:.4f}, p95={qs[2]:.4f}"
        )
    print()
    print("recommended gate params:")
    print(f"  tactile_raw_contact_threshold={threshold:.4f}")
    print(f"  tactile_raw_contact_temperature={recommended_temperature:.4f}")
    print()
    print("anchors:")
    print(f"  gate(0 force) ~= {args.zero_gate:.3f} -> temperature {tau_zero:.4f}")
    print(
        f"  gate(p{args.positive_percentile:g} positive={low_positive:.4f}) "
        f"~= {args.positive_gate:.3f} -> temperature {tau_positive:.4f}"
    )


if __name__ == "__main__":
    main()
