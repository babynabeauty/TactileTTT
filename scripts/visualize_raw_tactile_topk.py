#!/usr/bin/env python3
"""Visualize raw XHand tactile top-k points used by the tokenizer.

The script reads local LeRobot v2.1 parquet episodes, extracts
observation.state, reconstructs raw tactile as [5, 120, 3], and saves a
per-finger 10x12 heatmap with the selected top-k taxels marked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openpi.policies import xhand_policy


def _episode_file(repo: Path, episode_index: int) -> Path:
    candidates = sorted(repo.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find episode_{episode_index:06d}.parquet under {repo}")
    return candidates[0]


def _extract_raw_tactile(state: np.ndarray) -> np.ndarray:
    chunks = []
    for sensor_id in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + sensor_id * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        chunks.append(state[start:end].reshape(xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    return np.stack(chunks, axis=0).astype(np.float32)


def _select_frame(df: pd.DataFrame, requested: int | None) -> tuple[int, np.ndarray]:
    if requested is not None:
        if requested < 0:
            requested = int(df["frame_index"].max()) + 1 + requested
        row = df[df["frame_index"] == requested]
        if row.empty:
            raise ValueError(f"frame_index={requested} not found in episode.")
        return requested, np.asarray(row.iloc[0]["observation.state"], dtype=np.float32)

    best_frame = None
    best_state = None
    best_score = -np.inf
    for _, row in df.iterrows():
        state = np.asarray(row["observation.state"], dtype=np.float32)
        tactile = _extract_raw_tactile(state)
        score = float(np.linalg.norm(tactile, axis=-1).sum())
        if score > best_score:
            best_score = score
            best_frame = int(row["frame_index"])
            best_state = state
    if best_frame is None or best_state is None:
        raise ValueError("Episode has no frames.")
    return best_frame, best_state


def _selected_indices(values: np.ndarray, top_k: int, threshold: float) -> tuple[np.ndarray, str]:
    hard_contact = values > threshold
    contact_count = int(np.sum(hard_contact))
    if top_k <= 0:
        return np.flatnonzero(hard_contact), "threshold_only"
    if contact_count == 0:
        return np.array([], dtype=np.int64), "no_contact_no_hard_topk"
    if contact_count < top_k:
        selected = np.flatnonzero(hard_contact)
        return selected[np.argsort(values[selected])[::-1]], "all_above_threshold"
    selected = np.argpartition(values, -top_k)[-top_k:]
    return selected[np.argsort(values[selected])[::-1]], "topk_above_threshold"


def _plot(tactile: np.ndarray, top_k: int, threshold: float, temperature: float, output_png: Path) -> dict:
    magnitude = np.linalg.norm(tactile, axis=-1)
    gate = 1.0 / (1.0 + np.exp(-(magnitude - threshold) / max(temperature, 1e-6)))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    summary = {
        "top_k": top_k,
        "threshold": threshold,
        "temperature": temperature,
        "fingers": [],
    }
    vmax = float(np.percentile(magnitude, 99.0)) if np.any(magnitude > 0) else 1.0
    vmax = max(vmax, 1e-6)
    for finger, ax in enumerate(axes):
        values = magnitude[finger]
        selected, selection_mode = _selected_indices(values, top_k, threshold)
        heat = values.reshape(10, 12)
        ax.imshow(heat, cmap="magma", vmin=0.0, vmax=vmax)
        if selected.size:
            rows = selected // 12
            cols = selected % 12
            ax.scatter(cols, rows, s=80, facecolors="none", edgecolors="cyan", linewidths=1.8)
        ax.set_title(f"finger {finger}")
        ax.set_xticks(range(12))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=6)

        summary["fingers"].append(
            {
                "finger": finger,
                "max_magnitude": float(values.max()),
                "mean_magnitude": float(values.mean()),
                "mean_gate": float(gate[finger].mean()),
                "num_points_magnitude_gt_threshold": int(np.sum(values > threshold)),
                "selection_mode": selection_mode,
                "top_indices": selected.astype(int).tolist(),
                "top_magnitudes": values[selected].astype(float).tolist(),
            }
        )
    fig.suptitle("Raw tactile magnitude heatmap; cyan = tokenizer top-k candidates")
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Local LeRobot dataset path, e.g. data/task1_2_206ep")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index to visualize. If omitted, picks the frame with largest raw tactile magnitude.",
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--output-dir", default="tmp/raw_tactile_topk")
    args = parser.parse_args()

    repo = Path(args.repo_id)
    episode_path = _episode_file(repo, args.episode_index)
    df = pd.read_parquet(episode_path)
    frame_index, state = _select_frame(df, args.frame_index)
    tactile = _extract_raw_tactile(state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{repo.name}_ep{args.episode_index:06d}_frame{frame_index:06d}_top{args.top_k}"
    output_png = output_dir / f"{stem}.png"
    output_json = output_dir / f"{stem}.json"

    summary = _plot(tactile, args.top_k, args.threshold, args.temperature, output_png)
    summary.update(
        {
            "repo_id": str(repo),
            "episode_index": args.episode_index,
            "frame_index": frame_index,
            "episode_file": str(episode_path),
            "output_png": str(output_png),
        }
    )
    output_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved heatmap: {output_png}")
    print(f"Saved summary: {output_json}")


if __name__ == "__main__":
    main()
