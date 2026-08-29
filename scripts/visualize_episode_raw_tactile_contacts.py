#!/usr/bin/env python3
"""Visualize per-finger raw tactile contacts for one XHand episode.

The script reads a local LeRobot v2.1 episode parquet file, extracts XHand raw
tactile force as [frames, 5, 120, 3], and saves 10x12 heatmaps for each finger.

Typical use:
  env/.venv/bin/python scripts/visualize_episode_raw_tactile_contacts.py \
    --repo-id data/task1_2_3_315ep \
    --episode-index 0 \
    --mode max \
    --output-dir tmp/raw_tactile_contacts
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


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
PATCH_NAMES = ("tip", "center", "base", "left", "right")


# Keep this table in sync with AdaptiveFingertipPatchTokenizer.  Patch IDs:
# 0=tip, 1=center, 2=base, 3=left, 4=right.
T30_THUMB_PATCH_IDS = np.asarray(
    (
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ),
    dtype=np.int64,
)

T16_OTHER_PATCH_IDS = np.asarray(
    (
        2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0,
        2, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0,
        2, 2, 2, 1, 1, 3, 3, 3, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        2, 2, 2, 1, 1, 4, 4, 4, 0, 0, 0, 0,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0,
        2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0,
    ),
    dtype=np.int64,
)


def _episode_file(repo: Path, episode_index: int) -> Path:
    candidates = sorted(repo.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find episode_{episode_index:06d}.parquet under {repo}")
    return candidates[0]


def _extract_raw_tactile(states: np.ndarray) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError(f"Expected states [T,D], got {states.shape}.")
    required_dim = (
        xhand_policy.TACTILE_BLOCK_START
        + xhand_policy.TACTILE_SENSOR_COUNT * xhand_policy.TACTILE_BLOCK_SIZE
    )
    if states.shape[1] < required_dim:
        raise ValueError(
            "Raw tactile extraction requires full XHand observation.state. "
            f"Need at least {required_dim} dims, got {states.shape[1]}."
        )

    chunks = []
    for sensor_id in range(xhand_policy.TACTILE_SENSOR_COUNT):
        start = (
            xhand_policy.TACTILE_BLOCK_START
            + sensor_id * xhand_policy.TACTILE_BLOCK_SIZE
            + xhand_policy.TACTILE_RAW_FORCE_OFFSET
        )
        end = start + xhand_policy.TACTILE_RAW_FORCE_POINTS * 3
        chunks.append(states[:, start:end].reshape(states.shape[0], xhand_policy.TACTILE_RAW_FORCE_POINTS, 3))
    return np.stack(chunks, axis=1).astype(np.float32)


def _load_episode(repo: Path, episode_index: int) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    episode_path = _episode_file(repo, episode_index)
    df = pd.read_parquet(episode_path, columns=["observation.state", "frame_index", "timestamp"])
    states = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
    frame_indices = np.asarray(df["frame_index"].to_list(), dtype=np.int64)
    timestamps = np.asarray(df["timestamp"].to_list(), dtype=np.float32)
    tactile = _extract_raw_tactile(states)
    return episode_path, tactile, frame_indices, timestamps


def _gate(magnitude: np.ndarray, threshold: float, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(magnitude - threshold) / max(temperature, 1e-6)))


def _select_frame(magnitude: np.ndarray, frame_indices: np.ndarray, requested_frame: int | None) -> int:
    if requested_frame is not None:
        matches = np.flatnonzero(frame_indices == requested_frame)
        if matches.size == 0:
            raise ValueError(f"frame_index={requested_frame} not found in episode.")
        return int(matches[0])
    return int(np.argmax(np.sum(magnitude, axis=(1, 2))))


def _tokenizer_selected_indices(values: np.ndarray, top_k: int, threshold: float) -> tuple[np.ndarray, str]:
    hard_contact = values > threshold
    contact_count = int(np.sum(hard_contact))
    if top_k <= 0:
        selected = np.flatnonzero(hard_contact)
        return selected[np.argsort(values[selected])[::-1]], "threshold_only"
    if contact_count == 0:
        return np.array([], dtype=np.int64), "no_contact"
    if contact_count < top_k:
        selected = np.flatnonzero(hard_contact)
        return selected[np.argsort(values[selected])[::-1]], "all_above_threshold"
    selected = np.argpartition(values, -top_k)[-top_k:]
    return selected[np.argsort(values[selected])[::-1]], "topk"


def _patch_ids_for_fingers() -> np.ndarray:
    return np.stack([T30_THUMB_PATCH_IDS, *[T16_OTHER_PATCH_IDS] * 4], axis=0)


def _plot_patch_overlay(ax: plt.Axes, finger: int) -> None:
    patch_ids = _patch_ids_for_fingers()[finger].reshape(10, 12)
    colors = ("cyan", "white", "lime", "deepskyblue", "magenta")
    for patch_id, color in enumerate(colors):
        rows, cols = np.nonzero(patch_ids == patch_id)
        ax.scatter(cols, rows, s=13, marker="s", facecolors="none", edgecolors=color, linewidths=0.45, alpha=0.75)


def _plot_five_finger_heatmap(
    *,
    heatmaps: np.ndarray,
    selected_indices: list[np.ndarray],
    titles: list[str],
    output_png: Path,
    suptitle: str,
    cmap: str,
    vmax: float | None,
    show_patch_overlay: bool,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.4), constrained_layout=True)
    if vmax is None:
        positive = heatmaps[heatmaps > 0]
        vmax = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    vmax = max(vmax, 1e-6)
    image = None
    for finger, ax in enumerate(axes):
        image = ax.imshow(heatmaps[finger].reshape(10, 12), cmap=cmap, vmin=0.0, vmax=vmax)
        selected = selected_indices[finger]
        if selected.size:
            rows = selected // 12
            cols = selected % 12
            ax.scatter(cols, rows, s=85, facecolors="none", edgecolors="cyan", linewidths=1.8)
        if show_patch_overlay:
            _plot_patch_overlay(ax, finger)
        finger_name = FINGER_NAMES[finger] if finger < len(FINGER_NAMES) else f"finger{finger}"
        ax.set_title(f"{finger}: {finger_name}\n{titles[finger]}", fontsize=9)
        ax.set_xticks(range(12))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=6)
        ax.set_xlabel("taxel col")
        if finger == 0:
            ax.set_ylabel("taxel row")
    fig.colorbar(image, ax=axes, shrink=0.85, location="right")
    fig.suptitle(suptitle)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def _plot_episode_traces(
    magnitude: np.ndarray,
    timestamps: np.ndarray,
    output_png: Path,
    threshold: float,
) -> None:
    time_axis = timestamps - timestamps[0] if timestamps.size else np.arange(magnitude.shape[0])
    per_finger_sum = magnitude.sum(axis=-1)
    per_finger_max = magnitude.max(axis=-1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
        name = FINGER_NAMES[finger] if finger < len(FINGER_NAMES) else f"finger{finger}"
        axes[0].plot(time_axis, per_finger_sum[:, finger], label=name, linewidth=1.4)
        axes[1].plot(time_axis, per_finger_max[:, finger], label=name, linewidth=1.4)
    axes[0].set_title("Per-finger raw tactile total magnitude over episode")
    axes[0].set_ylabel("sum |force| over 120 taxels")
    axes[1].set_title("Per-finger strongest taxel magnitude over episode")
    axes[1].set_ylabel("max |force|")
    axes[1].set_xlabel("time (s)")
    axes[1].axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.55, label="threshold")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(ncols=5, fontsize=8)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _plot_active_points_curve(
    magnitude: np.ndarray,
    timestamps: np.ndarray,
    output_png: Path,
    threshold: float,
) -> None:
    time_axis = timestamps - timestamps[0] if timestamps.size else np.arange(magnitude.shape[0])
    active_counts = np.sum(magnitude > threshold, axis=-1)

    fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
        name = FINGER_NAMES[finger] if finger < len(FINGER_NAMES) else f"finger {finger}"
        ax.plot(time_axis, active_counts[:, finger], label=f"finger {finger}", linewidth=1.5)

    ax.set_title(f"Active raw tactile points per finger, threshold |F| > {threshold:g}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("active points out of 120")
    ax.set_ylim(0, xhand_policy.TACTILE_RAW_FORCE_POINTS)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=xhand_policy.TACTILE_SENSOR_COUNT, fontsize=8)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _build_summary(
    *,
    repo: Path,
    episode_path: Path,
    episode_index: int,
    mode: str,
    frame_index: int | None,
    frame_position: int | None,
    timestamps: np.ndarray,
    magnitude: np.ndarray,
    gate: np.ndarray,
    heatmaps: np.ndarray,
    selected_indices: list[np.ndarray],
    selection_modes: list[str],
    threshold: float,
    temperature: float,
    top_k: int,
    outputs: dict[str, str],
) -> dict[str, object]:
    fingers = []
    for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
        values = heatmaps[finger].reshape(-1)
        selected = selected_indices[finger]
        whole_episode_mag = magnitude[:, finger, :]
        fingers.append(
            {
                "finger": finger,
                "name": FINGER_NAMES[finger] if finger < len(FINGER_NAMES) else f"finger{finger}",
                "episode_positive_ratio": float(np.mean(whole_episode_mag > threshold)),
                "episode_max_magnitude": float(np.max(whole_episode_mag)),
                "episode_mean_magnitude": float(np.mean(whole_episode_mag)),
                "heatmap_max": float(np.max(values)),
                "heatmap_mean": float(np.mean(values)),
                "gate_mean": float(np.mean(gate[:, finger, :])),
                "selected_count": int(selected.size),
                "selection_mode": selection_modes[finger],
                "selected_taxel_indices": selected.astype(int).tolist(),
                "selected_taxel_values": values[selected].astype(float).tolist() if selected.size else [],
            }
        )
    return {
        "repo_id": str(repo),
        "episode_file": str(episode_path),
        "episode_index": episode_index,
        "mode": mode,
        "frame_index": frame_index,
        "frame_position": frame_position,
        "timestamp": None if frame_position is None else float(timestamps[frame_position]),
        "num_frames": int(magnitude.shape[0]),
        "threshold": threshold,
        "temperature": temperature,
        "top_k": top_k,
        "fingers": fingers,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, type=Path, help="Local LeRobot dataset path.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index, e.g. 0.")
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Frame index for --mode frame. If omitted, choose the frame with largest tactile magnitude.",
    )
    parser.add_argument(
        "--mode",
        choices=("frame", "max", "mean", "contact_count", "gate_mean"),
        default="max",
        help=(
            "frame: one frame; max: max magnitude per taxel over episode; "
            "mean: mean magnitude; contact_count: number of frames above threshold; "
            "gate_mean: mean contact gate."
        ),
    )
    parser.add_argument("--threshold", type=float, default=1.0, help="Contact threshold in raw force space.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Contact gate temperature.")
    parser.add_argument("--top-k", type=int, default=16, help="Tokenizer-style selected taxels per finger.")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/episode_raw_tactile_contacts"))
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--vmax", type=float, default=None, help="Optional heatmap max value.")
    parser.add_argument("--show-patch-overlay", action="store_true", help="Overlay adaptive patch regions.")
    parser.add_argument("--no-traces", action="store_true", help="Do not save per-finger time traces.")
    parser.add_argument(
        "--no-active-curve",
        action="store_true",
        help="Do not save active-point count curves.",
    )
    args = parser.parse_args()

    episode_path, tactile, frame_indices, timestamps = _load_episode(args.repo_id, args.episode_index)
    magnitude = np.linalg.norm(tactile, axis=-1)
    gate = _gate(magnitude, args.threshold, args.temperature)

    frame_position: int | None = None
    if args.mode == "frame":
        frame_position = _select_frame(magnitude, frame_indices, args.frame_index)
        frame_index = int(frame_indices[frame_position])
        heatmaps = magnitude[frame_position]
        mode_title = f"frame_index={frame_index}, timestamp={timestamps[frame_position]:.3f}s"
    elif args.mode == "max":
        frame_index = None
        heatmaps = magnitude.max(axis=0)
        mode_title = "per-taxel max magnitude over episode"
    elif args.mode == "mean":
        frame_index = None
        heatmaps = magnitude.mean(axis=0)
        mode_title = "per-taxel mean magnitude over episode"
    elif args.mode == "contact_count":
        frame_index = None
        heatmaps = np.sum(magnitude > args.threshold, axis=0).astype(np.float32)
        mode_title = f"per-taxel contact count over episode, threshold={args.threshold:g}"
    elif args.mode == "gate_mean":
        frame_index = None
        heatmaps = gate.mean(axis=0)
        mode_title = f"per-taxel mean gate over episode, threshold={args.threshold:g}, temp={args.temperature:g}"
    else:
        raise AssertionError(args.mode)

    selected_indices = []
    selection_modes = []
    for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
        selected, selection_mode = _tokenizer_selected_indices(heatmaps[finger].reshape(-1), args.top_k, args.threshold)
        selected_indices.append(selected)
        selection_modes.append(selection_mode)

    stem_parts = [args.repo_id.name, f"ep{args.episode_index:06d}", args.mode]
    if frame_index is not None:
        stem_parts.append(f"frame{frame_index:06d}")
    stem = "_".join(stem_parts)
    heatmap_png = args.output_dir / f"{stem}_contacts.png"
    traces_png = args.output_dir / f"{stem}_traces.png"
    active_points_png = args.output_dir / f"{stem}_active_points.png"
    summary_json = args.output_dir / f"{stem}_summary.json"

    titles = []
    for finger in range(xhand_policy.TACTILE_SENSOR_COUNT):
        vals = heatmaps[finger].reshape(-1)
        titles.append(
            f"max={vals.max():.2f}, active={(vals > args.threshold).sum()}/120, selected={len(selected_indices[finger])}"
        )
    _plot_five_finger_heatmap(
        heatmaps=heatmaps,
        selected_indices=selected_indices,
        titles=titles,
        output_png=heatmap_png,
        suptitle=(
            f"{args.repo_id.name} episode_{args.episode_index:06d}: {mode_title}\n"
            "cyan circles = tokenizer-style selected contact taxels"
        ),
        cmap=args.cmap,
        vmax=args.vmax,
        show_patch_overlay=args.show_patch_overlay,
    )

    outputs = {"heatmap_png": str(heatmap_png)}
    if not args.no_traces:
        _plot_episode_traces(magnitude, timestamps, traces_png, args.threshold)
        outputs["traces_png"] = str(traces_png)
    if not args.no_active_curve:
        _plot_active_points_curve(magnitude, timestamps, active_points_png, args.threshold)
        outputs["active_points_png"] = str(active_points_png)

    summary = _build_summary(
        repo=args.repo_id,
        episode_path=episode_path,
        episode_index=args.episode_index,
        mode=args.mode,
        frame_index=frame_index,
        frame_position=frame_position,
        timestamps=timestamps,
        magnitude=magnitude,
        gate=gate,
        heatmaps=heatmaps,
        selected_indices=selected_indices,
        selection_modes=selection_modes,
        threshold=args.threshold,
        temperature=args.temperature,
        top_k=args.top_k,
        outputs=outputs,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved contact heatmap: {heatmap_png}")
    if not args.no_traces:
        print(f"Saved tactile traces:  {traces_png}")
    if not args.no_active_curve:
        print(f"Saved active curve:    {active_points_png}")
    print(f"Saved summary:         {summary_json}")


if __name__ == "__main__":
    main()
