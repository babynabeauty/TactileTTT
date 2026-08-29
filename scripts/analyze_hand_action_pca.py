"""Analyze low-dimensional structure in XHand action trajectories.

This script reads local LeRobot v2.x parquet files directly, extracts the hand
action slice, optionally mixes in legacy zarr datasets with ``hand_action``
arrays, runs PCA with NumPy SVD, and writes plots/tables for choosing a
learnable hand-synergy bottleneck dimension.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def _episode_paths(repo_id: Path) -> list[Path]:
    paths = sorted((repo_id / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {repo_id / 'data'}")
    return paths


def _load_action_names(repo_id: Path, action_dim: int) -> list[str]:
    info_path = repo_id / "meta" / "info.json"
    if not info_path.exists():
        return [f"action_{i}" for i in range(action_dim)]
    info = json.loads(info_path.read_text())
    names = info.get("features", {}).get("action", {}).get("names")
    if names is None:
        return [f"action_{i}" for i in range(action_dim)]
    return list(names)


def _fixed_size_list_to_numpy(column) -> np.ndarray:
    values = column.combine_chunks().values.to_numpy(zero_copy_only=False)
    width = column.type.list_size
    return np.asarray(values, dtype=np.float32).reshape(-1, width)


def _load_actions(repo_id: Path, *, max_frames: int | None) -> np.ndarray:
    chunks = []
    total = 0
    for parquet_path in _episode_paths(repo_id):
        table = pq.read_table(parquet_path, columns=["action"])
        action = _fixed_size_list_to_numpy(table["action"])
        if max_frames is not None and total + len(action) > max_frames:
            action = action[: max_frames - total]
        chunks.append(action)
        total += len(action)
        if max_frames is not None and total >= max_frames:
            break
    return np.concatenate(chunks, axis=0)


def _zarr_dataset_dirs(root: Path) -> list[Path]:
    if (root / "episode_0").exists() or any(root.glob("episode_*")):
        return [root]
    return sorted([path for path in root.iterdir() if path.is_dir() and any(path.glob("episode_*"))])


def _zarr_episode_dirs(dataset_dir: Path) -> list[Path]:
    def _episode_index(path: Path) -> int:
        suffix = path.name.removeprefix("episode_")
        return int(suffix) if suffix.isdigit() else 10**9

    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.startswith("episode_")],
        key=_episode_index,
    )


def _load_zarr_hand_actions(
    root: Path,
    *,
    zarr_key: str,
    hand_dim: int,
    max_frames: int | None,
) -> list[tuple[str, np.ndarray]]:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Reading --zarr-root requires the zarr package in the active environment.") from exc

    outputs = []
    for dataset_dir in _zarr_dataset_dirs(root):
        chunks = []
        total = 0
        for episode_dir in _zarr_episode_dirs(dataset_dir):
            array_path = episode_dir / zarr_key
            if not array_path.exists():
                continue
            values = np.asarray(zarr.open_array(str(array_path), mode="r")[:], dtype=np.float32)
            if values.ndim != 2 or values.shape[-1] != hand_dim:
                raise ValueError(f"{array_path} has shape {values.shape}; expected [T,{hand_dim}].")
            if max_frames is not None and total + len(values) > max_frames:
                values = values[: max_frames - total]
            chunks.append(values)
            total += len(values)
            if max_frames is not None and total >= max_frames:
                break
        if chunks:
            outputs.append((str(dataset_dir), np.concatenate(chunks, axis=0)))
    if not outputs:
        raise FileNotFoundError(f"No zarr arrays named {zarr_key!r} found under {root}.")
    return outputs


def _pca(x: np.ndarray, *, standardize: bool) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    centered = x - mean
    scale = np.ones_like(mean)
    if standardize:
        scale = centered.std(axis=0) + 1e-8
        centered = centered / scale

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = (singular_values**2) / max(centered.shape[0] - 1, 1)
    explained_ratio = explained_variance / np.maximum(explained_variance.sum(), 1e-12)
    return {
        "mean": mean,
        "scale": scale,
        "components": vt,
        "explained_variance": explained_variance,
        "explained_ratio": explained_ratio,
        "cumulative_ratio": np.cumsum(explained_ratio),
    }


def _write_summary_csv(path: Path, explained_ratio: np.ndarray, cumulative_ratio: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "explained_ratio", "cumulative_ratio"])
        for i, (ratio, cumulative) in enumerate(zip(explained_ratio, cumulative_ratio, strict=True), start=1):
            writer.writerow([i, float(ratio), float(cumulative)])


def _write_source_counts(path: Path, sources: list[tuple[str, np.ndarray]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "num_frames", "min", "max", "mean", "std"])
        for name, values in sources:
            writer.writerow(
                [
                    name,
                    int(values.shape[0]),
                    json.dumps(values.min(axis=0).astype(float).tolist()),
                    json.dumps(values.max(axis=0).astype(float).tolist()),
                    json.dumps(values.mean(axis=0).astype(float).tolist()),
                    json.dumps(values.std(axis=0).astype(float).tolist()),
                ]
            )


def _plot_explained_variance(path: Path, explained_ratio: np.ndarray, cumulative_ratio: np.ndarray) -> None:
    xs = np.arange(1, len(explained_ratio) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xs, explained_ratio, alpha=0.65, label="individual")
    ax.plot(xs, cumulative_ratio, marker="o", color="tab:red", label="cumulative")
    for k in (4, 6, 8):
        if k <= len(cumulative_ratio):
            ax.axvline(k, linestyle="--", linewidth=1, color="gray", alpha=0.5)
            ax.text(k + 0.05, cumulative_ratio[k - 1], f"K={k}: {cumulative_ratio[k - 1]:.1%}")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_components(path: Path, components: np.ndarray, names: list[str], *, num_components: int) -> None:
    components = components[:num_components]
    vmax = float(np.max(np.abs(components))) if components.size else 1.0
    fig, ax = plt.subplots(figsize=(12, 0.65 * num_components + 3))
    im = ax.imshow(components, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(num_components))
    ax.set_yticklabels([f"PC{i + 1}" for i in range(num_components)])
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Hand action PCA components")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_component_bars(output_dir: Path, components: np.ndarray, names: list[str], *, num_components: int) -> None:
    for i in range(min(num_components, components.shape[0])):
        fig, ax = plt.subplots(figsize=(12, 4))
        values = components[i]
        colors = ["tab:red" if v >= 0 else "tab:blue" for v in values]
        ax.bar(np.arange(len(values)), values, color=colors, alpha=0.8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("loading")
        ax.set_title(f"PC{i + 1} hand joint pattern")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"pc_{i + 1:02d}_joint_pattern.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=Path,
        action="append",
        default=[],
        help="Local LeRobot dataset root. May be passed multiple times.",
    )
    parser.add_argument(
        "--zarr-root",
        type=Path,
        action="append",
        default=[],
        help="Legacy zarr root containing *_dataset/episode_*/hand_action. May be passed multiple times.",
    )
    parser.add_argument("--zarr-key", type=str, default="hand_action")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hand_action_pca"))
    parser.add_argument("--hand-start", type=int, default=6)
    parser.add_argument("--hand-dim", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames loaded per source.")
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Run PCA on per-joint standardized actions. Default uses centered raw action scale.",
    )
    parser.add_argument("--num-components-to-plot", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, np.ndarray]] = []
    hand_names: list[str] | None = None
    for repo_id_arg in args.repo_id:
        repo_id = repo_id_arg.expanduser().resolve()
        actions = _load_actions(repo_id, max_frames=args.max_frames)
        if args.hand_start < 0 or args.hand_dim <= 0 or args.hand_start + args.hand_dim > actions.shape[-1]:
            raise ValueError(
                f"Invalid hand slice [{args.hand_start}:{args.hand_start + args.hand_dim}] "
                f"for action_dim={actions.shape[-1]} in {repo_id}."
            )
        sources.append((str(repo_id), actions[:, args.hand_start : args.hand_start + args.hand_dim]))
        if hand_names is None:
            action_names = _load_action_names(repo_id, actions.shape[-1])
            hand_names = action_names[args.hand_start : args.hand_start + args.hand_dim]

    for zarr_root_arg in args.zarr_root:
        zarr_root = zarr_root_arg.expanduser().resolve()
        sources.extend(
            _load_zarr_hand_actions(
                zarr_root,
                zarr_key=args.zarr_key,
                hand_dim=args.hand_dim,
                max_frames=args.max_frames,
            )
        )

    if not sources:
        raise ValueError("Pass at least one --repo-id or --zarr-root.")

    hand_actions = np.concatenate([values for _, values in sources], axis=0)
    if hand_names is None:
        hand_names = [f"hand_joint_{i}.pos" for i in range(args.hand_dim)]

    result = _pca(hand_actions, standardize=args.standardize)
    summary = {
        "sources": [{"name": name, "num_frames": int(values.shape[0])} for name, values in sources],
        "num_frames": int(hand_actions.shape[0]),
        "hand_start": int(args.hand_start),
        "hand_dim": int(args.hand_dim),
        "zarr_key": args.zarr_key,
        "standardize": bool(args.standardize),
        "hand_names": hand_names,
        "explained_ratio": result["explained_ratio"].astype(float).tolist(),
        "cumulative_ratio": result["cumulative_ratio"].astype(float).tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_summary_csv(output_dir / "explained_variance.csv", result["explained_ratio"], result["cumulative_ratio"])
    _write_source_counts(output_dir / "source_counts.csv", sources)
    np.save(output_dir / "hand_action_mean.npy", result["mean"])
    np.save(output_dir / "hand_action_pca_components.npy", result["components"])

    _plot_explained_variance(
        output_dir / "explained_variance.png",
        result["explained_ratio"],
        result["cumulative_ratio"],
    )
    _plot_components(
        output_dir / "pca_components_heatmap.png",
        result["components"],
        hand_names,
        num_components=min(args.num_components_to_plot, args.hand_dim),
    )
    _plot_component_bars(
        output_dir,
        result["components"],
        hand_names,
        num_components=min(args.num_components_to_plot, args.hand_dim),
    )

    print(f"Loaded {hand_actions.shape[0]} frames from {len(sources)} source(s)")
    for name, values in sources:
        print(f"  - {name}: {values.shape[0]} frames")
    print(f"Saved PCA analysis to {output_dir}")
    for k in (2, 4, 6, 8, 10, 12):
        if k <= len(result["cumulative_ratio"]):
            print(f"K={k}: cumulative explained variance = {result['cumulative_ratio'][k - 1]:.4f}")


if __name__ == "__main__":
    main()
