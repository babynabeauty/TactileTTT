#!/usr/bin/env python3
"""Scan and visualize XHand1 tactile data size/statistics in a LeRobot dataset.

The script supports both common XHand encodings used in this repo:

1. Explicit tactile columns: tactile, observation.tactile, observation/tactile.
2. Full observation.state rows containing XHand tactile blocks.

For raw XHand1 tactile data the expected per-frame shape is [5, 120, 3]:
five fingertips, 120 taxels per fingertip, 3 force channels per taxel.
For reduced calc-force tactile data the expected per-frame shape is [5, 3].
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TACTILE_SENSOR_COUNT = 5
TACTILE_BLOCK_SIZE = 384
TACTILE_BLOCK_START = 52
TACTILE_CALC_FORCE_OFFSET = 0
TACTILE_RAW_FORCE_OFFSET = 24
TACTILE_RAW_FORCE_POINTS = 120
TACTILE_CHANNELS = 3
EXPECTED_RAW_SHAPE = (TACTILE_SENSOR_COUNT, TACTILE_RAW_FORCE_POINTS, TACTILE_CHANNELS)
EXPECTED_CALC_SHAPE = (TACTILE_SENSOR_COUNT, TACTILE_CHANNELS)
EXPECTED_RAW_VALUES = int(np.prod(EXPECTED_RAW_SHAPE))
EXPECTED_CALC_VALUES = int(np.prod(EXPECTED_CALC_SHAPE))
FLOAT32_BYTES = np.dtype(np.float32).itemsize
EXPECTED_RAW_BYTES = EXPECTED_RAW_VALUES * FLOAT32_BYTES
EXPECTED_CALC_BYTES = EXPECTED_CALC_VALUES * FLOAT32_BYTES

STATE_COLUMNS = ("observation.state", "observation/state", "state", "observation_state", "robot_state")
TACTILE_COLUMNS = ("observation.tactile", "observation/tactile", "tactile")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


def _episode_index_from_path(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def _find_episode_files(dataset: Path) -> list[Path]:
    if dataset.is_file() and dataset.suffix == ".parquet":
        return [dataset]

    files = sorted(dataset.glob("data/**/episode_*.parquet"))
    if files:
        return files

    files = sorted(dataset.glob("**/episode_*.parquet"))
    if files:
        return files

    raise FileNotFoundError(f"No episode_*.parquet files found under {dataset}")


def _read_parquet(path: Path, columns: list[str] | None = None) -> Any:
    try:
        import pandas as pd

        return pd.read_parquet(path, columns=columns)
    except Exception:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns)
        return table.to_pandas()


def _available_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        df = _read_parquet(path, columns=None)
        return list(df.columns)


def _first_present(candidates: tuple[str, ...], available: set[str]) -> str | None:
    for column in candidates:
        if column in available:
            return column
    return None


def _as_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist(), dtype=np.float32)
    return arr


def _reshape_explicit_tactile(value: Any, mode: str) -> np.ndarray:
    arr = _as_vector(value)
    if mode == "raw":
        if arr.shape == EXPECTED_RAW_SHAPE:
            return arr
        if arr.size == EXPECTED_RAW_VALUES:
            return arr.reshape(EXPECTED_RAW_SHAPE)
        raise ValueError(f"Expected raw tactile shape {EXPECTED_RAW_SHAPE} or {EXPECTED_RAW_VALUES} values, got {arr.shape}")

    if mode == "calc":
        if arr.shape == EXPECTED_CALC_SHAPE:
            return arr
        if arr.size == EXPECTED_CALC_VALUES:
            return arr.reshape(EXPECTED_CALC_SHAPE)
        raise ValueError(f"Expected calc tactile shape {EXPECTED_CALC_SHAPE} or {EXPECTED_CALC_VALUES} values, got {arr.shape}")

    if arr.shape == EXPECTED_RAW_SHAPE or arr.size == EXPECTED_RAW_VALUES:
        return arr.reshape(EXPECTED_RAW_SHAPE)
    if arr.shape == EXPECTED_CALC_SHAPE or arr.size == EXPECTED_CALC_VALUES:
        return arr.reshape(EXPECTED_CALC_SHAPE)
    raise ValueError(f"Cannot infer tactile mode from shape {arr.shape}")


def _extract_raw_from_state(state_value: Any) -> np.ndarray:
    state = _as_vector(state_value).reshape(-1)
    required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    if state.size < required_dim:
        raise ValueError(f"Need observation.state with at least {required_dim} values for raw tactile, got {state.size}")

    chunks = []
    for sensor_id in range(TACTILE_SENSOR_COUNT):
        start = TACTILE_BLOCK_START + sensor_id * TACTILE_BLOCK_SIZE + TACTILE_RAW_FORCE_OFFSET
        end = start + TACTILE_RAW_FORCE_POINTS * TACTILE_CHANNELS
        chunks.append(state[start:end].reshape(TACTILE_RAW_FORCE_POINTS, TACTILE_CHANNELS))
    return np.stack(chunks, axis=0).astype(np.float32)


def _extract_calc_from_state(state_value: Any) -> np.ndarray:
    state = _as_vector(state_value).reshape(-1)
    required_dim = TACTILE_BLOCK_START + TACTILE_SENSOR_COUNT * TACTILE_BLOCK_SIZE
    if state.size < required_dim:
        raise ValueError(f"Need observation.state with at least {required_dim} values for calc tactile, got {state.size}")

    chunks = []
    for sensor_id in range(TACTILE_SENSOR_COUNT):
        start = TACTILE_BLOCK_START + sensor_id * TACTILE_BLOCK_SIZE + TACTILE_CALC_FORCE_OFFSET
        end = start + TACTILE_CHANNELS
        chunks.append(state[start:end])
    return np.stack(chunks, axis=0).astype(np.float32)


def _finger_stats(tactile: np.ndarray, mode: str, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "raw":
        magnitudes = np.linalg.norm(tactile, axis=-1)
        return magnitudes.sum(axis=-1), magnitudes.max(axis=-1), np.sum(magnitudes > threshold, axis=-1)

    magnitudes = np.linalg.norm(tactile, axis=-1)
    return magnitudes, magnitudes, (magnitudes > threshold).astype(np.int64)


def _shape_text(arr: np.ndarray) -> str:
    return "x".join(str(dim) for dim in arr.shape)


def _expected_size(mode: str) -> tuple[int, int]:
    if mode == "raw":
        return EXPECTED_RAW_VALUES, EXPECTED_RAW_BYTES
    return EXPECTED_CALC_VALUES, EXPECTED_CALC_BYTES


def _scan_episode(
    *,
    path: Path,
    dataset_root: Path,
    tactile_mode: str,
    threshold: float,
    max_rows_per_episode: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    available = set(_available_columns(path))
    tactile_column = _first_present(TACTILE_COLUMNS, available)
    state_column = _first_present(STATE_COLUMNS, available)

    if tactile_column is not None:
        source_column = tactile_column
        columns = [source_column]
    elif state_column is not None:
        source_column = state_column
        columns = [source_column]
    else:
        raise ValueError(
            f"{path} has no tactile/state column. Tried tactile={TACTILE_COLUMNS}, state={STATE_COLUMNS}. "
            f"Available={sorted(available)}"
        )

    for optional in ("episode_index", "frame_index", "timestamp"):
        if optional in available:
            columns.append(optional)

    df = _read_parquet(path, columns=columns)
    if max_rows_per_episode is not None:
        df = df.head(max_rows_per_episode)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    fallback_episode_index = _episode_index_from_path(path)
    relative_file = str(path.relative_to(dataset_root)) if path.is_relative_to(dataset_root) else str(path)

    for row_pos, row in df.iterrows():
        try:
            if tactile_column is not None:
                tactile = _reshape_explicit_tactile(row[source_column], tactile_mode)
                actual_mode = "raw" if tactile.ndim == 3 else "calc"
            elif tactile_mode == "calc":
                tactile = _extract_calc_from_state(row[source_column])
                actual_mode = "calc"
            elif tactile_mode in ("raw", "auto"):
                tactile = _extract_raw_from_state(row[source_column])
                actual_mode = "raw"
            else:
                raise ValueError(f"Unsupported tactile mode: {tactile_mode}")

            finger_sum, finger_max, finger_active = _finger_stats(tactile, actual_mode, threshold)
            total_values = int(tactile.size)
            expected_values, expected_bytes = _expected_size(actual_mode)
            sample_bytes_float32 = total_values * FLOAT32_BYTES
            rows.append(
                {
                    "episode_file": relative_file,
                    "episode_index": int(row.get("episode_index", fallback_episode_index)),
                    "frame_index": int(row.get("frame_index", row_pos)),
                    "timestamp": float(row.get("timestamp", np.nan)),
                    "row_position": int(row_pos),
                    "source_column": source_column,
                    "mode": actual_mode,
                    "shape": _shape_text(tactile),
                    "num_values": total_values,
                    "expected_values": expected_values,
                    "sample_bytes_float32": sample_bytes_float32,
                    "expected_bytes_float32": expected_bytes,
                    "sample_kib_float32": sample_bytes_float32 / 1024.0,
                    "within_xhand1_shape_limit": int(total_values <= expected_values),
                    "within_xhand1_byte_limit_float32": int(sample_bytes_float32 <= expected_bytes),
                    "shape_matches_expected": int(total_values == expected_values),
                    "total_force_size": float(np.sum(finger_sum)),
                    "mean_force_size": float(np.mean(finger_sum)),
                    "max_taxel_or_finger_force": float(np.max(finger_max)),
                    "active_count": int(np.sum(finger_active)),
                    **{f"finger_{i}_{FINGER_NAMES[i]}_sum": float(finger_sum[i]) for i in range(TACTILE_SENSOR_COUNT)},
                    **{f"finger_{i}_{FINGER_NAMES[i]}_max": float(finger_max[i]) for i in range(TACTILE_SENSOR_COUNT)},
                    **{f"finger_{i}_{FINGER_NAMES[i]}_active": int(finger_active[i]) for i in range(TACTILE_SENSOR_COUNT)},
                }
            )
        except Exception as exc:
            errors.append(f"{relative_file}: row={row_pos}: {exc}")

    return rows, errors


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_shape_counts(rows: list[dict[str, Any]], output_dir: Path) -> None:
    counter = Counter(row["shape"] for row in rows)
    labels, counts = zip(*counter.most_common(), strict=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(labels)), 4.6), constrained_layout=True)
    ax.bar(range(len(labels)), counts, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("rows")
    ax.set_title("Tactile array shape counts")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "shape_counts.png", dpi=180)
    plt.close(fig)


def _plot_force_distributions(rows: list[dict[str, Any]], output_dir: Path) -> None:
    totals = np.asarray([row["total_force_size"] for row in rows], dtype=np.float32)
    max_values = np.asarray([row["max_taxel_or_finger_force"] for row in rows], dtype=np.float32)
    active = np.asarray([row["active_count"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    axes[0].hist(totals, bins=80, color="#4c78a8")
    axes[0].set_title("Total tactile force size")
    axes[0].set_xlabel("sum of magnitudes")
    axes[0].set_ylabel("rows")
    axes[1].hist(max_values, bins=80, color="#f58518")
    axes[1].set_title("Max taxel/finger force")
    axes[1].set_xlabel("max magnitude")
    axes[2].hist(active, bins=80, color="#54a24b")
    axes[2].set_title("Active tactile points")
    axes[2].set_xlabel("count above threshold")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "force_size_distributions.png", dpi=180)
    plt.close(fig)


def _plot_storage_size_distributions(rows: list[dict[str, Any]], output_dir: Path) -> None:
    num_values = np.asarray([row["num_values"] for row in rows], dtype=np.float32)
    sample_kib = np.asarray([row["sample_kib_float32"] for row in rows], dtype=np.float32)
    expected_values = np.asarray([row["expected_values"] for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axes[0].hist(num_values, bins=min(80, max(10, len(np.unique(num_values)))), color="#4c78a8")
    axes[0].axvline(EXPECTED_CALC_VALUES, color="#54a24b", linestyle="--", linewidth=1.4, label="calc limit: 15")
    axes[0].axvline(EXPECTED_RAW_VALUES, color="#e45756", linestyle="--", linewidth=1.4, label="raw limit: 1800")
    axes[0].set_title("Tactile sample element count")
    axes[0].set_xlabel("float values per row")
    axes[0].set_ylabel("rows")
    axes[0].legend()

    axes[1].hist(sample_kib, bins=min(80, max(10, len(np.unique(sample_kib)))), color="#f58518")
    axes[1].axvline(EXPECTED_CALC_BYTES / 1024.0, color="#54a24b", linestyle="--", linewidth=1.4, label="calc float32 KiB")
    axes[1].axvline(EXPECTED_RAW_BYTES / 1024.0, color="#e45756", linestyle="--", linewidth=1.4, label="raw float32 KiB")
    axes[1].set_title("Tactile sample storage size")
    axes[1].set_xlabel("KiB per row if stored as float32")
    axes[1].legend()

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(output_dir / "sample_storage_size_distributions.png", dpi=180)
    plt.close(fig)

    mismatch = np.flatnonzero(num_values != expected_values)
    if mismatch.size:
        bad_rows = [rows[int(i)] for i in mismatch[:2000]]
        _write_csv(bad_rows, output_dir / "shape_or_size_mismatch_preview.csv")


def _plot_episode_overview(rows: list[dict[str, Any]], output_dir: Path, max_points: int) -> None:
    ordered = sorted(rows, key=lambda item: (item["episode_index"], item["frame_index"], item["row_position"]))
    if len(ordered) > max_points:
        indices = np.linspace(0, len(ordered) - 1, max_points).astype(np.int64)
        ordered = [ordered[int(i)] for i in indices]

    x = np.arange(len(ordered))
    totals = np.asarray([row["total_force_size"] for row in ordered], dtype=np.float32)
    max_values = np.asarray([row["max_taxel_or_finger_force"] for row in ordered], dtype=np.float32)
    episode_indices = [row["episode_index"] for row in ordered]

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, totals, linewidth=0.9, color="#4c78a8")
    axes[0].set_ylabel("sum magnitude")
    axes[0].set_title("Per-row tactile force size across dataset")
    axes[1].plot(x, max_values, linewidth=0.9, color="#f58518")
    axes[1].set_ylabel("max magnitude")
    axes[1].set_xlabel("row order (episode, frame)")

    boundaries = []
    last_episode = None
    for idx, episode in enumerate(episode_indices):
        if episode != last_episode:
            boundaries.append((idx, episode))
            last_episode = episode
    for ax in axes:
        for idx, _episode in boundaries:
            ax.axvline(idx, color="black", alpha=0.08, linewidth=0.8)
        ax.grid(alpha=0.25)

    if len(boundaries) <= 40:
        tick_positions = [idx for idx, _episode in boundaries]
        tick_labels = [str(episode) for _idx, episode in boundaries]
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels(tick_labels, rotation=45, ha="right")
        axes[1].set_xlabel("episode index")

    fig.savefig(output_dir / "dataset_row_force_size.png", dpi=180)
    plt.close(fig)


def _plot_finger_boxplot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    data = [
        [row[f"finger_{finger}_{FINGER_NAMES[finger]}_sum"] for row in rows]
        for finger in range(TACTILE_SENSOR_COUNT)
    ]
    labels = [f"{i}:{name}" for i, name in enumerate(FINGER_NAMES)]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("per-row finger force size")
    ax.set_title("Per-finger tactile force size distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "per_finger_force_size_boxplot.png", dpi=180)
    plt.close(fig)


def _write_summary(rows: list[dict[str, Any]], errors: list[str], output_dir: Path, args: argparse.Namespace) -> None:
    totals = np.asarray([row["total_force_size"] for row in rows], dtype=np.float32)
    max_values = np.asarray([row["max_taxel_or_finger_force"] for row in rows], dtype=np.float32)
    active = np.asarray([row["active_count"] for row in rows], dtype=np.float32)
    sample_bytes = np.asarray([row["sample_bytes_float32"] for row in rows], dtype=np.float32)
    shapes = Counter(row["shape"] for row in rows)
    summary = {
        "dataset": str(args.dataset),
        "tactile_mode_requested": args.tactile_mode,
        "threshold": args.threshold,
        "xhand1_expected": {
            "raw_shape": list(EXPECTED_RAW_SHAPE),
            "raw_values": EXPECTED_RAW_VALUES,
            "raw_float32_bytes_per_frame": EXPECTED_RAW_BYTES,
            "calc_shape": list(EXPECTED_CALC_SHAPE),
            "calc_values": EXPECTED_CALC_VALUES,
            "calc_float32_bytes_per_frame": EXPECTED_CALC_BYTES,
        },
        "num_rows": len(rows),
        "num_errors": len(errors),
        "shape_counts": dict(shapes),
        "num_shape_mismatch": int(sum(1 for row in rows if not row["shape_matches_expected"])),
        "num_over_xhand1_shape_limit": int(sum(1 for row in rows if not row["within_xhand1_shape_limit"])),
        "sample_bytes_float32": {
            "mean": float(np.mean(sample_bytes)) if sample_bytes.size else None,
            "p50": float(np.percentile(sample_bytes, 50)) if sample_bytes.size else None,
            "max": float(np.max(sample_bytes)) if sample_bytes.size else None,
        },
        "total_force_size": {
            "mean": float(np.mean(totals)) if totals.size else None,
            "p50": float(np.percentile(totals, 50)) if totals.size else None,
            "p95": float(np.percentile(totals, 95)) if totals.size else None,
            "max": float(np.max(totals)) if totals.size else None,
        },
        "max_taxel_or_finger_force": {
            "mean": float(np.mean(max_values)) if max_values.size else None,
            "p95": float(np.percentile(max_values, 95)) if max_values.size else None,
            "max": float(np.max(max_values)) if max_values.size else None,
        },
        "active_count": {
            "mean": float(np.mean(active)) if active.size else None,
            "p95": float(np.percentile(active, 95)) if active.size else None,
            "max": float(np.max(active)) if active.size else None,
        },
        "errors_preview": errors[:50],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        (output_dir / "errors.txt").write_text("\n".join(errors) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset root or one episode parquet file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/xhand_tactile_dataset_size"))
    parser.add_argument(
        "--tactile-mode",
        choices=("raw", "calc", "auto"),
        default="raw",
        help="raw=[5,120,3], calc=[5,3]. auto only applies to explicit tactile columns.",
    )
    parser.add_argument("--threshold", type=float, default=1.0, help="Magnitude threshold for active tactile points.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional debugging cap.")
    parser.add_argument("--max-rows-per-episode", type=int, default=None, help="Optional debugging cap.")
    parser.add_argument("--max-overview-points", type=int, default=25000, help="Downsample line plot beyond this many rows.")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = _find_episode_files(dataset)
    if args.max_episodes is not None:
        episode_files = episode_files[: args.max_episodes]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, episode_path in enumerate(episode_files, start=1):
        print(f"[{index}/{len(episode_files)}] scanning {episode_path}", flush=True)
        episode_rows, episode_errors = _scan_episode(
            path=episode_path,
            dataset_root=dataset if dataset.is_dir() else dataset.parent,
            tactile_mode=args.tactile_mode,
            threshold=args.threshold,
            max_rows_per_episode=args.max_rows_per_episode,
        )
        rows.extend(episode_rows)
        errors.extend(episode_errors)

    csv_path = output_dir / "tactile_size_rows.csv"
    _write_csv(rows, csv_path)
    _write_summary(rows, errors, output_dir, args)

    if rows:
        _plot_shape_counts(rows, output_dir)
        _plot_storage_size_distributions(rows, output_dir)
        _plot_force_distributions(rows, output_dir)
        _plot_episode_overview(rows, output_dir, args.max_overview_points)
        _plot_finger_boxplot(rows, output_dir)

    print(f"Scanned rows: {len(rows)}")
    print(f"Errors:       {len(errors)}")
    print(f"CSV:          {csv_path}")
    print(f"Summary:      {output_dir / 'summary.json'}")
    if rows:
        print(f"Plots:        {output_dir}")


if __name__ == "__main__":
    main()
