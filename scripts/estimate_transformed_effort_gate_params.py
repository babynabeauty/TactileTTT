#!/usr/bin/env python3
"""Estimate tactile gate parameters on model-ready transformed effort.

This script uses the training config and data transforms, including normalization,
so the reported magnitude distribution matches what the model tokenizer actually
receives during training/inference.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import tqdm
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p}.")
    return math.log(p / (1.0 - p))


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _extract_effort_from_batch(batch) -> np.ndarray:
    """Returns transformed effort from either a raw dict batch or DataLoaderImpl output."""
    if isinstance(batch, tuple) and len(batch) == 2:
        observation, _actions = batch
        effort = getattr(observation, "effort", None)
        if effort is None:
            raise KeyError("Transformed Observation does not contain effort. Check config/data transforms.")
        return _as_numpy(effort).astype(np.float32)
    if isinstance(batch, dict):
        if "effort" not in batch:
            raise KeyError("Transformed batch does not contain 'effort'. Check config/data transforms.")
        return _as_numpy(batch["effort"]).astype(np.float32)
    raise TypeError(f"Unsupported dataloader batch type: {type(batch).__name__}")


def _summarize_magnitudes(
    magnitudes: np.ndarray,
    *,
    threshold: float | None,
    zero_gate: float,
    positive_gate: float,
    positive_percentile: float,
) -> None:
    flat = magnitudes.reshape(-1)
    positive = flat[flat > 0]
    abs_positive = flat[np.abs(flat) > 0]
    if positive.size == 0 and abs_positive.size == 0:
        raise ValueError("No non-zero effort magnitudes found.")

    # Magnitude is non-negative, so positive and non-zero are equivalent unless
    # the input has NaNs, which were filtered before this function.
    positive = positive if positive.size else abs_positive
    selected_threshold = float(threshold) if threshold is not None else float(np.percentile(positive, 1.0))
    low_positive = float(np.percentile(positive, positive_percentile))
    tau_zero = (0.0 - selected_threshold) / _logit(zero_gate)
    tau_positive = (low_positive - selected_threshold) / _logit(positive_gate)
    recommended_temperature = float(np.median([tau_zero, tau_positive]))

    print("all transformed magnitude percentiles:")
    for q in [0, 50, 75, 90, 95, 97, 99, 99.5, 99.9, 100]:
        print(f"  p{q:>5}: {np.percentile(flat, q):.6f}")
    print()
    print("non-zero transformed magnitude percentiles:")
    for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{q:>5}: {np.percentile(positive, q):.6f}")
    print()
    print("recommended gate params for transformed effort:")
    print(f"  tactile_raw_contact_threshold={selected_threshold:.6f}")
    print(f"  tactile_raw_contact_temperature={recommended_temperature:.6f}")
    print()
    print("anchors:")
    print(f"  gate(0 transformed magnitude) ~= {zero_gate:.3f} -> temperature {tau_zero:.6f}")
    print(
        f"  gate(p{positive_percentile:g} non-zero={low_positive:.6f}) "
        f"~= {positive_gate:.3f} -> temperature {tau_positive:.6f}"
    )


def main(
    config_name: str,
    repo_id: str | None = None,
    asset_id: str | None = None,
    assets_dir: str | None = None,
    max_batches: int = 200,
    batch_size: int = 8,
    num_workers: int = 0,
    threshold: float | None = None,
    zero_gate: float = 0.1,
    positive_gate: float = 0.8,
    positive_percentile: float = 5.0,
) -> None:
    config = _config.get_config(config_name)
    data = config.data
    if repo_id is not None:
        data = dataclasses.replace(data, repo_id=repo_id)
    if asset_id is not None:
        data = dataclasses.replace(data, assets=dataclasses.replace(data.assets, asset_id=asset_id))
    if assets_dir is not None:
        data = dataclasses.replace(data, assets=dataclasses.replace(data.assets, assets_dir=assets_dir))
    config = dataclasses.replace(config, data=data, batch_size=batch_size, num_workers=num_workers)

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        assets_root = data.assets.assets_dir or str(config.assets_dirs)
        raise ValueError(
            "Normalization stats are required because this script estimates the distribution after "
            "training-time Normalize(...). Could not find norm_stats.json for "
            f"config={config_name!r}, asset_id={data_config.asset_id!r}, assets_dir={assets_root!r}. "
            "Run scripts/compute_norm_stats.py for this config/dataset first, or pass "
            "--assets-dir to reuse an existing compatible assets directory."
        )
    loader = _data_loader.create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=batch_size,
        sharding=None,
        shuffle=True,
        num_batches=max_batches,
        num_workers=num_workers,
        seed=config.seed,
        skip_norm_stats=False,
        framework="pytorch",
    )

    chunks = []
    per_finger_chunks = []
    total_samples = 0
    for batch in tqdm.tqdm(loader, total=max_batches, desc="Scanning transformed effort"):
        effort = _extract_effort_from_batch(batch)
        if effort.ndim == 3:
            # [B, T, F*P*C] for legacy flatten configs. Try to recover raw tactile.
            if effort.shape[-1] % (5 * 120 * 3) != 0:
                raise ValueError(f"Cannot interpret flattened effort shape {effort.shape} as raw tactile.")
            effort = effort.reshape(*effort.shape[:-1], 5, 120, 3)
        if effort.ndim != 5 or effort.shape[-3:] != (5, 120, 3):
            raise ValueError(f"Expected transformed raw tactile effort [B,T,5,120,3], got {effort.shape}.")

        magnitude = np.linalg.norm(effort, axis=-1)
        finite = magnitude[np.isfinite(magnitude)]
        chunks.append(finite)
        per_finger_chunks.append(magnitude.reshape(-1, 5, 120))
        total_samples += effort.shape[0]

    magnitudes = np.concatenate(chunks)
    per_finger = np.concatenate(per_finger_chunks, axis=0)
    print(f"config_name: {config_name}")
    print(f"repo_id: {data_config.repo_id}")
    print(f"asset_id: {data_config.asset_id}")
    print(f"batches: {max_batches}")
    print(f"batch_size: {batch_size}")
    print(f"samples_seen: {total_samples}")
    print(f"total_taxel_values: {magnitudes.size}")
    print(f"nonzero_ratio: {(magnitudes > 0).mean():.6f}")
    print()
    _summarize_magnitudes(
        magnitudes,
        threshold=threshold,
        zero_gate=zero_gate,
        positive_gate=positive_gate,
        positive_percentile=positive_percentile,
    )
    print()
    print("per-finger transformed magnitude stats:")
    for finger in range(5):
        values = per_finger[:, finger, :].reshape(-1)
        nonzero = values[values > 0]
        if nonzero.size == 0:
            print(f"  finger {finger}: nonzero_ratio=0.000000")
            continue
        qs = np.percentile(nonzero, [5, 50, 95])
        print(
            f"  finger {finger}: nonzero_ratio={(values > 0).mean():.6f}, "
            f"p5={qs[0]:.6f}, p50={qs[1]:.6f}, p95={qs[2]:.6f}"
        )


if __name__ == "__main__":
    tyro.cli(main)
