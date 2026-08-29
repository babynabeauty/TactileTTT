#!/usr/bin/env python3
"""Evaluate and visualize trained future tactile token probes.

This script restores a probe trained by train_future_tactile_token_probe.py,
freezes the original policy, evaluates future-force decoding on an eval split,
and saves:
  - metrics.json
  - per-sample future force magnitude/xyz plots
"""

import dataclasses
import json
import logging
import os
import pathlib
import sys
import time
from typing import Literal

_START_TIME = time.time()


def _stage(message: str) -> None:
    print(f"[eval_probe +{time.time() - _START_TIME:7.1f}s] {message}", flush=True)


_stage("starting imports")
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

_stage("importing openpi modules")
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
from openpi.models import gemma as _gemma

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_future_tactile_token_probe as probe_lib

_stage("imports finished")


@dataclasses.dataclass(frozen=True)
class Args:
    probe_dir: str
    output_dir: str | None = None

    pretrained_params: str | None = None
    config_name: str | None = None
    repo_id: str | None = None
    asset_id: str | None = None
    assets_dir: str | None = None
    eval_filter_path: str | None = None
    token_source: Literal["student", "teacher"] | None = None
    probe_layer: int | None = None

    batch_size: int | None = None
    fsdp_devices: int | None = None
    num_workers: int | None = None
    seed: int | None = None

    max_batches: int = 100
    num_plot_samples: int = 8
    contact_threshold: float | None = None
    inspect_only: bool = False


def _resolve_probe_dir(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if (path / "args.json").exists():
        child_dirs = [child for child in (path / "student", path / "teacher") if child.exists()]
        if child_dirs:
            # User passed the experiment directory. Prefer the only child if possible.
            if len(child_dirs) == 1:
                return child_dirs[0]
            raise ValueError(
                f"{path} contains both student and teacher probe dirs. "
                "Pass one of them explicitly."
            )
        return path
    if path.name in ("student", "teacher") and (path.parent / "args.json").exists():
        return path
    for child_name in ("student", "teacher"):
        child = path / child_name
        if child.exists() and (path / "args.json").exists():
            return child
    raise FileNotFoundError(
        f"Could not find probe args/checkpoint under {path}. Expected either "
        "an experiment dir with args.json and student/teacher child, or the child dir itself."
    )


def _load_probe_args(probe_dir: pathlib.Path) -> probe_lib.Args:
    args_path = probe_dir / "args.json"
    if not args_path.exists():
        args_path = probe_dir.parent / "args.json"
    raw = json.loads(args_path.read_text())
    valid = {field.name for field in dataclasses.fields(probe_lib.Args)}
    return probe_lib.Args(**{key: value for key, value in raw.items() if key in valid})


def _inspect_probe(probe_dir: pathlib.Path, saved_args: probe_lib.Args, merged_args: probe_lib.Args) -> None:
    args_path = probe_dir / "args.json"
    if not args_path.exists():
        args_path = probe_dir.parent / "args.json"

    _stage(f"resolved probe_dir: {probe_dir}")
    _stage(f"args.json: {args_path}")
    _stage(f"probe_dir exists: {probe_dir.exists()}")
    _stage(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    _stage(f"HF_LEROBOT_HOME={os.environ.get('HF_LEROBOT_HOME', '<unset>')}")
    _stage(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '<unset>')}")

    print("\nSaved probe args:", flush=True)
    for key in (
        "exp_name",
        "config_name",
        "repo_id",
        "asset_id",
        "assets_dir",
        "pretrained_params",
        "token_source",
        "probe_layer",
        "batch_size",
        "fsdp_devices",
        "num_workers",
        "eval_filter_path",
    ):
        print(f"  {key}: {getattr(saved_args, key, None)}", flush=True)

    print("\nMerged eval args:", flush=True)
    for key in (
        "config_name",
        "repo_id",
        "asset_id",
        "assets_dir",
        "pretrained_params",
        "token_source",
        "probe_layer",
        "batch_size",
        "fsdp_devices",
        "num_workers",
        "eval_filter_path",
    ):
        print(f"  {key}: {getattr(merged_args, key, None)}", flush=True)

    print("\nProbe directory preview:", flush=True)
    if probe_dir.exists():
        entries = sorted(probe_dir.rglob("*"))
        for entry in entries[:80]:
            suffix = "/" if entry.is_dir() else ""
            print(f"  {entry.relative_to(probe_dir)}{suffix}", flush=True)
        if len(entries) > 80:
            print(f"  ... ({len(entries) - 80} more entries)", flush=True)


def _merge_args(cli: Args, saved: probe_lib.Args, probe_dir: pathlib.Path) -> probe_lib.Args:
    token_source = cli.token_source or (probe_dir.name if probe_dir.name in ("student", "teacher") else saved.token_source)
    return dataclasses.replace(
        saved,
        pretrained_params=cli.pretrained_params or saved.pretrained_params,
        config_name=cli.config_name or saved.config_name,
        repo_id=cli.repo_id if cli.repo_id is not None else saved.repo_id,
        asset_id=cli.asset_id if cli.asset_id is not None else saved.asset_id,
        assets_dir=cli.assets_dir if cli.assets_dir is not None else saved.assets_dir,
        eval_filter_path=cli.eval_filter_path if cli.eval_filter_path is not None else saved.eval_filter_path,
        train_filter_path=None,
        token_source=token_source,
        probe_layer=cli.probe_layer if cli.probe_layer is not None else saved.probe_layer,
        batch_size=cli.batch_size or saved.batch_size,
        fsdp_devices=cli.fsdp_devices or saved.fsdp_devices,
        num_workers=saved.num_workers if cli.num_workers is None else cli.num_workers,
        seed=saved.seed if cli.seed is None else cli.seed,
        contact_threshold=saved.contact_threshold if cli.contact_threshold is None else cli.contact_threshold,
        resume=True,
        overwrite=False,
    )


def _restore_probe_state(
    probe_dir: pathlib.Path,
    probe_state: probe_lib.ProbeTrainState,
) -> tuple[probe_lib.ProbeTrainState, int]:
    _stage(f"opening Orbax CheckpointManager: {probe_dir}")
    manager = ocp.CheckpointManager(
        probe_dir,
        item_handlers={"probe_state": ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    latest_step = manager.latest_step()
    if latest_step is None:
        raise FileNotFoundError(f"No Orbax probe checkpoints found under {probe_dir}")
    _stage(f"restoring probe checkpoint step={latest_step}")
    restored = manager.restore(
        latest_step,
        args=ocp.args.Composite(probe_state=ocp.args.PyTreeRestore(probe_state)),
    )["probe_state"]
    _stage("probe checkpoint restored")
    return restored, int(latest_step)


def _tree_scalar(value) -> float:
    value = jax.device_get(value)
    return float(np.asarray(value))


def _plot_sample(
    out_path: pathlib.Path,
    pred: np.ndarray,
    target: np.ndarray,
    history: np.ndarray,
    title: str,
) -> None:
    # pred/target: [32, 5, 3], history: [10, 5, 3]
    horizon = pred.shape[0]
    fingers = pred.shape[1]
    x_future = np.arange(1, horizon + 1)
    x_history = np.arange(-history.shape[0] + 1, 1)
    pred_mag = np.linalg.norm(pred, axis=-1)
    target_mag = np.linalg.norm(target, axis=-1)
    hist_mag = np.linalg.norm(history, axis=-1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for finger in range(fingers):
        axes[0].plot(x_history, hist_mag[:, finger], linestyle=":", alpha=0.65, label=f"hist f{finger}")
        axes[0].plot(x_future, target_mag[:, finger], linewidth=2, label=f"gt f{finger}")
        axes[0].plot(x_future, pred_mag[:, finger], linestyle="--", label=f"pred f{finger}")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title(title + " | force magnitude")
    axes[0].set_ylabel("norm(force)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=5, fontsize=8)

    names = ["Fx", "Fy", "Fz"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for dim, name in enumerate(names):
        axes[1].plot(x_future, target[:, :, dim].mean(axis=1), color=colors[dim], linewidth=2, label=f"gt mean {name}")
        axes[1].plot(
            x_future,
            pred[:, :, dim].mean(axis=1),
            color=colors[dim],
            linestyle="--",
            label=f"pred mean {name}",
        )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Mean over five fingers | xyz components")
    axes[1].set_xlabel("future step")
    axes[1].set_ylabel("force")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    _stage("main entered")
    probe_dir = _resolve_probe_dir(pathlib.Path(args.probe_dir))
    saved_args = _load_probe_args(probe_dir)
    probe_args = _merge_args(args, saved_args, probe_dir)
    _stage(f"resolved probe_dir={probe_dir}")

    if args.inspect_only:
        _inspect_probe(probe_dir, saved_args, probe_args)
        return

    output_dir = pathlib.Path(args.output_dir or f"outputs/future_tactile_token_probe_eval/{probe_args.exp_name}_{probe_args.token_source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_args.json").write_text(json.dumps(dataclasses.asdict(probe_args), indent=2, ensure_ascii=False))
    _stage(f"output_dir={output_dir}")

    _stage("building eval config")
    eval_config = probe_lib._override_config(probe_args, filter_path=probe_args.eval_filter_path)
    probe_lib._validate_model_config(eval_config)
    probe_layer = int(probe_args.probe_layer if probe_args.probe_layer is not None else eval_config.model.future_tactile_align_layer)
    _stage(
        "eval config ready: "
        f"config={probe_args.config_name}, repo={probe_args.repo_id}, "
        f"asset={probe_args.asset_id}, token_source={probe_args.token_source}, layer={probe_layer}"
    )

    _stage(f"creating mesh fsdp_devices={probe_args.fsdp_devices}; jax_devices={jax.devices()}")
    mesh = sharding.make_mesh(probe_args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    _stage("creating eval dataloader")
    eval_loader = _data_loader.create_data_loader(eval_config, sharding=data_sharding, shuffle=False)
    _stage("eval dataloader created")

    init_rng = jax.random.PRNGKey(probe_args.seed)
    model_rng, probe_rng, eval_rng = jax.random.split(init_rng, 3)
    with sharding.set_mesh(mesh):
        _stage("initializing frozen pi0 model and loading params")
        model_def, model_params = probe_lib._init_frozen_model(eval_config, model_rng)
        model_params = jax.device_put(model_params, replicated)
        _stage("frozen pi0 model ready")

        student_width = int(_gemma.get_config(eval_config.model.action_expert_variant).width)
        teacher_variant = getattr(eval_config.model, "force_expert_variant", eval_config.model.action_expert_variant)
        teacher_width = int(_gemma.get_config(teacher_variant).width)
        probe_input_width = teacher_width if probe_args.token_source == "teacher" else student_width
        probe = probe_lib.FutureTactileForceProbe(
            input_dim=probe_input_width,
            hidden_dim=probe_args.decoder_dim,
            action_horizon=eval_config.model.action_horizon,
            num_fingers=eval_config.model.tactile_num_fingers,
            force_dim=eval_config.model.tactile_dim_per_finger,
            depth=probe_args.decoder_depth,
            rngs=nnx.Rngs(probe_rng),
        )
        _stage("initialized probe decoder")
        probe_def, probe_params = nnx.split(probe)
        tx = optax.adamw(probe_args.learning_rate)
        state = probe_lib.ProbeTrainState(
            step=0,
            params=probe_params,
            opt_state=tx.init(probe_params),
            tx=tx,
        )
        state = jax.device_put(state, replicated)
        state, latest_step = _restore_probe_state(probe_dir, state)

        def _eval_with_preds(model_params_arg, state_arg, batch_arg, rng_arg):
            model = nnx.merge(model_def, model_params_arg)
            model.eval()
            probe_model = nnx.merge(probe_def, state_arg.params)
            future_hidden, target_force, history_force = probe_lib._extract_future_token_hiddens(
                model,
                rng_arg,
                batch_arg[0],
                batch_arg[1],
                probe_layer=probe_layer,
                token_source=probe_args.token_source,
            )
            pred_force = probe_model(future_hidden)
            _, stats = probe_lib._loss_and_stats(probe_args, pred_force, target_force, history_force)
            return stats, pred_force, target_force, history_force

        peval = jax.jit(_eval_with_preds)
        _stage("JIT eval function created; first batch will include compile time")
        totals: dict[str, float] = {}
        count = 0
        plotted = 0
        batch_idx = 0
        eval_iter = iter(eval_loader)
        while args.max_batches <= 0 or batch_idx < args.max_batches:
            _stage(f"fetching eval batch {batch_idx + 1}/{args.max_batches if args.max_batches > 0 else '?'}")
            try:
                batch = next(eval_iter)
            except StopIteration:
                break
            _stage(f"processing eval batch {batch_idx + 1}/{args.max_batches if args.max_batches > 0 else '?'}")
            stats, pred, target, history = peval(model_params, state, batch, eval_rng)
            jax.block_until_ready(pred)
            batch_size = int(pred.shape[0])
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + _tree_scalar(value) * batch_size
            count += batch_size

            if plotted < args.num_plot_samples:
                pred_np = np.asarray(jax.device_get(pred))
                target_np = np.asarray(jax.device_get(target))
                history_np = np.asarray(jax.device_get(history))
                for i in range(min(batch_size, args.num_plot_samples - plotted)):
                    _plot_sample(
                        output_dir / f"sample_{plotted:03d}.png",
                        pred_np[i],
                        target_np[i],
                        history_np[i],
                        f"{probe_args.token_source} probe step={latest_step} sample={plotted}",
                    )
                    plotted += 1
            batch_idx += 1

        if count == 0:
            raise RuntimeError("No eval batches were processed.")
        metrics = {key: value / count for key, value in sorted(totals.items())}
        metrics.update(
            {
                "probe_checkpoint_dir": str(probe_dir),
                "probe_step": latest_step,
                "token_source": probe_args.token_source,
                "probe_layer": probe_layer,
                "eval_samples": count,
            }
        )
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
        logging.info("Saved metrics and plots to %s", output_dir)
        for key, value in metrics.items():
            if isinstance(value, float):
                logging.info("%s = %.6f", key, value)
            else:
                logging.info("%s = %s", key, value)


if __name__ == "__main__":
    main(tyro.cli(Args))
