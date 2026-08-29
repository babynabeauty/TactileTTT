"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import warnings
import dataclasses
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

def debug():
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("✅ Waiting for debugger to attach on port 5678...")
    debugpy.wait_for_client()
    print("Start to debugging")

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches

def main(
    config_name: str,
    max_frames: int | None = None,
    repo_id: str | None = None,
    asset_id: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    print(
        f"[compute_norm_stats] start config={config_name} repo_id={repo_id} asset_id={asset_id} max_frames={max_frames}",
        flush=True,
    )
    config = _config.get_config(config_name)
    data = config.data
    if repo_id is not None:
        data = dataclasses.replace(data, repo_id=repo_id)
    if asset_id is not None:
        data = dataclasses.replace(data, assets=dataclasses.replace(data.assets, asset_id=asset_id))
    if repo_id is not None and (pathlib.Path(repo_id) / "meta" / "info.json").exists():
        base_config = data.base_config or _config.DataConfig()
        data = dataclasses.replace(
            data,
            base_config=dataclasses.replace(
                base_config,
                state_action_only=True,
                prompt_from_task=False,
            ),
        )
    config = dataclasses.replace(config, data=data)
    print("[compute_norm_stats] creating data config", flush=True)
    data_config = config.data.create(config.assets_dirs, config.model)
    print(
        "[compute_norm_stats] data_config created: "
        f"repo_id={data_config.repo_id} "
        f"state_action_only={data_config.state_action_only} "
        f"state_delta_timestamps={tuple(data_config.state_delta_timestamps)}",
        flush=True,
    )

    effective_batch_size = batch_size or config.batch_size
    effective_num_workers = config.num_workers if num_workers is None else num_workers
    print(
        "[compute_norm_stats] loader args: "
        f"batch_size={effective_batch_size} num_workers={effective_num_workers}",
        flush=True,
    )

    if data_config.rlds_data_dir is not None:
        print("[compute_norm_stats] creating RLDS dataloader", flush=True)
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, effective_batch_size, max_frames
        )
    else:
        print("[compute_norm_stats] creating torch dataloader", flush=True)
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            effective_batch_size,
            config.model,
            effective_num_workers,
            max_frames,
        )
    print(f"[compute_norm_stats] dataloader ready: num_batches={num_batches}", flush=True)

    required_keys = ("state", "actions")
    optional_keys = ("effort", "tactile")
    stats = {key: normalize.RunningStats() for key in required_keys}
    warned_missing_required: set[str] = set()

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        # Lazily enable optional modalities if they appear in data.
        for key in optional_keys:
            if key in batch and key not in stats:
                stats[key] = normalize.RunningStats()
                tqdm.tqdm.write(f"Found optional key '{key}', enabling normalization stats.")

        for key, running_stats in stats.items():
            if key in batch:
                running_stats.update(np.asarray(batch[key]))
            elif key in required_keys and key not in warned_missing_required:
                warnings.warn(f"Required key '{key}' not found in batch. This key will be skipped.")
                warned_missing_required.add(key)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    # debug()
    tyro.cli(main)
