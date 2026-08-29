from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# import openpi.models.model as _model
import openpi.models.model_tavla as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.shared.effort_type import EffortType


T_co = TypeVar("T_co", covariant=True)


def _load_episode_indices(filter_dict_path: str | None) -> list[int] | None:
    if not filter_dict_path:
        return None
    path = pathlib.Path(filter_dict_path).expanduser()
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict):
        episodes = (
            data.get("episodes")
            or data.get("episode_indices")
            or data.get("episode_index")
            or data.get("train")
            or data.get("val")
        )
    else:
        raise ValueError(f"Unsupported filter file format in {path}")
    if episodes is None:
        raise ValueError(
            f"Episode filter {path} must contain a list or one of keys: "
            "episodes, episode_indices, episode_index, train, val."
        )
    return [int(episode) for episode in episodes]


class EpisodeSubsetDataset:
    """Frame-level subset that keeps the wrapped LeRobotDataset's global indices."""

    def __init__(self, dataset, indices: Sequence[int]):
        self._dataset = dataset
        self._indices = [int(index) for index in indices]

    def __getitem__(self, index: SupportsIndex):
        return self._dataset[self._indices[int(index)]]

    def __len__(self) -> int:
        return len(self._indices)


def _to_numpy_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _column_to_numpy(column) -> np.ndarray:
    return np.asarray([int(np.asarray(item).item()) for item in column], dtype=np.int64)


def _arrow_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(array), array.type.list_size)
    return array.to_numpy(zero_copy_only=False)


class LocalParquetStateActionOnlyDataset:
    """Fast local LeRobot v2.1 parquet reader for tactile-only pretraining.

    This bypasses `LeRobotDataset`/HuggingFace dataset generation entirely and
    reads only parquet-backed state/action columns. It is intentionally limited
    to local datasets used by Stage-1 tactile encoder pretraining.
    """

    def __init__(
        self,
        root: pathlib.Path,
        *,
        state_delta_timestamps: Sequence[int],
        action_horizon: int,
        action_key: str,
    ):
        self.root = root
        self._state_offsets = tuple(int(offset) for offset in state_delta_timestamps) or (0,)
        self._action_offsets = tuple(range(int(action_horizon)))
        self._action_key = action_key
        self._dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)

        info_path = root / "meta" / "info.json"
        with info_path.open() as f:
            info = json.load(f)
        data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
        glob_pattern = data_pattern.replace("{episode_chunk:03d}", "*").replace("{episode_index:06d}", "*")
        parquet_files = sorted(root.glob(glob_pattern))
        if not parquet_files:
            parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under local dataset {root}.")

        columns = ["observation.state", action_key, "episode_index"]
        optional_columns = ["frame_index", "task_index", "timestamp"]
        logging.info("Loading local state/action parquet columns from %d files.", len(parquet_files))

        states = []
        actions = []
        metadata: dict[str, list[np.ndarray]] = {key: [] for key in ["episode_index", *optional_columns]}
        for parquet_file in parquet_files:
            schema_names = set(pq.read_schema(parquet_file).names)
            missing = [column for column in columns if column not in schema_names]
            if missing:
                raise ValueError(f"Missing columns {missing} in {parquet_file}.")
            read_columns = columns + [column for column in optional_columns if column in schema_names]
            table = pq.read_table(parquet_file, columns=read_columns)
            states.append(_arrow_column_to_numpy(table["observation.state"]).astype(np.float32))
            actions.append(_arrow_column_to_numpy(table[action_key]).astype(np.float32))
            for key in metadata:
                if key in table.column_names:
                    metadata[key].append(_arrow_column_to_numpy(table[key]))

        self._states = np.concatenate(states, axis=0)
        self._actions = np.concatenate(actions, axis=0)
        self._episode_index = np.concatenate(metadata["episode_index"], axis=0).astype(np.int64)
        if metadata["frame_index"]:
            self._frame_index = np.concatenate(metadata["frame_index"], axis=0).astype(np.int64)
        else:
            frame_index = np.zeros_like(self._episode_index)
            for episode in np.unique(self._episode_index):
                rows = np.nonzero(self._episode_index == episode)[0]
                frame_index[rows] = np.arange(len(rows), dtype=np.int64)
            self._frame_index = frame_index
        self._task_index = (
            np.concatenate(metadata["task_index"], axis=0).astype(np.int64) if metadata["task_index"] else None
        )
        self._timestamp = (
            np.concatenate(metadata["timestamp"], axis=0).astype(np.float32) if metadata["timestamp"] else None
        )

        self._episode_frame_to_row: dict[tuple[int, int], int] = {}
        self._episode_min_frame: dict[int, int] = {}
        self._episode_max_frame: dict[int, int] = {}
        for row, (episode, frame) in enumerate(zip(self._episode_index, self._frame_index, strict=True)):
            episode = int(episode)
            frame = int(frame)
            self._episode_frame_to_row[(episode, frame)] = row
            self._episode_min_frame[episode] = min(frame, self._episode_min_frame.get(episode, frame))
            self._episode_max_frame[episode] = max(frame, self._episode_max_frame.get(episode, frame))
        logging.info("Loaded local state/action dataset: %d frames.", len(self))

    def _row_for_offset(self, index: int, offset: int) -> int:
        episode = int(self._episode_index[index])
        frame = int(self._frame_index[index]) + int(offset)
        frame = min(max(frame, self._episode_min_frame[episode]), self._episode_max_frame[episode])
        return self._episode_frame_to_row[(episode, frame)]

    def __getitem__(self, index: SupportsIndex):
        index = int(index)
        state_rows = [self._row_for_offset(index, offset) for offset in self._state_offsets]
        action_rows = [self._row_for_offset(index, offset) for offset in self._action_offsets]
        sample = {
            "observation.state": self._states[np.asarray(state_rows, dtype=np.int64)],
            self._action_key: self._actions[np.asarray(action_rows, dtype=np.int64)],
            "observation.images.cam_front": self._dummy_image,
            "observation.images.cam_right": self._dummy_image,
            "observation.images.cam_left": self._dummy_image,
            "episode_index": int(self._episode_index[index]),
            "frame_index": int(self._frame_index[index]),
        }
        if self._task_index is not None:
            sample["task_index"] = int(self._task_index[index])
        if self._timestamp is not None:
            sample["timestamp"] = float(self._timestamp[index])
        return sample

    def __len__(self) -> int:
        return self._states.shape[0]


class StateActionOnlyDataset:
    """Fast LeRobot dataset wrapper that avoids video decoding.

    This wrapper is intentionally narrow: it reads only parquet-backed state,
    action, and metadata columns from `dataset.hf_dataset`. It is used by
    tactile-only encoder pretraining where RGB frames are dummy placeholders.
    """

    def __init__(
        self,
        dataset: lerobot_dataset.LeRobotDataset,
        *,
        state_delta_timestamps: Sequence[int],
        action_horizon: int,
        action_key: str,
    ):
        self._dataset = dataset
        self.hf_dataset = dataset.hf_dataset
        self._state_offsets = tuple(int(offset) for offset in state_delta_timestamps) or (0,)
        self._action_offsets = tuple(range(int(action_horizon)))
        self._action_key = action_key

        column_names = set(getattr(self.hf_dataset, "column_names", []))
        if "observation.state" not in column_names:
            raise ValueError("state_action_only requires `observation.state` in the dataset.")
        if action_key not in column_names:
            raise ValueError(f"state_action_only requires `{action_key}` in the dataset.")
        if "episode_index" not in column_names:
            raise ValueError("state_action_only requires `episode_index` in the dataset.")

        logging.info("Preloading state/action columns for state_action_only dataset.")
        self._states = np.stack(
            [_to_numpy_array(value).astype(np.float32) for value in self.hf_dataset["observation.state"]],
            axis=0,
        )
        self._actions = np.stack(
            [_to_numpy_array(value).astype(np.float32) for value in self.hf_dataset[action_key]],
            axis=0,
        )
        self._episode_index = _column_to_numpy(self.hf_dataset["episode_index"])
        if "frame_index" in column_names:
            self._frame_index = _column_to_numpy(self.hf_dataset["frame_index"])
        else:
            frame_index = np.zeros_like(self._episode_index)
            for episode in np.unique(self._episode_index):
                rows = np.nonzero(self._episode_index == episode)[0]
                frame_index[rows] = np.arange(len(rows), dtype=np.int64)
            self._frame_index = frame_index

        self._episode_frame_to_row: dict[tuple[int, int], int] = {}
        self._episode_min_frame: dict[int, int] = {}
        self._episode_max_frame: dict[int, int] = {}
        for row, (episode, frame) in enumerate(zip(self._episode_index, self._frame_index, strict=True)):
            episode = int(episode)
            frame = int(frame)
            self._episode_frame_to_row[(episode, frame)] = row
            self._episode_min_frame[episode] = min(frame, self._episode_min_frame.get(episode, frame))
            self._episode_max_frame[episode] = max(frame, self._episode_max_frame.get(episode, frame))

    def _row_for_offset(self, index: int, offset: int) -> int:
        episode = int(self._episode_index[index])
        frame = int(self._frame_index[index]) + int(offset)
        frame = min(max(frame, self._episode_min_frame[episode]), self._episode_max_frame[episode])
        return self._episode_frame_to_row[(episode, frame)]

    def __getitem__(self, index: SupportsIndex):
        index = int(index)
        state_rows = [self._row_for_offset(index, offset) for offset in self._state_offsets]
        action_rows = [self._row_for_offset(index, offset) for offset in self._action_offsets]
        sample = {
            "observation.state": self._states[np.asarray(state_rows, dtype=np.int64)],
            self._action_key: self._actions[np.asarray(action_rows, dtype=np.int64)],
        }
        column_names = set(getattr(self.hf_dataset, "column_names", []))
        for key in ("episode_index", "frame_index", "task_index", "timestamp"):
            if key in column_names:
                sample[key] = self.hf_dataset[index][key]
        return sample

    def __len__(self) -> int:
        return len(self.hf_dataset)


def _frame_indices_for_episodes(dataset: lerobot_dataset.LeRobotDataset, episodes: Sequence[int]) -> list[int]:
    selected = set(int(episode) for episode in episodes)
    episode_index = torch.stack(dataset.hf_dataset["episode_index"]).numpy()
    indices = np.nonzero(np.isin(episode_index, list(selected)))[0].astype(np.int64).tolist()
    if not indices:
        raise ValueError(f"Episode filter selected no frames. Requested episodes: {sorted(selected)[:20]}")
    return indices


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")
    
class DelayAwareDataset(Dataset):
    def __init__(
        self,
        dataset: lerobot_dataset.LeRobotDataset,
        max_delay: int,
        prob_zero: float = 0.5,  #  0延时的概率
        alpha: float = 1.0,
        beta: float = 3.0,
        image_keys: list[str] | None = None,
    ):
        self._dataset = dataset
        self._max_delay = max_delay
        self._prob_zero = prob_zero 
        self._dist = torch.distributions.Beta(alpha, beta)
        
        if image_keys is None:
            sample_keys = self._dataset[0].keys()
            self._image_keys = [k for k in sample_keys if "image" in k]
        else:
            self._image_keys = image_keys

    def __getitem__(self, index: int) -> dict:
        current_sample = self._dataset[index]

        # ---------------------------------------------------
        # 1. 先判定是否命中 "绝对 0 延时"
        if torch.rand(1).item() < self._prob_zero:
            delay_frac = 0.0
        else:
            # 2. 如果没命中，则从 Beta 分布采样剩余的延时
            # 注意：Beta 可能会采样出非常接近 0 的数，这没关系，代表"微小延时"
            delay_frac = self._dist.sample().item()
        # ---------------------------------------------------

        delay_steps = int(delay_frac * self._max_delay)
        
        delayed_index = index - delay_steps
        # ... (边界检查) ...
        actual_delay_steps = index - delayed_index
        
        if self._max_delay > 0:
            normalized_delay = actual_delay_steps / self._max_delay
        else:
            normalized_delay = 0.0

        current_sample["observation.delay"] = torch.tensor([normalized_delay], dtype=torch.float32)
        
        if delayed_index == index:
            return current_sample
            
        # 7. 读取旧数据并替换图像
        delayed_sample = self._dataset[delayed_index]

        for key in self._image_keys:
            if key in delayed_sample:
                current_sample[key] = delayed_sample[key]

        return current_sample

    def __len__(self) -> int:
        return len(self._dataset)
    


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def num_frames(self) -> int:
        return len(self._dataset.hf_dataset) if self._dataset.hf_dataset is not None else self._dataset.meta.total_frames


class EpisodeChunkSequenceDataset(Dataset):
    """Stacks ordered action-chunk anchors and pads episodes to one static length."""

    def __init__(self, dataset: Dataset, sequences: Sequence[Sequence[int]]):
        self._dataset = dataset
        self._sequences = [tuple(int(index) for index in sequence) for sequence in sequences]
        self._max_length = max(len(sequence) for sequence in self._sequences)

    def __getitem__(self, index: SupportsIndex):
        sequence = self._sequences[int(index)]
        items = [self._dataset[item_index] for item_index in sequence]
        mask = np.zeros((self._max_length,), dtype=np.float32)
        mask[: len(items)] = 1.0
        items.extend([items[-1]] * (self._max_length - len(items)))
        stacked = jax.tree.map(lambda *values: np.stack([np.asarray(value) for value in values]), *items)
        stacked["sequence_mask"] = mask
        return stacked

    def __len__(self) -> int:
        return len(self._sequences)


def _frame_metadata(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Read episode/frame columns without decoding videos."""
    if isinstance(dataset, TransformedDataset):
        return _frame_metadata(dataset._dataset)  # noqa: SLF001
    if isinstance(dataset, EpisodeSubsetDataset):
        episodes, frames = _frame_metadata(dataset._dataset)  # noqa: SLF001
        indices = np.asarray(dataset._indices, dtype=np.int64)  # noqa: SLF001
        return episodes[indices], frames[indices]
    if isinstance(dataset, LocalParquetStateActionOnlyDataset | StateActionOnlyDataset):
        return dataset._episode_index, dataset._frame_index  # noqa: SLF001
    if isinstance(dataset, lerobot_dataset.LeRobotDataset):
        episodes = _column_to_numpy(dataset.hf_dataset["episode_index"])
        if "frame_index" in set(dataset.hf_dataset.column_names):
            frames = _column_to_numpy(dataset.hf_dataset["frame_index"])
        else:
            frames = np.zeros_like(episodes)
            for episode in np.unique(episodes):
                rows = np.nonzero(episodes == episode)[0]
                frames[rows] = np.arange(len(rows), dtype=np.int64)
        return episodes, frames
    raise TypeError(f"Cannot obtain episode metadata from {type(dataset).__name__}.")


def _episode_chunk_sequences(dataset: Dataset, *, stride: int, action_horizon: int) -> list[list[int]]:
    episodes, frames = _frame_metadata(dataset)
    sequences = []
    for episode in np.unique(episodes):
        rows = np.nonzero(episodes == episode)[0]
        rows = rows[np.argsort(frames[rows])]
        episode_frames = frames[rows]
        last_anchor = int(episode_frames[-1]) - int(action_horizon) + 1
        anchors = rows[(episode_frames - int(episode_frames[0])) % int(stride) == 0]
        anchors = anchors[frames[anchors] <= last_anchor]
        if len(anchors) > 0:
            sequences.append(anchors.tolist())
    if not sequences:
        raise ValueError("No complete TactileTTT action chunks were found in the selected episodes.")
    return sequences


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    local_repo = pathlib.Path(repo_id).expanduser()
    if (local_repo / "meta" / "info.json").exists():
        repo_id = str(local_repo)

    episode_indices = _load_episode_indices(data_config.filter_dict_path)
    if data_config.state_action_only and local_repo.is_dir():
        if len(data_config.action_sequence_keys) != 1:
            raise ValueError("state_action_only currently supports exactly one action sequence key.")
        print(
            "[data_loader] FAST LOCAL PARQUET PATH: "
            f"repo={local_repo} action_horizon={action_horizon} "
            f"state_delta_timestamps={tuple(data_config.state_delta_timestamps)}",
            flush=True,
        )
        dataset = LocalParquetStateActionOnlyDataset(
            local_repo,
            state_delta_timestamps=data_config.state_delta_timestamps,
            action_horizon=action_horizon,
            action_key=data_config.action_sequence_keys[0],
        )
        if episode_indices is not None:
            selected = set(int(episode) for episode in episode_indices)
            frame_indices = np.nonzero(np.isin(dataset._episode_index, list(selected)))[0].astype(np.int64).tolist()
            logging.info(
                "Episode filter kept %d frames from %d episodes.",
                len(frame_indices),
                len(episode_indices),
            )
            dataset = EpisodeSubsetDataset(dataset, frame_indices)
        return dataset

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    delta_timestamps = {
        **{
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        }
    } # 因为需要预测未来 action_horizon 步的 action，因此此处要设置 delta_timestamps，而不只是读取1帧
    
    if "observation.effort" in dataset_meta.features:
        delta_timestamps["observation.effort"] = [t / dataset_meta.fps for t in data_config.effort_history] # 需要将过去 n 步 的 effort history 传入，注意，这里的 data_config.effort_history 必定是负的，例如,[-40,-36,-32...]

    if data_config.state_delta_timestamps:
        if "observation.state" not in dataset_meta.features:
            raise ValueError("state_delta_timestamps was set, but the dataset has no `observation.state` feature.")
        delta_timestamps["observation.state"] = [t / dataset_meta.fps for t in data_config.state_delta_timestamps]

    if model_config.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
          # 如果需要预测未来的 effort 的话，就需要多往后读取 action_horizon 个 effort，因此此时 effort 的区间：[-effort_history, +action_horizon]
          delta_timestamps["observation.effort"] += [(t + 1) / dataset_meta.fps for t in range(model_config.action_horizon)]

    if (
        isinstance(model_config, (pi0_config.Pi0LatentFlowConfig, pi0_config.Pi0SeerConfig))
        and model_config.use_future_rgb_instead_of_flow
    ):
        delta_timestamps["observation.images.head_camera"] = [
            0.0,
            model_config.future_rgb_step / dataset_meta.fps,
        ]
        delta_timestamps["observation.images.wrist_left_camera"] = [
            0.0,
            model_config.future_rgb_step / dataset_meta.fps,
        ]

    if episode_indices is not None:
        logging.info("Using %d filtered episodes from %s", len(episode_indices), data_config.filter_dict_path)

    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={} if data_config.state_action_only else delta_timestamps,
    )
    if data_config.state_action_only:
        if len(data_config.action_sequence_keys) != 1:
            raise ValueError("state_action_only currently supports exactly one action sequence key.")
        dataset = StateActionOnlyDataset(
            dataset,
            state_delta_timestamps=data_config.state_delta_timestamps,
            action_horizon=action_horizon,
            action_key=data_config.action_sequence_keys[0],
        )
    if episode_indices is not None:
        frame_indices = _frame_indices_for_episodes(dataset, episode_indices)
        logging.info(
            "Episode filter kept %d frames from %d episodes.",
            len(frame_indices),
            len(episode_indices),
        )
        dataset = EpisodeSubsetDataset(dataset, frame_indices)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    episode_sequences = None
    if bool(getattr(model_config, "tactile_ttt_enabled", False)):
        episode_sequences = _episode_chunk_sequences(
            dataset,
            stride=action_horizon,
            action_horizon=action_horizon,
        )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    if episode_sequences is not None:
        dataset = EpisodeChunkSequenceDataset(dataset, episode_sequences)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
