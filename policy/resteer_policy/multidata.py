"""Weighted co-training on several LeRobot datasets with unmodified upstream openpi.

openpi builds its training dataset in a single function,
``openpi.training.data_loader.create_torch_dataset``. :func:`install` wraps it so that configs using
:class:`MultiLeRobotLiberoDataConfig` get the concatenation of several LeRobot datasets, dataset
``i`` repeated ``w_i`` times. Each dataset keeps its own task table (for ``prompt_from_task``) and
shares the LIBERO transforms and normalization statistics.

With shuffling, a frame of dataset ``i`` is drawn with probability ``w_i / sum_j w_j N_j`` — the
same per-frame weighting as the paper's ``WeightedRandomSampler`` (in expectation; the paper
sampled with replacement). Note that the mixture is therefore proportional to ``w_i * N_i``, not to
``w_i`` alone.
"""

import dataclasses
import logging
import pathlib
from typing import Any

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from torch.utils.data import ConcatDataset
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class MultiDataConfig(_config.DataConfig):
    repo_ids: tuple[str, ...] = ()
    dataset_weights: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class MultiLeRobotLiberoDataConfig(_config.LeRobotLiberoDataConfig):
    """LIBERO data config over several LeRobot datasets (``repo_id`` is ignored)."""

    repo_ids: tuple[str, ...] = ()
    # Integer number of repetitions of each dataset (see module docstring).
    dataset_weights: tuple[int, ...] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> MultiDataConfig:
        if not self.repo_ids:
            raise ValueError("repo_ids must be non-empty")
        weights = self.dataset_weights or (1,) * len(self.repo_ids)
        if len(weights) != len(self.repo_ids) or any(int(w) != w or w < 1 for w in weights):
            raise ValueError(f"dataset_weights {weights} must be positive integers, one per repo id")
        base = super().create(assets_dirs, model_config)
        fields: dict[str, Any] = {f.name: getattr(base, f.name) for f in dataclasses.fields(_config.DataConfig)}
        fields["repo_id"] = self.repo_ids[0]
        return MultiDataConfig(**fields, repo_ids=tuple(self.repo_ids), dataset_weights=tuple(int(w) for w in weights))


_upstream_create_torch_dataset = _data_loader.create_torch_dataset


def create_torch_dataset(data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig):
    if not isinstance(data_config, MultiDataConfig):
        return _upstream_create_torch_dataset(data_config, action_horizon, model_config)
    parts = []
    for repo_id, weight in zip(data_config.repo_ids, data_config.dataset_weights, strict=True):
        single = dataclasses.replace(data_config, repo_id=repo_id)
        dataset = _upstream_create_torch_dataset(single, action_horizon, model_config)
        parts.extend([dataset] * weight)
        logging.info("Co-training dataset %s: %d frames x%d", repo_id, len(dataset), weight)
    return ConcatDataset(parts)


def install() -> None:
    """Routes openpi's dataset creation through :func:`create_torch_dataset` (idempotent)."""
    if not hasattr(_data_loader, "create_torch_dataset"):
        raise RuntimeError(
            "openpi.training.data_loader.create_torch_dataset not found; is openpi at the pinned commit?"
        )
    _data_loader.create_torch_dataset = create_torch_dataset
