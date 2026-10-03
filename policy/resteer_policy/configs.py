"""ReSteer training configs, derived from upstream openpi's ``pi05_libero``.

Model, optimizer and learning-rate schedule are exactly those of ``pi05_libero``: pi0.5 with
action horizon 10, AdamW (grad clip 1.0), cosine schedule with 10k warmup steps to 5e-5, EMA
0.999. Note that with 2k steps the learning rate is still in warmup when training ends; this matches
the paper's runs.

Normalization statistics are those of Physical Intelligence's released ``pi05_libero`` checkpoint
and are saved under ``assets/physical-intelligence/libero`` in every checkpoint, so trained
checkpoints are served with the stock config: ``serve_policy.py policy:checkpoint
--policy.config=pi05_libero --policy.dir=<checkpoint step dir>``.

Dataset repo ids are LeRobot ids, resolved under ``$HF_LEROBOT_HOME`` or on the Hugging Face hub.
Override them on the command line (``--data.repo-ids a b c``) to train on your own generated data.
"""

import dataclasses

from openpi.training import config as _config
from openpi.training import weight_loaders

from resteer_policy.multidata import MultiLeRobotLiberoDataConfig

LIBERO = "physical-intelligence/libero"
STEERGEN_DATA = "resteer/libero_goal_steergen"
SRBC_DATA = "resteer/libero_goal_srbc"

PI05_LIBERO_NORM_STATS = _config.AssetsConfig(
    assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets", asset_id="physical-intelligence/libero"
)


def _resteer_config(name: str, repo_ids, weights, init_params: str, **overrides) -> _config.TrainConfig:
    base = _config.get_config("pi05_libero")
    return dataclasses.replace(
        base,
        name=name,
        data=MultiLeRobotLiberoDataConfig(
            repo_id=repo_ids[0],  # required by openpi's CLI; the data comes from repo_ids
            repo_ids=tuple(repo_ids),
            dataset_weights=tuple(weights),
            assets=PI05_LIBERO_NORM_STATS,
            base_config=_config.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(init_params),
        **overrides,
    )


CONFIGS = [
    # SteerGen: co-train the released pi05_libero policy on LIBERO demonstrations plus SteerGen
    # steering segments (weights 1:3).
    _resteer_config(
        "pi05_libero_resteer_steergen",
        repo_ids=[LIBERO, STEERGEN_DATA],
        weights=[1, 3],
        init_params="gs://openpi-assets/checkpoints/pi05_libero/params",
        num_train_steps=2_000,
        save_interval=2_000,
        keep_period=2_000,
    ),
    # SRBC: continue from the SteerGen policy, adding successful task-switching rollouts collected
    # with it (weights 1:3:15). Point --weight-loader.params-path at your SteerGen checkpoint.
    _resteer_config(
        "pi05_libero_resteer_srbc",
        repo_ids=[LIBERO, STEERGEN_DATA, SRBC_DATA],
        weights=[1, 3, 15],
        init_params="checkpoints/pi05_libero_resteer_steergen/steergen/1999/params",
        num_train_steps=10_000,
        save_interval=1_000,
        keep_period=4_000,
    ),
]
