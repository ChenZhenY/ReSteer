"""Train a ReSteer policy with upstream openpi's training loop.

Runs inside openpi's environment (third_party/openpi). Accepts the ReSteer configs
(:mod:`resteer_policy.configs`) and every upstream openpi config, with the usual openpi flags:

    uv run --project third_party/openpi python -m resteer_policy.train pi05_libero_resteer_steergen \\
        --exp-name steergen --fsdp-devices 2 --batch-size 64

(``scripts/train.sh`` sets up ``PYTHONPATH`` for this.)
"""

import importlib.util
import pathlib

import openpi
from openpi.training import config as _config
import tyro

from resteer_policy import configs as _resteer_configs
from resteer_policy import multidata


def _openpi_train_module():
    path = pathlib.Path(openpi.__file__).resolve().parents[2] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    multidata.install()
    configs = {**_config._CONFIGS_DICT, **{c.name: c for c in _resteer_configs.CONFIGS}}  # noqa: SLF001
    config = tyro.extras.overridable_config_cli({name: (name, c) for name, c in configs.items()})
    _openpi_train_module().main(config)


if __name__ == "__main__":
    main()
