"""State banks: HDF5 files of flattened MuJoCo states of the steerability scene.

Layout (compatible with the paper-era files)::

    <task_name>_demo/                 one group per task
        demo_<i>_states  [T, D] f64   flattened sim states ``[time, qpos, qvel]``
    <task_name>_stages/demo_<i>       optional SteerGen stage labels (see resteer.steergen.label_stages)
    <task_name>_dist_to_go/demo_<i>   optional remaining end-effector path length
    attrs["num_warmup_states"]        leading states before the first policy step (policy-rollout banks: 10)

Banks come from raw LIBERO-Goal demonstrations (``build-from-raw``), from replaying them
(``resteer.steergen.regen_states``) or from policy rollouts (``resteer.cmi.collect_states``).

    python -m resteer.states build-from-raw --raw-dir data/libero_goal --out data/states/libero_goal_demos.hdf5
    python -m resteer.states merge --inputs a.hdf5 b.hdf5 --out merged.hdf5
    python -m resteer.states info data/states/libero_goal_demos.hdf5
"""

import dataclasses
import pathlib
import random
import re

import h5py
import numpy as np
import tyro

DEMO_SUFFIX = "_demo"


def _demo_index(key: str) -> int:
    match = re.match(r"demo_(\d+)", key)
    return int(match.group(1)) if match else -1


class StateBank:
    def __init__(self, path: "str | pathlib.Path"):
        self.path = pathlib.Path(path)
        with h5py.File(self.path, "r") as f:
            self._prefix = "data/" if "data" in f else ""
            root = f.get("data", f)
            self.task_names = [k[: -len(DEMO_SUFFIX)] for k in root if k.endswith(DEMO_SUFFIX)]
            self.num_warmup_states = int(f.attrs.get("num_warmup_states", 0))
        if not self.task_names:
            raise ValueError(f"{self.path} contains no '<task>{DEMO_SUFFIX}' groups")

    def demo_keys(self, task_name: str) -> list[str]:
        """Dataset keys ``demo_<i>_states`` in HDF5 (lexicographic) order."""
        with h5py.File(self.path, "r") as f:
            group = f[f"{self._prefix}{task_name}{DEMO_SUFFIX}"]
            return [k for k in group if k.startswith("demo_") and k.endswith("_states")]

    def load(self, task_name: str, demo_key: str) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f[f"{self._prefix}{task_name}{DEMO_SUFFIX}/{demo_key}"][:]

    def labels(self, task_name: str, demo_key: str, kind: str = "stages") -> np.ndarray | None:
        """SteerGen labels (``stages`` or ``dist_to_go``) for a demo, or None if the bank has none."""
        name = f"{self._prefix}{task_name}_{kind}/demo_{_demo_index(demo_key)}"
        with h5py.File(self.path, "r") as f:
            return f[name][:] if name in f else None

    def sample_uniform(
        self, task_name: str, num_steps: int = 100, stride: int = 5, offset: int | None = None, rng=random
    ) -> list[tuple[str, int, np.ndarray]]:
        """One state per step index ``offset, offset+stride, ..., offset+num_steps-stride``.

        ``offset`` defaults to the bank's number of warm-up states, so index ``offset + k`` is the
        state after ``k`` policy steps (the state at which evaluation would switch the prompt at
        step ``k``). For every index a random demo with ``len > index + offset`` is drawn with
        ``rng.choice`` (paper behavior, including the conservative length check).
        """
        offset = self.num_warmup_states if offset is None else offset
        with h5py.File(self.path, "r") as f:
            group = f[f"{self._prefix}{task_name}{DEMO_SUFFIX}"]
            demos = {k: group[k][:] for k in group if k.startswith("demo_") and k.endswith("_states")}
        samples = []
        for index in range(offset, num_steps + offset, stride):
            valid = [k for k, states in demos.items() if len(states) > index + offset]
            if not valid:
                continue
            key = rng.choice(valid)
            samples.append((key, index, demos[key][index]))
        return samples


def build_from_raw(raw_dir: "str | pathlib.Path", out: "str | pathlib.Path") -> None:
    """Collects ``data/demo_<i>/states`` of every raw LIBERO demo file into a state bank."""
    files = sorted(pathlib.Path(raw_dir).glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files in {raw_dir}")
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as dst:
        total = 0
        for path in files:
            with h5py.File(path, "r") as src:
                demos = sorted((k for k in src["data"] if k.startswith("demo_")), key=_demo_index)
                group = dst.create_group(path.stem)
                for key in demos:
                    group.create_dataset(f"{key}_states", data=src[f"data/{key}/states"][:], compression="gzip")
                group.attrs["source_file"] = path.name
                group.attrs["num_demos"] = len(demos)
                total += len(demos)
            print(f"{path.stem}: {len(demos)} demos")
        dst.attrs["num_warmup_states"] = 0
        dst.attrs["created_by"] = "resteer.states build-from-raw"
    print(f"Wrote {total} demos to {out}")


def write_task_demos(
    path: "str | pathlib.Path",
    task_name: str,
    demos: list[np.ndarray],
    num_warmup_states: int = 0,
    extra: dict[str, list[np.ndarray]] | None = None,
) -> None:
    """Writes (or replaces) one task's demos as ``<task>_demo/demo_<i>_states`` in a bank file.

    ``extra`` adds per-demo arrays stored next to the states, e.g. ``{"act": [...]}`` ->
    ``demo_<i>_act``.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        name = f"{task_name}{DEMO_SUFFIX}"
        if name in f:
            del f[name]
        group = f.create_group(name)
        for i, states in enumerate(demos):
            group.create_dataset(f"demo_{i}_states", data=np.asarray(states, dtype=np.float64), compression="gzip")
            for key, values in (extra or {}).items():
                group.create_dataset(f"demo_{i}_{key}", data=np.asarray(values[i]), compression="gzip")
        group.attrs["num_demos"] = len(demos)
        f.attrs["num_warmup_states"] = num_warmup_states


def merge(inputs: list[pathlib.Path], out: "str | pathlib.Path") -> None:
    """Merges banks (e.g. per-task rollout banks); groups already present are skipped."""
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as dst:
        for path in inputs:
            with h5py.File(path, "r") as src:
                for key, value in src.attrs.items():
                    if key not in dst.attrs:
                        dst.attrs[key] = value
                for key in src:
                    if key in dst:
                        print(f"skip duplicate group {key} from {path}")
                        continue
                    src.copy(src[key], dst, name=key)
    print(f"Merged {len(inputs)} files into {out}")


def info(path: "str | pathlib.Path") -> None:
    bank = StateBank(path)
    print(f"{path}: {len(bank.task_names)} tasks, num_warmup_states={bank.num_warmup_states}")
    for name in bank.task_names:
        lengths = [len(bank.load(name, k)) for k in bank.demo_keys(name)]
        print(f"  {name}: {len(lengths)} demos, length {min(lengths)}-{max(lengths)}")


@dataclasses.dataclass
class BuildFromRaw:
    raw_dir: pathlib.Path
    out: pathlib.Path


@dataclasses.dataclass
class Merge:
    inputs: list[pathlib.Path]
    out: pathlib.Path


@dataclasses.dataclass
class Info:
    path: tyro.conf.Positional[pathlib.Path]


def main() -> None:
    cmd = tyro.extras.subcommand_cli_from_dict({"build-from-raw": BuildFromRaw, "merge": Merge, "info": Info})
    if isinstance(cmd, BuildFromRaw):
        build_from_raw(cmd.raw_dir, cmd.out)
    elif isinstance(cmd, Merge):
        merge(cmd.inputs, cmd.out)
    else:
        info(cmd.path)


if __name__ == "__main__":
    main()
