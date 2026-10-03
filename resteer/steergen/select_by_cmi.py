"""CMI-guided selection of SteerGen bridges.

Keeps the bridges that start near the switch states where the policy is least steerable, i.e. with
the lowest conditional mutual information ``I(A; L | s)`` (see ``resteer.cmi.compute_cmi``). Each
low-CMI tuple ``(source instruction, target instruction, step)`` selects the bridges of that task
pair whose start step is within ``--window`` of the tuple's step. The ``--max-tuples`` lowest-CMI
tuples over all input files are used. ``--random-keep N`` instead keeps N random bridges (the
data-efficiency baseline; the paper kept 5 per tuple).

Step conventions: a bridge's ``step`` is an index into the state bank it was generated from
(demo states: steps since the demo started). CMI tuples carry ``new_prompt_step`` (an index into the
CMI state bank, which for policy-rollout banks includes 10 warm-up states) and ``policy_step`` (policy
steps since the warm-up). ``--step-field policy_step`` (default) compares steps since the start of
the episode; use ``new_prompt_step`` to reproduce paper-era selections on banks without warm-up.

The inputs are never modified: the selection is copied into ``--out``, with a manifest next to it.

    python -m resteer.steergen.select_by_cmi --episodes data/steergen/bridges.hdf5 \
        --cmi results/cmi/pi05_libero --max-tuples 300 --out data/steergen/bridges_low_cmi.hdf5
"""

import dataclasses
import json
import pathlib
import random
from typing import Literal

import h5py
import tyro

from resteer import tasks as _tasks
from resteer import utils

TUPLE_KEYS = ("low_cmi_tuples", "low_mutual_info_tuples")


def load_tuples(paths: tuple[pathlib.Path, ...]) -> list[dict]:
    """Low-CMI tuples from JSON files, or from every ``*.json`` below the given directories.

    ``compute_cmi`` writes per-task files and an aggregate file with the same tuples, so duplicates
    are dropped.
    """
    files = []
    for path in paths:
        files += sorted(path.rglob("*.json")) if path.is_dir() else [path]
    tuples, seen = [], set()
    for file in files:
        if file.name.startswith("old_"):
            continue
        data = utils.read_json(file)
        key = next((k for k in TUPLE_KEYS if isinstance(data, dict) and k in data), None)
        for t in data[key] if key is not None else []:
            identity = (t["old_prompt"], t["new_prompt"], t["new_prompt_step"], t.get("demo_name"), cmi_value(t))
            if identity not in seen:
                seen.add(identity)
                tuples.append(t)
    return tuples


def cmi_value(t: dict) -> float:
    return t["cmi"] if "cmi" in t else t["mutual_info"]


def select(bridges: dict[str, dict], tuples: list[dict], max_tuples, window: int, step_field: str):
    """Names of the bridges (``{name: meta}``) matched by the lowest-CMI tuples, plus the tuples used."""
    tuples = sorted(tuples, key=cmi_value)[:max_tuples]
    by_pair: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for name, meta in bridges.items():
        by_pair.setdefault((meta["source_task"], meta["target_task"]), []).append((name, int(meta["step"])))
    keep = set()
    for t in tuples:
        pair = (_tasks.get_task(t.get("source_task", t["old_prompt"])).name,
                _tasks.get_task(t.get("target_task", t["new_prompt"])).name)  # fmt: skip
        step = int(t.get(step_field, t["new_prompt_step"]))
        keep.update(name for name, s in by_pair.get(pair, []) if abs(s - step) <= window)
    return keep, tuples


def read_bridge_meta(paths: tuple[pathlib.Path, ...]) -> dict[str, tuple[pathlib.Path, dict]]:
    bridges = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            for name, group in f["episodes"].items():
                if name in bridges:
                    raise ValueError(f"Episode {name} appears in several input files")
                bridges[name] = (path, json.loads(group.attrs["meta"]))
    return bridges


@dataclasses.dataclass
class Args:
    episodes: tuple[pathlib.Path, ...]
    """SteerGen episode files to select from."""
    out: pathlib.Path
    cmi: tuple[pathlib.Path, ...] = ()
    """Low-CMI tuple JSON files, or directories searched recursively."""
    max_tuples: int | None = None
    window: int = 2
    step_field: Literal["policy_step", "new_prompt_step"] = "policy_step"
    random_keep: int | None = None
    """Keep this many random bridges instead of using CMI."""
    seed: int = 0


def main(args: Args) -> None:
    bridges = read_bridge_meta(args.episodes)
    if args.random_keep is not None:
        names = sorted(bridges)
        keep = set(random.Random(args.seed).sample(names, min(args.random_keep, len(names))))
        tuples = []
    else:
        if not args.cmi:
            raise SystemExit("Pass --cmi (low-CMI tuple files) or --random-keep")
        keep, tuples = select({n: m for n, (_, m) in bridges.items()}, load_tuples(args.cmi), args.max_tuples,
                              args.window, args.step_field)  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as dst:
        with h5py.File(args.episodes[0], "r") as src:
            for key in ("format", "version"):
                dst.attrs[key] = src.attrs[key]
        root = dst.create_group("episodes")
        for name in sorted(keep):
            with h5py.File(bridges[name][0], "r") as src:
                src.copy(src[f"episodes/{name}"], root, name=name)
    manifest = {
        "kept": sorted(keep),
        "num_candidates": len(bridges),
        "tuples": tuples,
        "args": dataclasses.asdict(args),
    }
    utils.write_json(args.out.with_suffix(".manifest.json"), manifest)
    print(f"Kept {len(keep)} of {len(bridges)} bridges ({len(tuples)} tuples) -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
