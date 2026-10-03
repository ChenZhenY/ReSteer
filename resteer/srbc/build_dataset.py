"""Index, sample and combine SRBC episode files into one training set.

The paper mixed two pools of SRBC episodes: collected from success-rate-selected switches (db1) and
from low-CMI switches (db2). ``sample`` draws ``total_demos`` episodes with a fraction ``ratio``
from db1, using one of three strategies:
- ``random``: uniform over each database;
- ``stratified``: per source task, proportional to that task's share of the pooled episodes;
- ``balanced``: an equal share per source task.
Arithmetic and seeding (``random.Random(42)``) match the paper-era sampler.

    python -m resteer.srbc.build_dataset index --data-dir data/srbc/success_rate
    python -m resteer.srbc.build_dataset sample --db1 data/srbc/success_rate/database.json \
        --db2 data/srbc/cmi/database.json --total-demos 2000 --ratio 1.0 --out data/srbc/sampled.json
    python -m resteer.srbc.build_dataset combine --database data/srbc/sampled.json --out data/srbc/srbc.hdf5

Database JSON: ``{"tasks": {source_task: {"num_demos", "demos": [{"file", "episode", ...}]}},
"statistics": {...}}``. ``file`` paths are relative to the JSON file.
"""

import dataclasses
import json
import os
import pathlib
import random
from typing import Literal

import h5py
import tyro

from resteer import utils
from resteer.data import episodes as _episodes

DATABASE_NAME = "database.json"


def _statistics(tasks: dict) -> dict:
    return {
        "total_demos": sum(t["num_demos"] for t in tasks.values()),
        "num_tasks": len(tasks),
        "demos_per_task": {name: t["num_demos"] for name, t in tasks.items()},
    }


def index(data_dir: pathlib.Path, out: pathlib.Path | None = None) -> dict:
    """Lists every episode below ``data_dir``, grouped by source task (sorted by name)."""
    out = out or data_dir / DATABASE_NAME
    demos: dict[str, list[dict]] = {}
    for path in sorted(data_dir.rglob("*.hdf5")):
        with h5py.File(path, "r") as f:
            if f.attrs.get("format") != _episodes.FORMAT:
                continue
            for name, group in f["episodes"].items():
                meta = json.loads(group.attrs["meta"])
                segments = json.loads(group.attrs["segments"])
                source = meta.get("source_task") or segments[0]["task"]
                demos.setdefault(source, []).append(
                    {
                        "file": os.path.relpath(path, out.parent),
                        "episode": name,
                        "target_task": meta.get("target_task"),
                        "switch_step": meta.get("switch_step"),
                        "config_source": meta.get("config_source"),
                        "num_frames": int(group["actions"].shape[0]),
                    }
                )
    tasks = {name: {"num_demos": len(demos[name]), "demos": demos[name]} for name in sorted(demos)}
    database = {"tasks": tasks, "statistics": _statistics(tasks)}
    utils.write_json(out, database)
    print(f"Indexed {database['statistics']['total_demos']} episodes of {len(tasks)} source tasks -> {out}")
    return database


def load_database(path: pathlib.Path) -> dict:
    """Loads a database; demo ``file`` entries become absolute paths."""
    database = utils.read_json(path)
    for task in database["tasks"].values():
        for demo in task["demos"]:
            demo["file"] = str((path.parent / demo["file"]).resolve())
    return database


def _all_demos(database: dict) -> list[tuple[str, dict]]:
    return [(task, demo) for task, info in database["tasks"].items() for demo in info["demos"]]


def _demos_by_task(database: dict) -> dict[str, list[dict]]:
    return {task: list(info["demos"]) for task, info in database["tasks"].items()}


def _take(rng: random.Random, task: str, pool: list, n: int, db: int) -> list:
    return [(task, demo, db) for demo in rng.sample(pool, n)] if n > 0 else []


def sample_random(db1, db2, total, ratio, rng):
    demos1, demos2 = _all_demos(db1), _all_demos(db2)
    n1 = int(total * ratio)
    n2 = total - n1
    n1, n2 = min(n1, len(demos1)), min(n2, len(demos2))
    return [(t, d, 1) for t, d in rng.sample(demos1, n1)], [(t, d, 2) for t, d in rng.sample(demos2, n2)]


def sample_stratified(db1, db2, total, ratio, rng):
    by1, by2 = _demos_by_task(db1), _demos_by_task(db2)
    tasks = set(by1) | set(by2)
    per_task = {t: len(by1.get(t, [])) + len(by2.get(t, [])) for t in tasks}
    grand_total = sum(per_task.values())
    if grand_total == 0:
        return [], []
    out1, out2 = [], []
    for task in sorted(tasks):
        n_task = int(total * (per_task[task] / grand_total))
        if n_task == 0:
            continue
        pool1, pool2 = by1.get(task, []), by2.get(task, [])
        n1 = int(n_task * ratio)
        n2 = n_task - n1
        out1 += _take(rng, task, pool1, min(n1, len(pool1)), 1)
        out2 += _take(rng, task, pool2, min(n2, len(pool2)), 2)
    return out1, out2


def sample_balanced(db1, db2, total, ratio, rng):
    by1, by2 = _demos_by_task(db1), _demos_by_task(db2)
    tasks = set(by1) | set(by2)
    if not tasks:
        return [], []
    per_task = total // len(tasks)
    out1, out2 = [], []
    for task in sorted(tasks):
        pool1, pool2 = by1.get(task, []), by2.get(task, [])
        n1 = int(per_task * ratio)
        n2 = per_task - n1
        out1 += _take(rng, task, pool1, min(n1, len(pool1)), 1)
        out2 += _take(rng, task, pool2, min(n2, len(pool2)), 2)
    return out1, out2


STRATEGIES = {"random": sample_random, "stratified": sample_stratified, "balanced": sample_balanced}


def sample(
    db1_path: pathlib.Path,
    db2_path: pathlib.Path,
    out: pathlib.Path,
    total_demos: int | None = None,
    ratio: float = 0.5,
    strategy: Literal["random", "stratified", "balanced"] = "random",
    seed: int = 42,
) -> dict:
    db1, db2 = load_database(db1_path), load_database(db2_path)
    available = db1["statistics"]["total_demos"] + db2["statistics"]["total_demos"]
    total = min(total_demos or available, available)
    sampled1, sampled2 = STRATEGIES[strategy](db1, db2, total, ratio, random.Random(seed))
    tasks: dict[str, dict] = {}
    for task, demo, _ in sampled1 + sampled2:
        entry = tasks.setdefault(task, {"num_demos": 0, "demos": []})
        entry["demos"].append({**demo, "file": os.path.relpath(demo["file"], out.parent)})
        entry["num_demos"] += 1
    database = {
        "tasks": tasks,
        "statistics": _statistics(tasks),
        "sampling_info": {
            "db1": str(db1_path),
            "db2": str(db2_path),
            "strategy": strategy,
            "ratio": ratio,
            "seed": seed,
            "requested_total": total,
            "demos_from_db1": len(sampled1),
            "demos_from_db2": len(sampled2),
        },
    }
    utils.write_json(out, database)
    print(f"Sampled {len(sampled1)} episodes from db1 and {len(sampled2)} from db2 ({strategy}) -> {out}")
    return database


def combine(database_path: pathlib.Path, out: pathlib.Path, links: bool = False) -> None:
    """Copies the episodes of a database into one file, renamed 000000, 000001, ... (sorted by source).

    With ``links``, the file holds HDF5 external links to the episodes instead of copies (no extra
    disk space; the source files must stay in place)."""
    database = load_database(database_path)
    entries = sorted((d["file"], d["episode"]) for _, d in _all_demos(database))
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as dst:
        dst.attrs["format"] = _episodes.FORMAT
        dst.attrs["version"] = _episodes.VERSION
        root = dst.create_group("episodes")
        for i, (file, episode) in enumerate(entries):
            if links:
                root[f"{i:06d}"] = h5py.ExternalLink(str(pathlib.Path(file).resolve()), f"episodes/{episode}")
                continue
            with h5py.File(file, "r") as src:
                name = f"{i:06d}"
                src.copy(src[f"episodes/{episode}"], root, name=name)
                meta = json.loads(root[name].attrs["meta"])
                meta["origin"] = {"file": os.path.relpath(file, out.parent), "episode": episode}
                root[name].attrs["meta"] = json.dumps(meta)
    print(f"Combined {len(entries)} episodes -> {out}")


@dataclasses.dataclass
class Index:
    data_dir: pathlib.Path
    out: pathlib.Path | None = None
    """Default: <data-dir>/database.json"""


@dataclasses.dataclass
class Sample:
    db1: pathlib.Path
    db2: pathlib.Path
    out: pathlib.Path
    total_demos: int | None = None
    """Default: all episodes of both databases."""
    ratio: float = 0.5
    """Fraction drawn from db1."""
    strategy: Literal["random", "stratified", "balanced"] = "random"
    seed: int = 42


@dataclasses.dataclass
class Combine:
    database: pathlib.Path
    out: pathlib.Path
    links: bool = False
    """Store external links to the episodes instead of copies (the source files must stay in place)."""


def main() -> None:
    cmd = tyro.extras.subcommand_cli_from_dict({"index": Index, "sample": Sample, "combine": Combine})
    if isinstance(cmd, Index):
        index(cmd.data_dir, cmd.out)
    elif isinstance(cmd, Sample):
        sample(cmd.db1, cmd.db2, cmd.out, cmd.total_demos, cmd.ratio, cmd.strategy, cmd.seed)
    else:
        combine(cmd.database, cmd.out, cmd.links)


if __name__ == "__main__":
    main()
