import json
import random

import h5py
import numpy as np
import pytest

from resteer.data import episodes
from resteer.srbc import build_dataset


def _write_task_file(path, source, target, n):
    with episodes.EpisodeWriter(path) as writer:
        for i in range(n):
            episode = episodes.EpisodeBuilder()
            for _ in range(3):
                episode.add(
                    action=np.full(7, i, np.float32),
                    agentview=np.zeros((8, 8, 3), np.uint8),
                    wrist=np.zeros((8, 8, 3), np.uint8),
                    state=np.zeros(8),
                )
            segments = [episodes.Segment(0, 1, source, source), episodes.Segment(1, 3, target, target)]
            meta = {"source_task": source, "target_task": target, "switch_step": 1, "config_source": "cmi"}
            writer.write(f"episode_{i:05d}", episode, segments, "srbc", meta)


@pytest.fixture
def two_databases(tmp_path):
    for pool, sizes in (("db1", {"task_a": 7, "task_b": 3}), ("db2", {"task_a": 2, "task_c": 5})):
        for task, n in sizes.items():
            _write_task_file(tmp_path / pool / f"{task}.hdf5", task, "task_z", n)
        build_dataset.index(tmp_path / pool)
    return tmp_path / "db1" / "database.json", tmp_path / "db2" / "database.json"


# Reference: the paper-era utils/hdf5_database_sampler.py (verbatim arithmetic).
def _reference(strategy, db1, db2, total, ratio, rng):
    def all_demos(db):
        return [(t, d) for t, info in db["tasks"].items() for d in info["demos"]]

    def by_task(db):
        return {t: info["demos"].copy() for t, info in db["tasks"].items()}

    if strategy == "random":
        d1, d2 = all_demos(db1), all_demos(db2)
        n1 = int(total * ratio)
        n2 = total - n1
        n1, n2 = min(n1, len(d1)), min(n2, len(d2))
        return [(t, d, 1) for t, d in rng.sample(d1, n1)], [(t, d, 2) for t, d in rng.sample(d2, n2)]
    b1, b2 = by_task(db1), by_task(db2)
    all_tasks = set(b1.keys()) | set(b2.keys())
    out1, out2 = [], []
    if strategy == "stratified":
        per = {t: len(b1.get(t, [])) + len(b2.get(t, [])) for t in all_tasks}
        grand = sum(per.values())
    for task in sorted(all_tasks):
        n_task = int(total * (per[task] / grand)) if strategy == "stratified" else total // len(all_tasks)
        if n_task == 0:
            continue
        p1, p2 = b1.get(task, []), b2.get(task, [])
        n1 = int(n_task * ratio)
        n2 = n_task - n1
        n1, n2 = min(n1, len(p1)), min(n2, len(p2))
        if n1 > 0:
            out1.extend([(task, d, 1) for d in rng.sample(p1, n1)])
        if n2 > 0:
            out2.extend([(task, d, 2) for d in rng.sample(p2, n2)])
    return out1, out2


@pytest.mark.parametrize("strategy", ["random", "stratified", "balanced"])
@pytest.mark.parametrize(("total", "ratio"), [(10, 0.5), (12, 0.75), (7, 1.0), (17, 0.0)])
def test_sampling_matches_paper_era_sampler(two_databases, strategy, total, ratio):
    db1_path, db2_path = two_databases
    db1, db2 = build_dataset.load_database(db1_path), build_dataset.load_database(db2_path)
    fn = build_dataset.STRATEGIES[strategy]
    got = fn(db1, db2, total, ratio, random.Random(42))
    want = _reference(strategy, db1, db2, total, ratio, random.Random(42))
    key = lambda s: [(t, d["file"], d["episode"], src) for t, d, src in s]  # noqa: E731
    assert key(got[0]) == key(want[0]) and key(got[1]) == key(want[1])


def test_index_sample_combine(two_databases, tmp_path):
    db1_path, db2_path = two_databases
    db1 = json.loads(db1_path.read_text())
    assert db1["statistics"] == {"total_demos": 10, "num_tasks": 2, "demos_per_task": {"task_a": 7, "task_b": 3}}
    assert db1["tasks"]["task_a"]["demos"][0]["file"] == "task_a.hdf5"  # relative to the database

    out = tmp_path / "mix" / "sampled.json"
    sampled = build_dataset.sample(db1_path, db2_path, out, total_demos=8, ratio=0.5, strategy="balanced")
    # balanced: 8 // 3 tasks = 2 per task, split 1/1 between the pools; pools without the task give 0.
    assert (sampled["sampling_info"]["demos_from_db1"], sampled["sampling_info"]["demos_from_db2"]) == (2, 2)
    combined = tmp_path / "mix" / "combined.hdf5"
    build_dataset.combine(out, combined)
    names = [name for name, *_ in episodes.read_episodes(combined)]
    assert names == [f"{i:06d}" for i in range(4)]
    with h5py.File(combined) as f:
        origin = json.loads(f["episodes/000000"].attrs["meta"])["origin"]
        assert origin["file"].startswith("..")
