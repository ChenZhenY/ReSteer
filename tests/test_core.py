"""Unit tests for the simulation-independent parts of resteer."""

import json
import socket
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

from resteer import tasks
from resteer import utils
from resteer.cmi import compute_cmi
from resteer.cmi import kde
from resteer.data import episodes
from resteer.eval import score


def test_task_table():
    assert [t.id for t in tasks.TASKS] == list(range(10))
    assert [t.name for t in tasks.TASKS] == sorted(t.name for t in tasks.TASKS)  # ids are alphabetical
    task = tasks.get_task("Put The Bowl On The Plate")
    assert task is tasks.get_task("put_the_bowl_on_the_plate") is tasks.get_task(3)
    assert task.prompt == "put the bowl on the plate" and task.title == "Put The Bowl On The Plate"
    assert tasks.is_success(task, {"on_akita_black_bowl_1_plate_1": True})
    assert not tasks.is_success(task, {"on_akita_black_bowl_1_plate_1": False, "turnon_flat_stove_1": True})
    assert sorted(tasks.LEGACY_STEERGEN_ORDER) == [t.name for t in tasks.TASKS]
    with pytest.raises(KeyError):
        tasks.get_task("fly to the moon")


def _reference_kde(x: np.ndarray) -> float:
    """Float64 re-derivation of the estimator, for comparison."""
    n, d = x.shape
    h = np.std(x, axis=0, ddof=1).mean() * n ** (-1 / (d + 4))
    sq = ((x[:, None] - x[None]) ** 2).sum(-1)
    return float(-np.log(np.exp(-sq / (2 * h**2)).mean(1) + 1e-8).mean())


def test_kde_entropy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    assert kde.kde_entropy(x) == pytest.approx(_reference_kde(x.astype(np.float64)), rel=1e-5)
    # Invariant to the scale of the data (bandwidth follows the spread), bounded by log N.
    assert kde.kde_entropy(10 * x) == pytest.approx(kde.kde_entropy(x), rel=1e-4)
    assert 0 <= kde.kde_entropy(x) <= np.log(32)
    # A tight cluster plus far outliers has lower entropy than a well-spread set.
    clustered = np.concatenate([np.zeros((30, 3)), 100 * np.ones((2, 3))]) + 1e-3 * rng.normal(size=(32, 3))
    assert kde.kde_entropy(clustered) < kde.kde_entropy(x)


def _write_step(root, source, k, outcomes):
    rollouts = [{"target": t, "repeat": r, "success": s} for t, ss in outcomes.items() for r, s in enumerate(ss)]
    task = tasks.get_task(source)
    utils.write_json(
        root / f"task_{task.id}_{task.name}" / f"step_{k:03d}.json",
        {"source_task": task.name, "switch_step": k, "rollouts": rollouts},
    )


def test_score_excludes_same_task(tmp_path):
    a, b = tasks.TASKS[0].name, tasks.TASKS[1].name
    _write_step(tmp_path, a, 0, {a: [1, 1], b: [1, 0]})
    _write_step(tmp_path, a, 5, {a: [1, 1], b: [0, 0]})
    _write_step(tmp_path, b, 0, {a: [0, 0], b: [1, 1]})
    summary = score.summarize(score.load_cells(tmp_path))
    assert summary["steerability_score"] == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    assert summary["score_incl_same_task"] == pytest.approx((1 + 0.5 + 1 + 0 + 0 + 1) / 6)
    assert summary["per_task"][a]["score"] == pytest.approx(0.25)
    assert summary["score_by_switch_step"][5] == pytest.approx(0.0)


def test_compute_cmi(tmp_path):
    source = tasks.TASKS[2]
    for index, h_s in [(10, 2.0), (15, 1.0)]:
        entropies = {t.name: h_s - 0.1 * t.id for t in tasks.TASKS}
        utils.write_json(tmp_path / f"task_2_{source.name}" / f"state_{index:03d}.json", {
            "source_task": source.name, "demo": "demo_0_states", "state_index": index, "policy_step": index - 10,
            "state_entropy": h_s, "prompt_entropy": entropies})  # fmt: skip
    records = compute_cmi.load_records(tmp_path, normalize=False)[source.name]
    assert len(records) == 20
    assert records[1]["cmi"] == pytest.approx(0.1) and records[0]["cmi"] == 0.0
    assert compute_cmi.load_records(tmp_path, normalize=True)[source.name][11]["cmi"] == pytest.approx(0.1)
    args = compute_cmi.Args(tmp_path, normalize=False, low_cmi_percent=10)
    low = compute_cmi.select_low_cmi(records, args)
    assert len(low) == 1 and low[0]["target_task"] != source.name  # same-task cells excluded by default
    assert low[0]["policy_step"] == 0 and low[0]["new_prompt_step"] == 10
    paper = compute_cmi.select_low_cmi(records, compute_cmi.Args(tmp_path, low_cmi_percent=10, paper_compat=True))
    assert paper[0]["target_task"] == tasks.TASKS[0].name  # min CMI (0) among all cells, stable order


def test_sample_actions_explains_rollouts_too_short(tmp_path):
    from resteer import states
    from resteer.cmi import sample_actions

    task = tasks.get_task(3)
    bank_path = tmp_path / "bank.hdf5"
    # reset + 10 warm-up states + 5 policy steps: too short for the first sampled index (10 needs > 20 states)
    states.write_task_demos(bank_path, task.name, [np.zeros((16, 9))], num_warmup_states=10)
    args = sample_actions.Args(states=bank_path, out=tmp_path / "out")
    with pytest.raises(SystemExit, match="long enough"):
        sample_actions.sample_task(args, task, states.StateBank(bank_path), client=None)


def test_episode_roundtrip(tmp_path):
    builder = episodes.EpisodeBuilder()
    for t in range(4):
        builder.add(action=np.full(7, t), agentview=np.zeros((512, 512, 3), np.uint8),
                    wrist=np.zeros((224, 224, 3), np.uint8), state=np.arange(8))  # fmt: skip
    segments = [episodes.Segment(0, 2, "turn_on_the_stove", "turn on the stove"),
                episodes.Segment(2, 4, "put_the_bowl_on_the_plate", "put the bowl on the plate")]  # fmt: skip
    with episodes.EpisodeWriter(tmp_path / "e.hdf5") as w:
        w.write("ep0", builder, segments, "srbc", {"switch_step": 2})
        with pytest.raises(ValueError):
            w.write("bad", builder, segments[:1], "srbc", {})
    [(name, arrays, segs, source, meta)] = list(episodes.read_episodes(tmp_path / "e.hdf5"))
    assert name == "ep0" and source == "srbc" and meta == {"switch_step": 2} and segs == segments
    assert arrays["agentview"].shape == (4, 256, 256, 3) and arrays["wrist"].shape == (4, 256, 256, 3)
    assert arrays["actions"].dtype == np.float32 and arrays["actions"][3, 0] == 3


def test_episode_writer_drops_cut_short_episode(tmp_path):
    builder = episodes.EpisodeBuilder()
    builder.add(action=np.zeros(7), agentview=np.zeros((256, 256, 3), np.uint8),
                wrist=np.zeros((256, 256, 3), np.uint8), state=np.zeros(8))  # fmt: skip
    segments = [episodes.Segment(0, 1, "turn_on_the_stove", "turn on the stove")]
    with episodes.EpisodeWriter(tmp_path / "e.hdf5") as w:
        w.write("ep0", builder, segments, "srbc", {})
    with h5py.File(tmp_path / "e.hdf5", "a") as f:  # what a crash in the middle of write() leaves behind
        f["episodes"].create_group("ep1").create_dataset("actions", data=np.zeros((1, 7)))
    with episodes.EpisodeWriter(tmp_path / "e.hdf5") as w:
        assert len(w) == 1
        w.write("ep1", builder, segments, "srbc", {})
    assert [name for name, *_ in episodes.read_episodes(tmp_path / "e.hdf5")] == ["ep0", "ep1"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("batch", [False, True])
def test_policy_client(tmp_path, batch):
    from resteer.client import PolicyClient

    port, log = _free_port(), tmp_path / "requests.txt"
    cmd = [sys.executable, "tests/fake_policy_server.py", "--port", str(port), "--log", str(log)]
    server = subprocess.Popen(cmd + (["--batch"] if batch else []))
    try:
        client = PolicyClient("127.0.0.1", port, connect_timeout_s=30)
        assert client.supports_batch == batch
        obs = {"observation/state": np.zeros(8), "prompt": "turn on the stove"}
        out = client.infer(obs)
        assert out["actions"].shape == (10, 7)
        samples = client.sample_actions(obs, num_samples=5, batch_size=2)
        assert samples.shape == (5, 10, 7)
        assert np.array_equal(samples[0], out["actions"])  # the fake server is deterministic
        client.close()
        assert len(log.read_text().splitlines()) == 6
    finally:
        server.terminate()
        server.wait()


def test_write_json_is_atomic(tmp_path):
    utils.write_json(tmp_path / "a" / "b.json", {"x": np.float32(1.5), "y": np.arange(2)})
    assert json.loads((tmp_path / "a" / "b.json").read_text()) == {"x": 1.5, "y": [0, 1]}
    assert not list((tmp_path / "a").glob(".*.tmp"))
    rng_state = np.random.RandomState(3).get_state()
    decoded = utils.decode_rng_state(json.loads(json.dumps(utils.encode_rng_state(rng_state))))
    np.random.set_state(decoded)
    reference = np.random.RandomState(3).random(4)
    assert np.array_equal(np.random.random(4), reference)


def test_connect_timeout():
    from resteer.client import PolicyClient
    from resteer.client import PolicyServerError

    start = time.monotonic()
    with pytest.raises(PolicyServerError):
        PolicyClient("127.0.0.1", _free_port(), connect_timeout_s=0.1)
    assert time.monotonic() - start < 30
