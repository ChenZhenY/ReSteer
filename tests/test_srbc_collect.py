"""SRBC collection logic (resteer.srbc.collect) with a fake environment and policy.

The fake environment draws its "object layout" from NumPy's global RNG at every reset, like LIBERO, and
encodes (layout, env step) in its observations, so the tests can check which observation each frame
recorded and which layouts a (resumed) collection used.
"""

import h5py
import numpy as np
import pytest

from resteer import tasks
from resteer.data import episodes
from resteer.sim import libero as sim
from resteer.srbc import collect
from resteer.srbc import select

SOURCE, TARGET = tasks.get_task("turn_on_the_stove"), tasks.get_task("put_the_bowl_on_the_plate")


class FakeEnv:
    """The target goal holds after every step iff ``succeeds(layout)``; every reset is logged."""

    resets: list[int] = []

    def __init__(self, succeeds):
        self.env, self.succeeds, self.layout, self.t = self, succeeds, -1, 0

    def reset(self):
        self.layout, self.t = int(np.random.randint(1_000_000)), 0
        FakeEnv.resets.append(self.layout)

    def modify_observable(self, name, attribute, value):  # camera rendering toggle (no-op here)
        pass

    def _get_observations(self, force_update=False):
        image = np.full((8, 8, 3), self.t % 256, np.uint8)
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image,
            "robot0_eef_pos": np.array([self.layout, self.t, 0.0]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def step(self, action, return_success_dict=False):
        self.t += 1
        return self._get_observations(), 0.0, {TARGET.goal_keys[0]: self.succeeds(self.layout)}, {}

    def close(self):
        pass


class FakeClient:
    def __init__(self):
        self.prompts = []

    def infer(self, obs):
        self.prompts.append(obs["prompt"])
        return {"actions": np.ones((10, 7))}


@pytest.fixture
def fake_sim(monkeypatch):
    FakeEnv.resets = []
    rule = {"succeeds": lambda layout: True}

    def make_env(resolution=sim.RENDER_RESOLUTION, seed=None):
        if seed is not None:
            np.random.seed(seed)  # what LIBERO's env.seed does
        return FakeEnv(lambda layout: rule["succeeds"](layout))

    monkeypatch.setattr(sim, "make_env", make_env)
    return rule


def _args(tmp_path, **kwargs):
    return collect.Args(source="success_rate", results=tmp_path, out=tmp_path / "out", **kwargs)


def _config(k, **kwargs):
    return select.SwitchConfig(SOURCE.name, TARGET.name, k, **kwargs)


def _episodes(path):
    return list(episodes.read_episodes(path))


def test_default_mode_records_fresh_frames_and_lower_case_segments(tmp_path, fake_sim):
    args = _args(tmp_path, successes_per_config=2, max_attempts_per_config=3)
    client = FakeClient()
    collect.collect_task(args, SOURCE.name, [_config(3, success_rate=0.5)], client)

    eps = _episodes(args.out / f"{SOURCE.name}.hdf5")
    assert len(eps) == 2 and len(FakeEnv.resets) == 2  # stops after 2 successes; one env for all attempts
    name, arrays, segments, source, meta = eps[0]
    # The target goal already holds, so the rollout succeeds at the first step after the switch.
    assert len(arrays["actions"]) == 5 and meta["first_success_step"] == 4
    # Fresh frames: the observation of every step (after the 10 warm-up steps).
    np.testing.assert_array_equal(arrays["state"][:, 1], 10 + np.arange(5))
    assert [(s.start, s.end, s.task, s.prompt) for s in segments] == [
        (0, 3, SOURCE.name, SOURCE.prompt),
        (3, 5, TARGET.name, TARGET.prompt),
    ]
    assert source == "srbc" and meta["switch_step"] == 3 and meta["success_rate"] == 0.5
    assert arrays["agentview"].shape == (5, episodes.IMAGE_SIZE, episodes.IMAGE_SIZE, 3)
    assert set(client.prompts) == {SOURCE.prompt, TARGET.prompt}  # the policy always sees lower case


def test_paper_compat_new_env_per_rollout_stale_frames_title_labels(tmp_path, fake_sim):
    args = _args(tmp_path, paper_compat=True, successes_per_config=2, max_attempts_per_config=3)
    collect.collect_task(args, SOURCE.name, [_config(7), _config(0)], FakeClient())

    eps = _episodes(args.out / f"{SOURCE.name}.hdf5")
    assert len(eps) == 4
    assert len(set(FakeEnv.resets)) == 1  # every rollout re-seeds: the same layout every time
    _, arrays, segments, _, meta = eps[0]
    # Stale frames: the observation of the latest policy query (steps 0 and 5, then 7 after the switch);
    # no warm-up steps.
    np.testing.assert_array_equal(arrays["state"][:, 1], [0, 0, 0, 0, 0, 5, 5, 7, 7])
    assert [(s.start, s.end, s.prompt) for s in segments] == [(0, 7, SOURCE.title), (7, 9, TARGET.title)]
    assert meta["paper_compat"] and meta["max_steps"] == collect.PAPER_MAX_STEPS
    _, arrays, segments, _, _ = eps[2]  # switch at step 0: one segment
    assert [(s.start, s.end, s.task) for s in segments] == [(0, 2, TARGET.name)]


def test_attempt_limits_and_failed_rollouts(tmp_path, fake_sim):
    fake_sim["succeeds"] = lambda layout: False
    args = _args(tmp_path, max_attempts_per_config=4, max_steps=12)
    collect.collect_task(args, SOURCE.name, [_config(2), _config(5)], FakeClient())
    assert not (args.out / f"{SOURCE.name}.hdf5").exists() or not _episodes(args.out / f"{SOURCE.name}.hdf5")
    assert len(FakeEnv.resets) == 8

    FakeEnv.resets = []
    args = _args(tmp_path / "b", max_attempts_per_config=4, max_steps=12, max_total_rollouts=6)
    collect.collect_task(args, SOURCE.name, [_config(2), _config(5)], FakeClient())
    assert len(FakeEnv.resets) == 6  # max_total_rollouts stops the second config early


def test_resume_is_exact(tmp_path, fake_sim):
    fake_sim["succeeds"] = lambda layout: layout % 3 == 0
    configs = [_config(2), _config(4), _config(6)]
    kwargs = dict(successes_per_config=2, max_attempts_per_config=6, max_steps=10)

    full = _args(tmp_path / "full", **kwargs)
    collect.collect_task(full, SOURCE.name, configs, FakeClient())
    full_resets = FakeEnv.resets

    FakeEnv.resets = []
    part = _args(tmp_path / "part", max_total_rollouts=5, **kwargs)
    collect.collect_task(part, SOURCE.name, configs, FakeClient())  # interrupted after 5 rollouts
    resumed = _args(tmp_path / "part", **kwargs)
    collect.collect_task(resumed, SOURCE.name, configs, FakeClient())
    assert FakeEnv.resets == full_resets

    got, want = _episodes(part.out / f"{SOURCE.name}.hdf5"), _episodes(full.out / f"{SOURCE.name}.hdf5")
    assert len(got) == len(want) > 0
    for (n1, a1, s1, _, m1), (n2, a2, s2, _, m2) in zip(got, want):
        assert n1 == n2 and s1 == s2 and m1 == m2
        np.testing.assert_array_equal(a1["state"], a2["state"])

    FakeEnv.resets = []
    collect.collect_task(resumed, SOURCE.name, configs, FakeClient())  # everything done: no rollouts
    assert FakeEnv.resets == []


def test_resume_after_crash_mid_write(tmp_path, fake_sim):
    args = _args(tmp_path, successes_per_config=3, max_attempts_per_config=5)
    collect.collect_task(args, SOURCE.name, [_config(3)], FakeClient())
    path = args.out / f"{SOURCE.name}.hdf5"
    with h5py.File(path, "a") as f:  # a crash inside EpisodeWriter.write leaves an episode without metadata
        del f["episodes/episode_00002"].attrs["meta"]
    assert collect._saved_successes(path) == {_config(3).key: 2}
    collect.collect_task(args, SOURCE.name, [_config(3)], FakeClient())
    assert len(_episodes(path)) == 3
