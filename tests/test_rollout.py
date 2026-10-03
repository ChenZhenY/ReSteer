"""Prompt-switching logic of resteer.eval.rollout, with a fake environment and policy."""

import numpy as np
import pytest

from resteer import tasks
from resteer.eval import rollout

SOURCE, TARGET = tasks.get_task("turn_on_the_stove"), tasks.get_task("put_the_bowl_on_the_plate")


def _obs():
    return {
        "agentview_image": np.zeros((8, 8, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((8, 8, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }


class FakeEnv:
    """Target goal holds from step `goal_from` on; the source goal from step 0 on."""

    def __init__(self, goal_from: int):
        self.goal_from, self.t = goal_from, 0
        self.env = self

    def modify_observable(self, name, attribute, value):  # camera rendering toggle (no-op here)
        pass

    def _get_observations(self, force_update=False):
        return _obs()

    def step(self, action, return_success_dict=False):
        done = {SOURCE.goal_keys[0]: True, TARGET.goal_keys[0]: self.t >= self.goal_from}
        self.t += 1
        return _obs(), 0.0, done, {}


class FakeClient:
    def __init__(self):
        self.prompts = []

    def infer(self, obs):
        self.prompts.append(obs["prompt"])
        return {"actions": np.zeros((10, 7))}


def test_success_counts_only_after_switch():
    client = FakeClient()
    result = rollout.run_switch_rollout(FakeEnv(goal_from=0), client, SOURCE, TARGET, switch_step=7, max_steps=20)
    # The source goal holds throughout but never counts; the target goal counts at step 8 (> k).
    assert result.success and result.first_success_step == 8 and result.steps_taken == 8
    # Queries at steps 0 and 5 use the source prompt; the switch clears the plan and requeries at 7.
    assert client.prompts == [SOURCE.prompt, SOURCE.prompt, TARGET.prompt]


def test_switch_at_zero_and_failure():
    client = FakeClient()
    result = rollout.run_switch_rollout(FakeEnv(goal_from=100), client, SOURCE, TARGET, switch_step=0, max_steps=12)
    assert not result.success and result.first_success_step is None and result.steps_taken == 12
    assert set(client.prompts) == {TARGET.prompt}  # the source instruction is never used when k = 0


class _Crash(Exception):
    pass


def _fake_eval(monkeypatch, crash_after: int | None):
    """evaluate_task with a fake scene: layouts come from NumPy's global RNG (as in LIBERO), outcomes from layouts."""
    from resteer.eval import steerability
    from resteer.sim import libero as sim

    count = {"rollouts": 0}

    class Env:
        layout = None

        def close(self):
            pass

    def make_env(seed=None, **_):
        np.random.seed(seed)
        return Env()

    def reset(env, **_):
        env.layout = np.random.rand()

    def run_switch_rollout(env, client, task, target, k, **_):
        if crash_after is not None and count["rollouts"] == crash_after:
            raise _Crash
        count["rollouts"] += 1
        success = bool(env.layout > 0.5)
        return rollout.RolloutResult(success, 1 if success else None, 1 if success else 300)

    monkeypatch.setattr(sim, "make_env", make_env)
    monkeypatch.setattr(sim, "reset", reset)
    monkeypatch.setattr(rollout, "run_switch_rollout", run_switch_rollout)
    monkeypatch.setattr(steerability.utils, "git_revision", lambda: "test")
    return steerability


def _rollouts(out, task):
    from resteer import utils

    return {p.name: utils.read_json(p)["rollouts"] for p in sorted(out.glob(f"task_{task.id}_*/step_*.json"))}


def test_eval_resumes_mid_step_exactly(tmp_path, monkeypatch):
    client = type("Client", (), {"metadata": {}})()
    for scope in ("task", "step"):
        args = dict(tasks=(SOURCE.id,), switch_steps=(0, 5), targets=(3, 9), num_repeats=4, seed_scope=scope)
        steerability = _fake_eval(monkeypatch, crash_after=None)
        reference = steerability.Args(out=tmp_path / scope / "reference", **args)
        steerability.evaluate_task(reference, SOURCE, client)

        resumed = steerability.Args(out=tmp_path / scope / "resumed", **args)
        for crash_after in (3, 6):  # the 4th rollout of switch step 0, then the 7th rollout after resuming
            steerability = _fake_eval(monkeypatch, crash_after=crash_after)
            with pytest.raises(_Crash):
                steerability.evaluate_task(resumed, SOURCE, client)
        steerability = _fake_eval(monkeypatch, crash_after=None)
        steerability.evaluate_task(resumed, SOURCE, client)

        assert _rollouts(reference.out, SOURCE) == _rollouts(resumed.out, SOURCE)
        assert len(_rollouts(resumed.out, SOURCE)["step_005.json"]) == 8
        assert not list(resumed.out.glob("**/*.partial"))
