"""One prompt-switching rollout: the unit of the steerability protocol (and of SRBC data collection).

Starting from a freshly reset scene, the policy follows the source task's instruction; at step
``switch_step`` the instruction is replaced by the target task's and the queued actions are
discarded. The rollout succeeds if the *target* task's goal holds after any step ``> switch_step``.
With ``switch_step=0`` the whole rollout runs under the target instruction.
"""

import collections
from collections.abc import Callable
import dataclasses

import numpy as np

from resteer import tasks as _tasks
from resteer.client import PolicyClient
from resteer.sim import libero as sim


@dataclasses.dataclass
class RolloutResult:
    success: bool
    # Step index after which the target goal first held (None if it never did).
    first_success_step: int | None
    # Paper-era field: number of steps completed before the success step (== first_success_step on success).
    steps_taken: int
    error: str | None = None


# Called once per executed action: (step, obs_before_action, last_policy_input, action, prompt).
StepCallback = Callable[[int, dict, dict, np.ndarray, str], None]


def run_switch_rollout(
    env,
    client: PolicyClient,
    source_task: "_tasks.Task",
    target_task: "_tasks.Task",
    switch_step: int,
    *,
    max_steps: int = 300,
    replan_steps: int = 5,
    prompt_style: _tasks.PromptStyle = "lower",
    on_step: StepCallback | None = None,
    render_every_step: bool = False,
) -> RolloutResult:
    """Runs one rollout on an env that has already been reset (and warmed up).

    Cameras are rendered only when an observation is taken: when the policy is queried (every
    ``replan_steps`` steps), or every step if ``render_every_step`` is set or ``on_step`` needs every
    frame; never inside ``env.step``. This leaves the policy inputs and the rollout unchanged:
    rendering does not affect the simulation.

    Exceptions from the environment are recorded in ``error``; ``PolicyServerError`` propagates so
    that the caller can stop instead of silently counting server failures as task failures.
    """
    render_every_step = render_every_step or on_step is not None
    sim.set_camera_rendering(env, False)  # never render inside env.step; only when observing
    action_plan: collections.deque = collections.deque()
    task = source_task
    query = obs = None
    result = RolloutResult(success=False, first_success_step=None, steps_taken=0)
    for step in range(max_steps):
        if step == switch_step:
            task = target_task
            action_plan.clear()
        prompt = _tasks.format_prompt(task, prompt_style)
        if render_every_step or not action_plan:
            sim.set_camera_rendering(env, True)
            obs = sim.get_observation(env)
            sim.set_camera_rendering(env, False)
        if not action_plan:
            query = sim.policy_input(obs, prompt)
            action_plan.extend(client.infer(query)["actions"][:replan_steps])
        action = np.asarray(action_plan.popleft())
        if on_step is not None:
            on_step(step, obs, query, action, prompt)
        try:
            _, success_dict = sim.step(env, action)
        except Exception as e:  # noqa: BLE001 - recorded, matching the paper's failure accounting
            result.error = f"{type(e).__name__}: {e}"
            return result
        if step > switch_step and _tasks.is_success(task, success_dict):
            result.success = True
            result.first_success_step = step
            return result
        result.steps_taken = step + 1
    return result
