"""Collects a state bank of the states a policy visits while following each task (CMI's state set).

For every task: ``num_rollouts`` rollouts from fresh random scenes. The flattened MuJoCo state is
recorded after ``reset``, after each of the 10 zero-action warm-up steps, and after every policy
step (the policy follows the task's own instruction; no early stop, as in the paper). Index
``10 + k`` of a rollout is therefore the state at which evaluation would switch the prompt at step
``k``; the bank stores ``num_warmup_states = 10`` so that samplers apply this offset.

One file per task is written to ``--out-dir`` (tasks can run in parallel); merge them with
``python -m resteer.states merge``.

    python -m resteer.cmi.collect_states --port 8000 --out-dir data/states/pi05_libero --tasks 0 1
"""

import collections
import dataclasses
import logging
import pathlib

import imageio
import numpy as np
import tyro

from resteer import states as _states
from resteer import tasks as _tasks
from resteer import utils
from resteer.client import PolicyClient
from resteer.sim import libero as sim

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path
    host: str = "127.0.0.1"
    port: int = 8000
    tasks: tuple[int, ...] = tuple(range(10))
    num_rollouts: int = 10
    max_steps: int = 400
    replan_steps: int = 5
    seed: int = 7
    stop_on_success: bool = False
    """Stop a rollout once the task succeeds (the paper kept rolling out for max_steps)."""
    save_video: bool = False
    log_level: str = "INFO"


def rollout_states(env, client: PolicyClient, task: _tasks.Task, args: Args, frames: list | None = None) -> np.ndarray:
    env.reset()
    render = frames is not None  # otherwise cameras are rendered only for policy queries
    sim.set_camera_rendering(env, render)
    states = [env.get_sim_state()]
    for _ in range(sim.NUM_WARMUP_STEPS):
        env.step(np.zeros(sim.ACTION_DIM).tolist())
        states.append(env.get_sim_state())
    plan: collections.deque = collections.deque()
    for _ in range(args.max_steps):
        if not plan:
            sim.set_camera_rendering(env, True)
            query = sim.policy_input(sim.get_observation(env), task.prompt)
            sim.set_camera_rendering(env, render)
            plan.extend(client.infer(query)["actions"][: args.replan_steps])
        obs, success_dict = sim.step(env, plan.popleft())
        states.append(env.get_sim_state())
        if frames is not None:
            frames.append(sim.camera_images(obs)[0])
        if args.stop_on_success and _tasks.is_success(task, success_dict):
            break
    return np.asarray(states)


def main(args: Args) -> None:
    utils.setup_logging(args.log_level)
    client = PolicyClient(args.host, args.port)
    for task_id in args.tasks:
        task = _tasks.get_task(task_id)
        out = args.out_dir / f"rollouts_{task.name}.hdf5"
        if out.exists():
            logger.info("%s exists, skipping", out)
            continue
        env = sim.make_env(seed=args.seed)
        demos = []
        try:
            for i in range(args.num_rollouts):
                frames = [] if args.save_video else None
                demos.append(rollout_states(env, client, task, args, frames))
                logger.info("%s rollout %d: %d states", task.name, i, len(demos[-1]))
                if frames:
                    video = args.out_dir / "videos" / f"{task.name}_{i}.mp4"
                    video.parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimwrite(video, frames, fps=30)
        finally:
            env.close()
        tmp = out.with_suffix(".tmp")
        _states.write_task_demos(tmp, task.name, demos, num_warmup_states=sim.NUM_WARMUP_STEPS)
        tmp.rename(out)
    client.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
