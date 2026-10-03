"""Regenerates a state bank by replaying the raw LIBERO-Goal demonstrations in the steerability scene.

For each demo: reset to its first recorded state, settle for 10 zero-action steps, then replay its
actions with LIBERO's controller (``replay_delta``), skipping no-op actions. The bank stores the
state after settling (index 0) and after every replayed action, so bank index ``t`` is the state
after ``t`` non-no-op demo actions; ``stage_matched`` generation replays the demo the same way to
reach a start state online. ``act`` and ``qacc_warmstart`` are stored next to the states.

Simulator states are MuJoCo-version specific; the version is recorded in the file attributes.
Fixture placement (baked into each env's XML at construction) comes from NumPy's global RNG, which
is seeded with ``--seed``; envs are built in the order of ``--tasks``.

    python -m resteer.steergen.regen_states --raw-dir data/libero_raw/libero_goal \
        --out data/states/libero_goal_regen.hdf5 [--tasks put_the_bowl_on_the_plate ...] [--max-demos 5]
"""

import dataclasses
import logging
import pathlib

import h5py
import numpy as np
import tqdm
import tyro

from resteer import states as _states
from resteer import tasks as _tasks
from resteer import utils
from resteer.steergen import interp

logger = logging.getLogger(__name__)
NUM_SETTLE_STEPS = 10


def settle(env, num_steps: int = NUM_SETTLE_STEPS) -> None:
    """Zero-motion steps with a neutral gripper command, as done before replaying LIBERO demos."""
    for _ in range(num_steps):
        env.step_delta(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), 0.0)
        env.observe()


def replay(env, actions: np.ndarray, max_steps: int | None = None, on_step=None) -> int:
    """Replays non-no-op LIBERO actions (at most ``max_steps``); returns the number replayed."""
    kept = []
    for action in actions:
        if max_steps is not None and len(kept) >= max_steps:
            break
        if interp.is_noop(action, kept[-1] if kept else None):
            continue
        kept.append(action)
        env.observe()
        env.step_delta(action[:3], interp.axis_angle_to_quat(action[3:6]), action[6])
        if on_step is not None:
            on_step()
    return len(kept)


def regen_demo(env, raw_actions: np.ndarray, raw_states: np.ndarray):
    env.reset_to(raw_states[0])
    settle(env)
    states, acts, qaccs = [], [], []

    def record():
        states.append(env.get_state())
        acts.append(env.sim.data.act.copy())
        qaccs.append(env.sim.data.qacc_warmstart.copy())

    record()
    replay(env, raw_actions, on_step=record)
    return states, acts, qaccs


@dataclasses.dataclass
class Args:
    raw_dir: pathlib.Path
    """Directory with the raw LIBERO-Goal files ``<task>_demo.hdf5``."""
    out: pathlib.Path
    tasks: tuple[str, ...] = ()
    """Task ids or names to replay (default: all 10), in this order."""
    max_demos: int | None = None
    camera_size: int = 256
    seed: int = 0
    log_level: str = "INFO"


def main(args: Args) -> None:
    import mujoco

    from resteer.steergen.env import SteerGenEnv

    utils.setup_logging(args.log_level)
    np.random.seed(args.seed)
    names = [_tasks.get_task(int(t) if t.isdigit() else t).name for t in args.tasks] or [t.name for t in _tasks.TASKS]
    written = 0
    for name in names:
        path = args.raw_dir / f"{name}_demo.hdf5"
        if not path.exists():
            logger.warning("No raw demos for %s at %s, skipping", name, path)
            continue
        env = SteerGenEnv("replay_delta", args.camera_size)
        demos, acts, qaccs = [], [], []
        with h5py.File(path, "r") as f:
            num = len(f["data"]) if args.max_demos is None else min(args.max_demos, len(f["data"]))
            for i in tqdm.trange(num, desc=name):
                data = f[f"data/demo_{i}"]
                s, a, q = regen_demo(env, data["actions"][()], data["states"][()])
                demos.append(np.stack(s))
                acts.append(np.stack(a))
                qaccs.append(np.stack(q))
        env.close()
        _states.write_task_demos(args.out, name, demos, extra={"act": acts, "qacc": qaccs})
        written += 1
        logger.info("%s: %d demos -> %s", name, len(demos), args.out)
    if not written:
        return
    with h5py.File(args.out, "a") as f:
        f.attrs["mujoco_version"] = mujoco.__version__
        f.attrs["created_by"] = "resteer.steergen.regen_states"


if __name__ == "__main__":
    main(tyro.cli(Args))
