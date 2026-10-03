"""SteerGen: synthesize steering bridges between LIBERO-Goal tasks.

A *bridge* starts from a state of source task A and moves the end effector (straight-line position,
SLERP orientation, closed-loop OSC tracking) to the end-effector pose of a state of target task B.
It is labelled with B's instruction. Co-trained with the original demonstrations, which show how to
finish B from such poses, bridges teach a policy to switch tasks mid-execution.

Modes:
- ``step_matched`` (the paper's datasets): for every step ``t`` (default 0-49, the pre-grasp phase
  of LIBERO-Goal demos), take the states at step ``t`` of a random demo of A and of B. The gripper
  is held open. A bridge is kept if the final 7-D end-effector pose (position + wxyz quaternion)
  is within ``--success-threshold`` (default 0.05) of the target. Bridges are executed with the
  identity-scaled controller in metres/radians and stored in LIBERO's action space
  (x20 position, x2 rotation), as the paper's post-processing did.
- ``stage_matched`` (experimental): only for start states in the transport stage (object in hand)
  and targets that transport the same object (the bowl tasks; the wine-bottle tasks). The target
  state is the same-stage state of B that minimizes ``dist_to_go + |eef_B - eef_A|`` among up to
  ``--max-candidates`` random candidates. With ``--raw-dir`` the source demo is replayed online up
  to ``t`` so the grasp is physically established. The gripper command keeps the start state's
  open/closed state. A bridge is kept if the final position error is below ``--success-threshold``
  (default 0.1 m). Needs a bank from ``regen_states`` labelled by ``label_stages``.

Randomness: ``random.Random(--seed)`` picks demos and candidates. NumPy's global RNG, seeded once
with ``--seed``, fixes the fixture placement of each env; one env is built per task pair, in
source-major order. Bridges draw no random numbers and start from a freshly loaded simulator, so a rerun
with the same arguments resumes: bridges already in ``--out`` are kept (the random draws before them are
replayed) and the result equals that of an uninterrupted run.

    python -m resteer.steergen.generate --states data/states/libero_goal_demos.hdf5 \
        --out data/steergen/bridges.hdf5
"""

import dataclasses
import logging
import pathlib
import random
import time
from typing import Literal

import h5py
import numpy as np
import tyro

from resteer import states as _states
from resteer import tasks as _tasks
from resteer import utils
from resteer.data import episodes
from resteer.steergen import controller
from resteer.steergen import interp
from resteer.steergen import label_stages

logger = logging.getLogger(__name__)

# Tasks that carry the same object, between which a transport-stage bridge is meaningful.
TRANSPORT_GROUPS = (
    {
        "open_the_top_drawer_and_put_the_bowl_inside",
        "put_the_bowl_on_top_of_the_cabinet",
        "put_the_bowl_on_the_plate",
        "put_the_bowl_on_the_stove",
    },
    {"put_the_wine_bottle_on_top_of_the_cabinet", "put_the_wine_bottle_on_the_rack"},
)


def task_name(key: str) -> str:
    """Task name from a CLI value: a task id (\"3\"), a task name or an instruction."""
    return _tasks.get_task(int(key) if key.isdigit() else key).name


def transportable(source: str, target: str) -> bool:
    return source != target and any(source in group and target in group for group in TRANSPORT_GROUPS)


@dataclasses.dataclass
class Args:
    states: pathlib.Path
    """State bank (``resteer.states build-from-raw`` for step_matched; regen + labels for stage_matched)."""
    out: pathlib.Path
    """Episode file to append the bridges to."""
    mode: Literal["step_matched", "stage_matched"] = "step_matched"
    source_tasks: tuple[str, ...] = ()
    """Source task ids or names (default: all 10)."""
    target_tasks: tuple[str, ...] = ()
    """Target task ids or names (default: all 10); pairs with source == target are skipped."""
    steps: tuple[int, ...] | None = None
    """Start steps (bank indices). Default: 0-49 (step_matched), 0, 5, ..., 95 (stage_matched)."""
    seed: int = 0
    source_demo: int | None = None
    """Use this demo of the source task instead of a random one (debugging / parity tests)."""
    target_demo: int | None = None
    """Use this demo of the target task instead of a random one (step_matched only)."""
    max_xyz_delta: float = 0.005
    max_rot_delta: float = 0.05
    success_threshold: float | None = None
    clip_size: int | None = None
    """Keep only the first waypoints of every bridge."""
    prompt_style: _tasks.PromptStyle = "lower"
    """Instruction label: lower case (as sent to the policy) or the paper-era Title Case."""
    camera_size: int = 256
    raw_dir: pathlib.Path | None = None
    """stage_matched: raw LIBERO-Goal demos, to replay the source demo online up to the start step."""
    max_candidates: int = 50
    gripper_close_threshold: float = 0.03
    log_level: str = "INFO"


def _pose(env) -> np.ndarray:
    return interp.pose_from_matrix(env.eef_pose())


def step_matched_bridge(env, start_state, end_state, args: Args):
    """Paper-era bridge (collect_interpolation_data.py); returns (obs, sim_states, actions, error)."""
    env.reset_to(start_state)
    env.observe(force_update=True)
    start = _pose(env)
    env.reset_to(end_state)
    env.observe(force_update=True)
    end = _pose(env)
    waypoints = interp.interpolate_poses(start, end, args.max_xyz_delta, args.max_rot_delta)[: args.clip_size]
    env.reset_to(start_state)
    env.observe(force_update=True)
    observations, sim_states, actions = [env.observe(force_update=True)], [env.get_state()], []
    current = start
    for waypoint in waypoints:
        delta_pos, delta_quat = interp.delta_to(current, waypoint)
        actions.append(env.step_delta(delta_pos, delta_quat, -1.0) * controller.LIBERO_ACTION_SCALE)
        observations.append(env.observe())
        sim_states.append(env.get_state())
        current = _pose(env)
    return observations, sim_states, actions, float(np.linalg.norm(current - end))


def load_labelled_demos(bank, task: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``{demo_key: (states, stages, dist_to_go)}`` for every demo of a task in a labelled bank."""
    demos = {}
    for key in bank.demo_keys(task):
        stages, remaining = bank.labels(task, key, "stages"), bank.labels(task, key, "dist_to_go")
        if stages is None or remaining is None:
            raise SystemExit(f"{bank.path} has no stage labels for {task}; run resteer.steergen.label_stages first")
        demos[key] = (bank.load(task, key), stages, remaining)
    return demos


def select_target_state(env, target_demos, start_state, start_stage, rng, max_candidates):
    """Same-stage target state minimizing dist_to_go + end-effector distance (stage_matched)."""
    env.reset_to(start_state)
    start_pos = env.eef_pose()[:3, 3]
    feasible = [
        (key, int(t)) for key, (_, stages, _) in target_demos.items() for t in np.flatnonzero(stages == start_stage)
    ]
    if not feasible:
        return None
    candidates = rng.sample(feasible, max_candidates) if len(feasible) > max_candidates else feasible
    best = None
    for key, t in candidates:
        states, _, remaining = target_demos[key]
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()
        cost = remaining[t] + np.linalg.norm(env.eef_pose()[:3, 3] - start_pos)
        if best is None or cost < best[0]:
            best = (float(cost), key, t, states[t])
    return best


def stage_matched_bridge(env, start_state, end_state, raw_demo, start_step, args: Args):
    """Transport-stage bridge (collect_interpolation_data_v3.py, fixed); returns (obs, states, actions, error)."""
    from resteer.steergen import regen_states

    env.reset_to(start_state, reset_controller=True)
    target_start = _pose(env)
    env.reset_to(end_state, reset_controller=True)
    target_end = _pose(env)
    if raw_demo is not None:  # reach the start state by replaying the source demo (keeps the grasp)
        raw_actions, raw_states = raw_demo
        env.reset_to(raw_states[0])
        regen_states.settle(env)
        regen_states.replay(env, raw_actions, max_steps=start_step)
        start = _pose(env)
    else:
        env.reset_to(start_state, reset_controller=True)
        start = target_start
    waypoints = interp.interpolate_poses(start, target_end, args.max_xyz_delta, args.max_rot_delta)[: args.clip_size]
    nq = env.sim.model.nq
    aperture = np.mean(np.abs(start_state[1 : 1 + nq][env.gripper_qpos_indices()]))
    gripper = 1.0 if aperture < args.gripper_close_threshold else -1.0
    observations, sim_states, actions = [env.observe(force_update=True)], [env.get_state()], []
    current = start
    for waypoint in waypoints:
        delta_pos, delta_quat = interp.delta_to(current, waypoint)
        # LIBERO's controller maps ±1 to ±5 cm / ±0.5 rad: scale the raw delta into that space.
        action = env.step_delta(delta_pos * 20.0, interp.scale_rotation(delta_quat, 2.0), gripper)
        actions.append(action)
        observations.append(env.observe())
        sim_states.append(env.get_state())
        current = _pose(env)
    return observations, sim_states, actions, float(np.linalg.norm(current[:3] - target_end[:3]))


def write_bridge(writer, name, target, observations, sim_states, actions, args: Args, meta: dict) -> None:
    from resteer.sim import libero as sim

    builder = episodes.EpisodeBuilder()
    for obs, state, action in zip(observations, sim_states, actions):
        builder.add(
            action=action,
            agentview=obs["agentview_image"],
            wrist=obs["robot0_eye_in_hand_image"],
            state=sim.robot_state(obs),
            sim_state=state,
        )
    prompt = _tasks.format_prompt(target, args.prompt_style)
    segments = [episodes.Segment(0, len(builder), target, prompt)]
    writer.write(name, builder, segments, source="steergen", meta=meta)


def main(args: Args) -> None:
    from resteer.steergen.env import SteerGenEnv

    utils.setup_logging(args.log_level)
    stage_mode = args.mode == "stage_matched"
    steps = args.steps if args.steps is not None else tuple(range(0, 100, 5) if stage_mode else range(50))
    threshold = args.success_threshold if args.success_threshold is not None else (0.1 if stage_mode else 0.05)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    bank = _states.StateBank(args.states)
    sources = [task_name(t) for t in args.source_tasks] or [t.name for t in _tasks.TASKS]
    targets = [task_name(t) for t in args.target_tasks] or [t.name for t in _tasks.TASKS]

    def pick(task: str, forced: int | None) -> str:
        return f"demo_{forced}_states" if forced is not None else rng.choice(bank.demo_keys(task))

    kept = failed = 0
    start_time = time.time()
    with episodes.EpisodeWriter(args.out) as writer:  # appends: a rerun (e.g. of a crashed worker) resumes
        for source in sources:
            for target in targets:
                if source == target or (stage_mode and not transportable(source, target)):
                    continue
                if stage_mode:
                    source_demos, target_demos = load_labelled_demos(bank, source), load_labelled_demos(bank, target)
                else:  # demos are drawn before the env is built, as in the paper-era script
                    choices = [(t, pick(source, args.source_demo), pick(target, args.target_demo)) for t in steps]
                env = SteerGenEnv("replay_delta" if stage_mode else "interpolation_delta", args.camera_size)
                for i, t in enumerate(steps):
                    meta = {"mode": args.mode, "source_task": source, "target_task": target, "step": t,
                            "seed": args.seed, "action_units": "libero", "prompt_style": args.prompt_style,
                            "success_threshold": threshold, "states_file": str(args.states)}  # fmt: skip
                    if stage_mode:
                        source_key = pick(source, args.source_demo)
                        source_states, source_stages, _ = source_demos[source_key]
                        if t >= len(source_states) or source_stages[t] != label_stages.TRANSPORT:
                            continue
                        stage = int(source_stages[t])
                        selected = select_target_state(
                            env, target_demos, source_states[t], stage, rng, args.max_candidates
                        )
                        if selected is None:
                            continue
                        cost, target_key, target_step, end_state = selected
                        raw_demo = None
                        if args.raw_dir is not None:
                            with h5py.File(args.raw_dir / f"{source}_demo.hdf5", "r") as f:
                                data = f[f"data/demo_{source_key.split('_')[1]}"]
                                raw_demo = (data["actions"][()], data["states"][()])
                        start_state = source_states[t]
                        meta.update(stage=stage, target_step=target_step, path_cost=cost, controller="replay_delta")
                    else:
                        _, source_key, target_key = choices[i]
                        source_states, target_states = bank.load(source, source_key), bank.load(target, target_key)
                        if t >= len(source_states) or t >= len(target_states):
                            logger.warning("%s/%s shorter than step %d, skipping", source_key, target_key, t)
                            continue
                        start_state, end_state = source_states[t], target_states[t]
                        meta.update(target_step=t, controller="interpolation_delta",
                                    action_scale=controller.LIBERO_ACTION_SCALE.tolist())  # fmt: skip
                    meta.update(source_demo=int(source_key.split("_")[1]), target_demo=int(target_key.split("_")[1]))
                    name = f"{source}__{target}__t{t:03d}"
                    if name in writer:  # written by an interrupted earlier run
                        kept += 1
                        continue
                    try:
                        if stage_mode:
                            result = stage_matched_bridge(env, start_state, end_state, raw_demo, t, args)
                        else:
                            result = step_matched_bridge(env, start_state, end_state, args)
                    except Exception:
                        logger.exception("Bridge %s -> %s at step %d failed", source, target, t)
                        failed += 1
                        continue
                    observations, sim_states, actions, error = result
                    if error > threshold:
                        logger.info(
                            "%s -> %s t=%d: final error %.4f > %.4f, dropped", source, target, t, error, threshold
                        )
                        failed += 1
                        continue
                    meta["final_error"] = error
                    write_bridge(writer, name, target, observations, sim_states, actions, args, meta)
                    kept += 1
                env.close()
                logger.info("%s -> %s done (%d kept, %d dropped so far)", source, target, kept, failed)
    logger.info("Kept %d bridges, dropped %d, in %.0fs -> %s", kept, failed, time.time() - start_time, args.out)


if __name__ == "__main__":
    main(tyro.cli(Args))
