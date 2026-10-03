"""Steerability evaluation on LIBERO-Goal (the protocol of the ReSteer paper).

For each source task, switch step ``k`` and target task, run ``num_repeats`` rollouts that start
from a fresh random scene, follow the source instruction, and switch to the target instruction at
step ``k`` (see :mod:`resteer.eval.rollout`). Results are written as one JSON file per
(source task, switch step); :mod:`resteer.eval.score` aggregates them into the steerability score.

Reproducibility: object layouts come from NumPy's global RNG, seeded once per source task right
after the environment is created, and consumed only by ``env.reset()`` in the fixed loop order
switch step -> target -> repeat. Each step file stores the RNG state at its end, and a switch step in
progress is saved after every rollout (``step_<k>.json.partial``, with the RNG state), so an
interrupted run resumes with exactly the layouts an uninterrupted run would have used and loses at
most one rollout.

Usage (a policy server must be running, see scripts/serve_policy.sh):
    python -m resteer.eval.steerability --out results/pi05_libero --port 8000 --tasks 0 1
"""

import dataclasses
import logging
import pathlib
import sys
import time
from typing import Literal

import imageio
import numpy as np
import tyro

from resteer import tasks as _tasks
from resteer import utils
from resteer.client import PolicyClient
from resteer.eval import rollout as _rollout
from resteer.sim import libero as sim

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    out: pathlib.Path
    """Output directory; one sub-directory per source task."""
    host: str = "127.0.0.1"
    port: int = 8000
    tasks: tuple[int, ...] = tuple(range(10))
    """Source task ids to evaluate (see resteer.tasks)."""
    switch_steps: tuple[int, ...] = tuple(range(0, 100, 5))
    targets: tuple[int, ...] = tuple(range(10))
    """Target task ids. The paper protocol runs all 10; the score excludes target == source by default."""
    skip_same_task: bool = False
    """Do not run target == source rollouts (saves 10%; the include-same-task score is then unavailable)."""
    num_repeats: int = 10
    max_steps: int = 300
    replan_steps: int = 5
    seed: int = 7
    seed_scope: Literal["task", "step"] = "task"
    """"task": seed once per source task and run its switch steps in order (the paper's protocol).
    "step": seed every (task, switch step) independently, so switch steps can be split across workers
    (statistically equivalent, different layouts)."""
    save_video: bool = False
    video_every: int = 2
    """Record every n-th frame when saving videos."""
    log_level: str = "INFO"


def _step_path(task_dir: pathlib.Path, k: int) -> pathlib.Path:
    return task_dir / f"step_{k:03d}.json"


def _partial_path(task_dir: pathlib.Path, k: int) -> pathlib.Path:
    return task_dir / f"step_{k:03d}.json.partial"


# The layout sequence depends on these; resuming with different values would silently diverge.
_RESUME_KEYS = ("targets", "skip_same_task", "num_repeats", "max_steps", "replan_steps", "seed", "seed_scope")


def _check_resumable(saved_config: dict, args: Args, path: pathlib.Path) -> None:
    current = utils.to_jsonable(dataclasses.asdict(args))
    changed = [k for k in _RESUME_KEYS if saved_config.get(k) != current[k]]
    if changed:
        raise RuntimeError(
            f"{path}: cannot resume with different {changed} (saved: {[saved_config.get(k) for k in changed]})"
        )


def step_seed(seed: int, task_id: int, switch_step: int) -> int:
    """Seed of one (source task, switch step) in the "step" seed scope."""
    return int(np.random.SeedSequence([seed, task_id, switch_step]).generate_state(1)[0])


def task_dir_name(task: _tasks.Task) -> str:
    return f"task_{task.id}_{task.name}"


class _VideoRecorder:
    def __init__(self, every: int):
        self.every, self.frames = every, []

    def __call__(self, step, obs, query, action, prompt):
        if step % self.every == 0:
            import cv2

            agentview, _ = sim.camera_images(obs)
            frame = agentview.copy()
            cv2.putText(frame, f"step {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, prompt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            self.frames.append(frame)


def evaluate_task(args: Args, task: _tasks.Task, client: PolicyClient) -> None:
    task_dir = args.out / task_dir_name(task)
    targets = [_tasks.get_task(t) for t in args.targets if not (args.skip_same_task and t == task.id)]
    done = [k for k in args.switch_steps if _step_path(task_dir, k).exists()]
    todo = [k for k in args.switch_steps if k not in done]
    if not todo:
        logger.info("%s: all %d switch steps done", task.name, len(done))
        return
    # Resuming is exact only if completed steps form a prefix of the schedule (task seed scope).
    if args.seed_scope == "task" and done and args.switch_steps[: len(done)] != tuple(done):
        raise RuntimeError(f"{task_dir}: completed steps {done} are not a prefix of {args.switch_steps}")

    last = utils.read_json(_step_path(task_dir, done[-1])) if done else None
    if last is not None:
        _check_resumable(last["config"], args, _step_path(task_dir, done[-1]))

    env = sim.make_env(seed=args.seed)
    if last is not None and args.seed_scope == "task":
        np.random.set_state(utils.decode_rng_state(last["rng_state_end"]))
        logger.info("%s: resuming after switch step %d", task.name, done[-1])
    try:
        for k in todo:
            start = time.time()
            if args.seed_scope == "step":
                np.random.seed(step_seed(args.seed, task.id, k))
            rollouts = []
            partial = _partial_path(task_dir, k)
            if partial.exists():
                saved = utils.read_json(partial)
                _check_resumable(saved["config"], args, partial)
                rollouts = saved["rollouts"]
                np.random.set_state(utils.decode_rng_state(saved["rng_state"]))
                logger.info("%s k=%d: resuming after %d rollouts", task.name, k, len(rollouts))
            index = 0
            for target in targets:
                for rep in range(args.num_repeats):
                    index += 1
                    if index <= len(rollouts):  # done before an interruption (rollouts run in this order)
                        continue
                    sim.reset(env, render_cameras=args.save_video)
                    recorder = _VideoRecorder(args.video_every) if args.save_video else None
                    result = _rollout.run_switch_rollout(
                        env, client, task, target, k,
                        max_steps=args.max_steps, replan_steps=args.replan_steps, on_step=recorder,
                    )  # fmt: skip
                    rollouts.append({"target": target.name, "repeat": rep, **dataclasses.asdict(result)})
                    rng_state = utils.encode_rng_state(np.random.get_state())
                    utils.write_json(
                        partial,
                        {"rollouts": rollouts, "rng_state": rng_state, "config": dataclasses.asdict(args)},
                        indent=None,
                    )
                    if recorder is not None:
                        outcome = "success" if result.success else "failure"
                        video = task_dir / "videos" / f"step_{k:03d}" / f"{target.name}_r{rep}_{outcome}.mp4"
                        video.parent.mkdir(parents=True, exist_ok=True)
                        imageio.mimwrite(video, recorder.frames, fps=30)
            rates = {t.name: float(np.mean([r["success"] for r in rollouts if r["target"] == t.name])) for t in targets}
            utils.write_json(
                _step_path(task_dir, k),
                {
                    "source_task": task.name,
                    "switch_step": k,
                    "success_rate": rates,
                    "rollouts": rollouts,
                    "rng_state_end": utils.encode_rng_state(np.random.get_state()),
                    "config": dataclasses.asdict(args),
                    "server_metadata": client.metadata,
                    "resteer_git": utils.git_revision(),
                },
            )
            partial.unlink(missing_ok=True)
            errors = sum(r["error"] is not None for r in rollouts)
            logger.info(
                "%s k=%d: mean success %.3f over %d rollouts (%d env errors) in %.0fs",
                task.name, k, np.mean([r["success"] for r in rollouts]), len(rollouts), errors, time.time() - start,
            )  # fmt: skip
    finally:
        env.close()


def main(args: Args) -> None:
    utils.setup_logging(args.log_level)
    args.out.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(args.host, args.port)
    logger.info("Connected to policy server %s:%d (metadata %s)", args.host, args.port, client.metadata)
    for task_id in args.tasks:
        evaluate_task(args, _tasks.get_task(task_id), client)
    client.close()


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        sys.exit(130)
