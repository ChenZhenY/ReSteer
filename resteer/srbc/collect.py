"""SRBC data collection: successful prompt-switching rollouts of the current policy.

For every switch config (see :mod:`resteer.srbc.select`), the policy is rolled out with the
source instruction and switched to the target instruction at the config's step. Each rollout that
achieves the target goal after the switch is saved, until ``successes_per_config`` successes or
``max_attempts_per_config`` attempts.

Episodes go to ``<out>/<source_task>.hdf5`` (format: :mod:`resteer.data.episodes`) as two segments:
frames before the switch, labelled with the source instruction, and frames from the switch on,
labelled with the target instruction. A switch at step 0 gives a single segment. The policy is
always queried with lower-case instructions; ``--prompt-style`` only sets the stored labels.

Each source task uses one environment, seeded once, with the evaluation's rendering and warm-up.
Rerunning the same command resumes an interrupted collection: successes are counted from the
episode file, and attempts and the object-layout RNG state from ``<source_task>.progress.json``.

``--paper-compat`` reproduces the paper-era collector exactly:
- a new environment seeded with 7 for every rollout, so every attempt starts from the same object layout;
- 224x224 rendering and no warm-up steps;
- 250 steps;
- frames carry the observation of the most recent policy query, not the current one;
- CMI tuples switch at the raw state-bank index;
- Title Case labels.

The paper's launch scripts additionally passed ``--max-steps 400 --max-attempts-per-config 25`` and, for the
success-rate source, ``--max-success-rate 0.7``.

    python -m resteer.srbc.collect --source success_rate --results results/steergen_policy \
        --out data/srbc/success_rate --tasks 0 --port 8000
"""

import dataclasses
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import tyro

from resteer import tasks as _tasks
from resteer import utils
from resteer.client import PolicyClient
from resteer.data import episodes as _episodes
from resteer.eval import rollout as _rollout
from resteer.sim import libero as sim
from resteer.srbc import select as _select

logger = logging.getLogger(__name__)

PAPER_RESOLUTION = 224
PAPER_MAX_STEPS = 250


@dataclasses.dataclass
class Args:
    source: _select.Source
    results: pathlib.Path
    """success_rate: a steerability results directory. cmi: a compute_cmi output directory or tuples JSON."""
    out: pathlib.Path
    """Output directory; one episode file per source task."""
    host: str = "127.0.0.1"
    port: int = 8000
    tasks: tuple[int, ...] | None = None
    """Only these source task ids (e.g. one per worker)."""
    min_success_rate: float = 0.2
    max_success_rate: float = 0.9
    include_same_task: bool = True
    """cmi source: keep tuples whose target is the source task (paper-era tuple files contain them;
    compute_cmi excludes them unless run with --include-same-task or --paper-compat)."""
    max_attempts_per_config: int = 50
    successes_per_config: int = 5
    max_total_rollouts: int = 10000
    """Per source task."""
    max_steps: int | None = None
    """Default 300, or 250 with --paper-compat."""
    replan_steps: int = 5
    seed: int = 7
    prompt_style: _tasks.PromptStyle | None = None
    """Stored training labels; default lower, or title with --paper-compat."""
    paper_compat: bool = False
    limit: int | None = None
    """Only the first N switch configs (for testing)."""
    num_shards: int = 1
    shard: int = 0
    """Split each task's configs over workers: this worker takes configs[shard::num_shards], seeds its
    environment with seed + shard and writes <task>.shard<I>of<N>.hdf5 (build_dataset reads them all)."""
    log_level: str = "INFO"


class _Recorder:
    """Collects frames during a rollout (``on_step`` callback of run_switch_rollout)."""

    def __init__(self, stale: bool):
        self.stale = stale
        self.episode = _episodes.EpisodeBuilder()

    def __call__(self, step, obs, query, action, prompt):
        if self.stale:  # paper-era: the frame of the most recent policy query
            agentview, wrist = query["observation/image"], query["observation/wrist_image"]
            state = query["observation/state"]
        else:
            agentview, wrist = sim.camera_images(obs)
            state = sim.robot_state(obs)
        self.episode.add(action=action, agentview=agentview, wrist=wrist, state=state)


def _segments(config: _select.SwitchConfig, num_frames: int, style: _tasks.PromptStyle) -> list[_episodes.Segment]:
    def segment(start, end, task_name):
        return _episodes.Segment(start, end, task_name, _tasks.format_prompt(task_name, style))

    k = config.switch_step
    if k == 0:
        return [segment(0, num_frames, config.target)]
    return [segment(0, k, config.source), segment(k, num_frames, config.target)]


def _saved_successes(path: pathlib.Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if path.exists():
        with h5py.File(path, "r") as f:
            for group in f["episodes"].values() if "episodes" in f else []:
                if "meta" not in group.attrs:  # cut short by a crash; EpisodeWriter removes it
                    continue
                meta = json.loads(group.attrs["meta"])
                key = f"{meta['source_task']}->{meta['target_task']}@{meta['switch_step']}"
                counts[key] = counts.get(key, 0) + 1
    return counts


def collect_task(args: Args, source_task: str, configs: list[_select.SwitchConfig], client: PolicyClient) -> None:
    style = args.prompt_style or ("title" if args.paper_compat else "lower")
    max_steps = args.max_steps or (PAPER_MAX_STEPS if args.paper_compat else 300)
    stem = source_task if args.num_shards == 1 else f"{source_task}.shard{args.shard}of{args.num_shards}"
    out_path = args.out / f"{stem}.hdf5"
    progress_path = args.out / f"{stem}.progress.json"
    seed = args.seed + args.shard
    progress = utils.read_json(progress_path) if progress_path.exists() else {"attempts": {}}
    successes = _saved_successes(out_path)
    total = sum(progress["attempts"].values())

    env = None
    if not args.paper_compat:
        env = sim.make_env(seed=seed)
        if "rng_state" in progress:
            np.random.set_state(utils.decode_rng_state(progress["rng_state"]))
    try:
        with _episodes.EpisodeWriter(out_path) as writer:
            for config in configs:
                attempts = progress["attempts"].get(config.key, 0)
                found = successes.get(config.key, 0)
                start = time.time()
                while (
                    found < args.successes_per_config
                    and attempts < args.max_attempts_per_config
                    and total < args.max_total_rollouts
                ):
                    if args.paper_compat:
                        env = sim.make_env(resolution=PAPER_RESOLUTION, seed=seed)
                        sim.reset(env, num_warmup_steps=0)
                    else:
                        sim.reset(env, render_cameras=False)
                    recorder = _Recorder(stale=args.paper_compat)
                    result = _rollout.run_switch_rollout(
                        env,
                        client,
                        _tasks.get_task(config.source),
                        _tasks.get_task(config.target),
                        config.switch_step,
                        max_steps=max_steps,
                        replan_steps=args.replan_steps,
                        on_step=recorder,
                    )
                    if args.paper_compat:
                        env.close()
                        env = None
                    if result.error:
                        logger.warning("%s attempt %d: env error %s", config.key, attempts, result.error)
                    if result.success:
                        meta = {
                            "config_source": args.source,
                            "source_task": config.source,
                            "target_task": config.target,
                            "switch_step": config.switch_step,
                            "success_rate": config.success_rate,
                            "mutual_info": config.mutual_info,
                            "state_index": config.state_index,
                            "demo_name": config.demo_name,
                            "attempt": attempts,
                            "first_success_step": result.first_success_step,
                            "max_steps": max_steps,
                            "seed": seed,
                            "paper_compat": args.paper_compat,
                        }
                        segments = _segments(config, len(recorder.episode), style)
                        writer.write(f"episode_{len(writer):05d}", recorder.episode, segments, "srbc", meta)
                        found += 1
                    attempts += 1
                    total += 1
                    progress["attempts"][config.key] = attempts
                    if env is not None:
                        progress["rng_state"] = utils.encode_rng_state(np.random.get_state())
                    utils.write_json(progress_path, progress)
                successes[config.key] = found
                if attempts:
                    logger.info(
                        "%s: %d/%d successful (%.0fs)", config.key, found, attempts, time.time() - start
                    )  # fmt: skip
                if total >= args.max_total_rollouts:
                    logger.info("Reached max_total_rollouts=%d", args.max_total_rollouts)
                    break
            logger.info("%s: %d episodes in %s", source_task, len(writer), out_path)
    finally:
        if env is not None:
            env.close()


def main(args: Args) -> None:
    utils.setup_logging(args.log_level)
    configs = _select.select(
        args.source,
        args.results,
        min_success_rate=args.min_success_rate,
        max_success_rate=args.max_success_rate,
        include_same_task=args.include_same_task,
        source_tasks=args.tasks,
        paper_compat=args.paper_compat,
    )
    if args.limit is not None:
        configs = configs[: args.limit]
    logger.info("%d switch configs from %s", len(configs), args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(args.host, args.port)
    by_source: dict[str, list[_select.SwitchConfig]] = {}
    for config in configs:
        by_source.setdefault(config.source, []).append(config)
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"--shard must be in [0, {args.num_shards})")
    for source_task, task_configs in by_source.items():
        collect_task(args, source_task, task_configs[args.shard :: args.num_shards], client)
    client.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
