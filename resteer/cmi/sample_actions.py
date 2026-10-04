"""Samples policy actions at bank states and estimates the entropies behind CMI.

For every sampled state ``s`` of a source task and every instruction ``l`` (all 10 tasks), draw
``num_samples`` action chunks from the policy and reduce each to its end-point displacement
(sum of the chunk's delta xyz). Then

    H(A|s,l) = kde_entropy(end points for l)          (32 samples)
    H(A|s)   = kde_entropy(end points for all l)      (10 x 32 samples)

and :mod:`resteer.cmi.compute_cmi` turns them into ``I(A;L|s) = H(A|s) - H(A|s,l)``.

States: for each policy step k in ``range(0, num_steps, stride)`` one bank state at index
``offset + k`` (``offset`` = the bank's warm-up length, 10 for policy-rollout banks) from a random
rollout (Python ``random`` seeded per task). Before querying, the scene is reset, set to the
state, and stepped once with the zero action; resets also re-sample fixture placements, so this
sequence is kept exactly as in the paper for reproducibility.

One JSON per state is written to ``<out>/task_<id>_<name>/state_<index>.json``; runs resume.

    python -m resteer.cmi.sample_actions --states data/states/pi05_libero.hdf5 --out results/cmi/pi05_libero
"""

import dataclasses
import logging
import pathlib
import random

import numpy as np
import tyro

from resteer import states as _states
from resteer import tasks as _tasks
from resteer import utils
from resteer.client import PolicyClient
from resteer.cmi import kde
from resteer.sim import libero as sim

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    states: pathlib.Path
    """State bank (see resteer.cmi.collect_states)."""
    out: pathlib.Path
    host: str = "127.0.0.1"
    port: int = 8000
    tasks: tuple[int, ...] = tuple(range(10))
    """Source tasks (bank groups) to sample states from."""
    targets: tuple[int, ...] = tuple(range(10))
    """Instructions to query at every state."""
    num_steps: int = 100
    stride: int = 5
    offset: int | None = None
    """Bank index of policy step 0 (default: the bank's num_warmup_states)."""
    num_samples: int = 32
    batch_size: int = 8
    """Samples per request when the server supports batched inference."""
    prompt_style: _tasks.PromptStyle = "lower"
    """Instruction format sent to the policy. "title" reproduces the paper's CMI runs (Title Case)."""
    seed: int = 7
    save_actions: bool = False
    """Also store the sampled end points (npz per state)."""
    log_level: str = "INFO"


def set_scene_state(env, state: np.ndarray) -> None:
    env.reset()
    env.set_state(state)
    env.sim.forward()
    env.step(np.zeros(sim.ACTION_DIM))


def endpoints(action_chunks: np.ndarray) -> np.ndarray:
    """[N, horizon, 7] delta actions -> [N, 3] summed xyz displacement (float32)."""
    return np.asarray(action_chunks, dtype=np.float32).sum(axis=1)[:, :3]


def sample_task(args: Args, task: _tasks.Task, bank: _states.StateBank, client: PolicyClient) -> None:
    task_dir = args.out / f"task_{task.id}_{task.name}"
    targets = [_tasks.get_task(t) for t in args.targets]
    random.seed(args.seed)
    np.random.seed(args.seed)
    offset = bank.num_warmup_states if args.offset is None else args.offset
    samples = bank.sample_uniform(task.name, args.num_steps, args.stride, offset, rng=random)
    if not samples:
        raise SystemExit(
            f"{task.name}: no rollout in {args.states} is long enough to sample. States are taken at bank indices "
            f"{offset}, {offset + args.stride}, ...; an index is used only from rollouts with more than index + "
            f"{offset} states (the paper's conservative check), so the first needs more than {2 * offset}. "
            "Collect longer rollouts."
        )
    paths = [task_dir / f"state_{index:03d}.json" for _, index, _ in samples]
    done = [p.exists() for p in paths]
    if all(done):
        logger.info("%s: all %d states done", task.name, len(samples))
        return
    if any(done) and done != sorted(done, reverse=True):
        raise RuntimeError(f"{task_dir}: completed states are not a prefix of the schedule")

    env = sim.make_env(seed=args.seed)
    if any(done):
        last = utils.read_json(paths[sum(done) - 1])
        saved, current = last.get("config", {}), utils.to_jsonable(dataclasses.asdict(args))
        changed = [k for k in ("targets", "num_steps", "stride", "offset", "seed") if saved.get(k) != current[k]]
        if changed:
            raise RuntimeError(f"{task_dir}: cannot resume with different {changed}")
        np.random.set_state(utils.decode_rng_state(last["rng_state_end"]))
    try:
        for (demo_key, index, state), path in zip(samples, paths, strict=True):
            if path.exists():
                continue
            set_scene_state(env, state)  # paper: one extra reset per state before the prompt loop
            points, entropy = {}, {}
            for target in targets:
                set_scene_state(env, state)
                query = sim.policy_input(sim.get_observation(env), _tasks.format_prompt(target, args.prompt_style))
                points[target.name] = endpoints(client.sample_actions(query, args.num_samples, args.batch_size))
                entropy[target.name] = kde.kde_entropy(points[target.name])
            state_entropy = kde.kde_entropy(np.concatenate(list(points.values())))
            utils.write_json(
                path,
                {
                    "source_task": task.name,
                    "demo": demo_key,
                    "state_index": index,
                    "policy_step": index - offset,
                    "state_entropy": state_entropy,
                    "prompt_entropy": entropy,
                    "prompt_style": args.prompt_style,
                    "num_samples": args.num_samples,
                    "config": dataclasses.asdict(args),
                    "rng_state_end": utils.encode_rng_state(np.random.get_state()),
                    "server_metadata": client.metadata,
                    "resteer_git": utils.git_revision(),
                },
            )
            if args.save_actions:
                actions_dir = task_dir / "actions"
                actions_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(actions_dir / f"state_{index:03d}.npz", **points)
            logger.info("%s state %d (step %d): H(A|s)=%.3f", task.name, index, index - offset, state_entropy)
    finally:
        env.close()


def main(args: Args) -> None:
    utils.setup_logging(args.log_level)
    bank = _states.StateBank(args.states)
    client = PolicyClient(args.host, args.port)
    if not client.supports_batch:
        logger.info("Server has no batched inference; drawing %d sequential samples per query", args.num_samples)
    for task_id in args.tasks:
        sample_task(args, _tasks.get_task(task_id), bank, client)
    client.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
