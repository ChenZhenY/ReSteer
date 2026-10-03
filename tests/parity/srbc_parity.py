"""Exact parity of SRBC collection (``resteer.srbc.collect --paper-compat``) with the paper-era collectors.

Against the deterministic fake policy server (tests/fake_policy_server.py), where actions are a pure function
of the request, this checks:

1. success-rate source: ``filtered_bc_libero_full_rollout.py`` and ``collect --source success_rate`` send the
   identical sequence of policy requests (same switch configs, attempts, observations, prompts, in order).
2. CMI source: the same for ``filtered_bc_libero_full_rollout_mi.py`` and ``collect --source cmi``.
3. recorded frames: one rollout of the old ``run_rollout`` and of the new recorder store identical images
   (after the 224 -> 256 resize the old converter applied), states and actions.

    uv run python tests/parity/srbc_parity.py --old-repo OLD_REPO --old-env WORK/old_env.sh \
        --states WORK/task_names.hdf5 --out /tmp/srbc_parity

OLD_REPO is the research repository (pi0-anytime-steerability @ 2db0966); ``old_env.sh`` comes from
tests/parity/setup_old_env.sh; ``--states`` is any bank with the 10 ``<task>_demo`` groups (the old
collectors only read the task names from it). The old code imports nothing from resteer; this file runs in
both environments, so module-level imports are limited to the standard library and NumPy.
"""

import argparse
import contextlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time
import types

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SOURCE = "put_the_bowl_on_the_plate"
MAX_STEPS = 30
ATTEMPTS = 2
FRAMES_CONFIG = ("Turn On The Stove", 7)  # (new prompt, switch step) of the recorded-frames check


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def write_inputs(out: pathlib.Path, port: int) -> None:
    """Switch configs in the old formats and the new eval-results layout (same success rates)."""
    cells = {  # switch step -> {target: success rate}; alphabetical target order, like the old evaluation
        0: {"open_the_middle_drawer_of_the_cabinet": 0.0, "put_the_bowl_on_the_plate": 0.6, "turn_on_the_stove": 0.5},
        5: {"push_the_plate_to_the_front_of_the_stove": 0.3, "put_the_wine_bottle_on_the_rack": 1.0},
        10: {"turn_on_the_stove": 0.9},
    }  # fmt: skip
    samples = []
    for k, rates in cells.items():
        prompt_results = {
            f"prompt_{i}": {
                "prompt": _title(target),
                "success_rate": rate,
                "experiments": [{"old_prompt": _title(SOURCE), "new_prompt": _title(target), "new_prompt_step": k}],
            }
            for i, (target, rate) in enumerate(rates.items())
        }
        samples.append({"demo_name": "demo_0_states", "step_idx": k, "prompt_results": prompt_results})
        step_file = out / "new_results" / f"task_3_{SOURCE}" / f"step_{k:03d}.json"
        step_file.parent.mkdir(parents=True, exist_ok=True)
        step_file.write_text(json.dumps({"source_task": SOURCE, "switch_step": k, "success_rate": rates}))
    old_args = {"bddl_scene": "libero_goal", "resize_size": 224, "seed": 7, "host": "127.0.0.1", "port": port}
    task = {"task_name": SOURCE, "task_description": _title(SOURCE), "samples": samples}
    (out / "old_results.json").write_text(json.dumps({"args": old_args, "task_results": {"task_3": task}}))

    tuples = [  # paper-era low-MI tuples: raw state-bank indices (two off the replanning grid), a same-task one
        {"new_prompt": "Turn On The Stove", "new_prompt_step": 17},
        {"new_prompt": _title(SOURCE), "new_prompt_step": 10},
        {"new_prompt": "Push The Plate To The Front Of The Stove", "new_prompt_step": 23},
    ]
    for i, t in enumerate(tuples):
        t.update(old_prompt=_title(SOURCE), mutual_info=0.001 * (i + 1), demo_name="demo_0_states")
    (out / "low_mi_tuples.json").write_text(
        json.dumps({"task_name": SOURCE, "task_description": _title(SOURCE), "low_mutual_info_tuples": tuples})
    )


@contextlib.contextmanager
def fake_server(port: int, log: pathlib.Path):
    log.unlink(missing_ok=True)
    cmd = [sys.executable, str(REPO / "tests" / "fake_policy_server.py"), "--port", str(port), "--log", str(log)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        time.sleep(4)
        yield
    finally:
        proc.terminate()
        proc.wait()


def run(cmd: str, log: pathlib.Path, cwd: pathlib.Path, old_env: str | None = None) -> None:
    if old_env:
        cmd = f"source {shlex.quote(old_env)} && {cmd}"
    with open(log, "w") as f:
        result = subprocess.run(["bash", "-c", cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"command failed ({result.returncode}), see {log}:\n{cmd}")


def compare_requests(name: str, old_log: pathlib.Path, new_log: pathlib.Path) -> bool:
    old, new = ([] if not p.exists() else p.read_text().splitlines() for p in (old_log, new_log))
    same = old == new and len(old) > 0
    first = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), None)
    diff = "" if first is None else f" (first difference at request {first})"
    print(f"[{name}] requests: old {len(old)}, new {len(new)}, identical: {same}{diff}")
    return same


def old_frames(old_repo: str, results_json: str, port: int, out: str) -> None:
    """Runs in the old environment: one rollout of the paper-era run_rollout()."""
    sys.path[:0] = [old_repo, os.path.join(old_repo, "examples", "libero_steerability")]
    from examples.libero_steerability import filtered_bc_libero_full_rollout as old

    tester = old.build_env(types.SimpleNamespace(simplified_json=results_json, seed=7, server_port=port))
    prompt, k = FRAMES_CONFIG
    config = {"old_prompt": _title(SOURCE), "new_prompt": prompt, "new_prompt_step": k, "task_name": SOURCE}
    frames, success = old.run_rollout(tester, config, max_steps=MAX_STEPS, bddl_scene="libero_goal")
    np.savez(
        out,
        agentview=np.stack([f.image for f in frames]),
        wrist=np.stack([f.wrist_image for f in frames]),
        state=np.asarray([f.state for f in frames], np.float32),
        actions=np.asarray([f.action for f in frames], np.float32),
        success=success,
    )


def new_frames(port: int, out: pathlib.Path) -> None:
    from resteer import tasks
    from resteer.client import PolicyClient
    from resteer.eval import rollout
    from resteer.sim import libero as sim
    from resteer.srbc import collect

    env = sim.make_env(resolution=collect.PAPER_RESOLUTION, seed=7)
    sim.reset(env, num_warmup_steps=0)
    recorder = collect._Recorder(stale=True)
    client = PolicyClient("127.0.0.1", port, connect_timeout_s=60)
    prompt, k = FRAMES_CONFIG
    result = rollout.run_switch_rollout(
        env, client, tasks.get_task(SOURCE), tasks.get_task(prompt), k, max_steps=MAX_STEPS, on_step=recorder
    )
    client.close()
    env.close()
    ep = recorder.episode
    np.savez(
        out,
        agentview=np.stack(ep.agentview),
        wrist=np.stack(ep.wrist),
        state=np.stack(ep.state),
        actions=np.stack(ep.actions),
        success=result.success,
    )


def compare_frames(old_path: pathlib.Path, new_path: pathlib.Path) -> bool:
    import cv2

    old, new = np.load(old_path), np.load(new_path)
    checks = {
        # The old pipeline stored 224x224 frames and resized them to 256 in the LeRobot converter.
        "agentview": np.stack([cv2.resize(im, (256, 256)) for im in old["agentview"]]),
        "wrist": np.stack([cv2.resize(im, (256, 256)) for im in old["wrist"]]),
        "state": old["state"],
        "actions": old["actions"],
    }
    ok = len(old["actions"]) == len(new["actions"]) > 0
    for key, want in checks.items():
        same = want.shape == new[key].shape and np.array_equal(want, new[key])
        ok &= same
        print(f"[frames] {key}: old {want.shape} new {new[key].shape} identical: {same}")
    print(f"[frames] frames: {len(new['actions'])}, success old/new: {bool(old['success'])}/{bool(new['success'])}")
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-repo", required=True)
    parser.add_argument("--old-env", required=True, help="old_env.sh from tests/parity/setup_old_env.sh")
    parser.add_argument("--states", required=True, help="state bank with the 10 <task>_demo groups")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--role", default="main", choices=["main", "old-frames"], help=argparse.SUPPRESS)
    parser.add_argument("--results-json", help=argparse.SUPPRESS)
    parser.add_argument("--frames-out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.role == "old-frames":
        old_frames(args.old_repo, args.results_json, args.port, args.frames_out)
        return

    old_repo, out = pathlib.Path(args.old_repo).resolve(), pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)  # the old collectors construct a tester that validates this file
    bank = out / "data" / "libero_goal_states.hdf5"
    bank.unlink(missing_ok=True)
    bank.symlink_to(pathlib.Path(args.states).resolve())
    write_inputs(out, args.port)
    old_script = old_repo / "examples" / "libero_steerability"
    common = f"--max_steps {MAX_STEPS} --max_num_rollouts_per_state {ATTEMPTS} --seed 7 --server_port {args.port}"
    new_common = f"--max-steps {MAX_STEPS} --max-attempts-per-config {ATTEMPTS} --paper-compat --port {args.port}"
    new_py = shlex.quote(sys.executable)
    ok = True

    cases = [
        ("success_rate", "filtered_bc_libero_full_rollout.py", out / "old_results.json", out / "new_results"),
        ("cmi", "filtered_bc_libero_full_rollout_mi.py", out / "low_mi_tuples.json", out / "low_mi_tuples.json"),
    ]
    for source, script, old_input, new_input in cases:
        for side in ("old", "new"):
            shutil.rmtree(out / f"{side}_{source}", ignore_errors=True)  # the new collector would resume
            (out / f"{side}_{source}").mkdir()
            with fake_server(args.port, out / f"requests_{side}_{source}.log"):
                if side == "old":
                    cmd = f"python {old_script / script} --simplified_json {old_input} --output_dir {side}_{source} {common}"
                    run(cmd, out / f"{side}_{source}.log", out, old_env=args.old_env)
                else:
                    cmd = f"{new_py} -m resteer.srbc.collect --source {source} --results {new_input} --out {side}_{source} {new_common}"
                    run(cmd, out / f"{side}_{source}.log", out)
        ok &= compare_requests(source, out / f"requests_old_{source}.log", out / f"requests_new_{source}.log")

    with fake_server(args.port, out / "requests_frames.log"):
        this = pathlib.Path(__file__).resolve()
        cmd = (
            f"python {this} --role old-frames --old-repo {old_repo} --old-env '' --states '' --out {out} "
            f"--results-json {out / 'old_results.json'} --port {args.port} --frames-out {out / 'frames_old.npz'}"
        )
        run(cmd, out / "frames_old.log", out, old_env=args.old_env)
        new_frames(args.port, out / "frames_new.npz")
    ok &= compare_frames(out / "frames_old.npz", out / "frames_new.npz")
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
