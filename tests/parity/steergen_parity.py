"""Old-vs-new parity check for SteerGen (not a pytest test: it runs the simulator for minutes).

Runs the paper-era scripts from mimiclabs-priv (branch libero_steerability) and the resteer CLIs on
the same inputs and seeds, and compares their outputs bit for bit:

1. regen:  playback_libero_collect_sim_state.py  vs  resteer.steergen.regen_states
2. labels: label_libero_goal_states.py           vs  resteer.steergen.label_stages --legacy-dist-to-go
3. v1:     collect_interpolation_data.py         vs  resteer.steergen.generate --mode step_matched

The old code is copied to --work and only its hardcoded /home/droid_robot path prefix is rewritten;
robomimic and torch are replaced by import stubs (they are imported but unused on these paths).
Both sides use the same interpreter; seeds are applied to Python's and NumPy's global RNGs before
the old code runs.

    python tests/parity/steergen_parity.py --old-repo ../pi0-anytime-steerability \
        --mimiclabs ../mimiclabs-priv --raw-dir data/libero_raw/libero_goal --work /tmp/steergen_parity
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import h5py
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OLD_PREFIX = "/home/droid_robot/zhenyang/pi0-anytime-steerability"
TASKS = ("put_the_bowl_on_the_plate", "put_the_bowl_on_the_stove")  # legacy SteerGen ids 4 and 9
SEED = 3


def run(code: str, env: dict, cwd: pathlib.Path) -> None:
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, cwd=cwd, text=True,
                            capture_output=True)  # fmt: skip
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-5000:])
        raise SystemExit(f"command failed in {cwd}")


def truncate_raw(raw_dir: pathlib.Path, out_dir: pathlib.Path, num_demos: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        with h5py.File(raw_dir / f"{task}_demo.hdf5", "r") as src, h5py.File(out_dir / f"{task}_demo.hdf5", "w") as dst:
            data = dst.create_group("data")
            for key, value in src["data"].attrs.items():
                data.attrs[key] = value
            for i in range(num_demos):
                src.copy(src[f"data/demo_{i}"], data, name=f"demo_{i}")


def setup_old(args, work: pathlib.Path) -> dict:
    src = work / "old_src"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(args.mimiclabs / "mimiclabs", src / "mimiclabs", ignore=shutil.ignore_patterns("demos", "*.mp4"))
    libero_root = args.old_repo / "third_party" / "modified_libero"
    for path in (src / "mimiclabs" / "data_collection" / "sim" / "scripts").glob("*.py"):
        path.write_text(path.read_text().replace(OLD_PREFIX, str(args.old_repo)))
    stubs = work / "stubs"
    (stubs / "robomimic" / "envs").mkdir(parents=True, exist_ok=True)
    (stubs / "torch").mkdir(parents=True, exist_ok=True)
    for init in ("robomimic/__init__.py", "robomimic/envs/__init__.py", "torch/__init__.py"):
        (stubs / init).touch()
    (stubs / "robomimic" / "envs" / "env_base.py").write_text("class EnvType:\n    ROBOSUITE_TYPE = 1\n")
    cfg = work / "libero_cfg_old"
    cfg.mkdir(exist_ok=True)
    root = libero_root / "libero" / "libero"
    (cfg / "config.yaml").write_text(
        "".join(
            f"{k}: {root / v}\n"
            for k, v in {
                "benchmark_root": ".",
                "bddl_files": "bddl_files",
                "init_states": "init_files",
                "datasets": "../datasets",
                "assets": "assets",
            }.items()
        )  # fmt: skip
    )
    return dict(os.environ, PYTHONPATH=f"{src}:{libero_root}:{stubs}", LIBERO_CONFIG_PATH=str(cfg),
                PYTHONBREAKPOINT="0")  # fmt: skip


def new_env(work: pathlib.Path) -> dict:
    return dict(os.environ, PYTHONPATH=f"{REPO}:{REPO / 'third_party' / 'libero'}",
                LIBERO_CONFIG_PATH=str(work / "libero_cfg_new"))  # fmt: skip


def compare(name: str, a, b) -> bool:
    ok = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if not ok:
        diff = np.abs(a.astype(float) - b.astype(float)).max() if a.shape == b.shape else "shape"
        print(f"  MISMATCH {name}: {a.shape} {a.dtype} vs {b.shape} {b.dtype}, max diff {diff}")
    return ok


def check_regen(args, work, old_env, raw_small) -> pathlib.Path:
    old_file, new_file = work / "regen_old.hdf5", work / "regen_new.hdf5"
    for f in work.glob("regen_old*.hdf5"):
        f.unlink()
    new_file.unlink(missing_ok=True)
    # The old script visits tasks in LIBERO benchmark order (bowl-on-stove before bowl-on-plate).
    run(f"""
        import argparse, random, numpy as np, runpy
        random.seed({SEED}); np.random.seed({SEED})
        mod = runpy.run_path("{work}/old_src/mimiclabs/data_collection/sim/scripts/playback_libero_collect_sim_state.py")
        mod["main"](argparse.Namespace(libero_raw_data_dir="{raw_small}", libero_target_file="{old_file}", num_videos=0))
    """, old_env, work)  # fmt: skip
    old_file = next(work.glob("regen_old*.hdf5"))  # the old script appends _mujoco<version>
    run(f"""
        from pathlib import Path
        from resteer.steergen import regen_states
        regen_states.main(regen_states.Args(raw_dir=Path("{raw_small}"), out=Path("{new_file}"), seed={SEED},
                          tasks=("put_the_bowl_on_the_stove", "put_the_bowl_on_the_plate")))
    """, new_env(work), work)  # fmt: skip
    ok = True
    with h5py.File(old_file, "r") as old, h5py.File(new_file, "r") as new:
        for task in TASKS:
            for key in old[f"{task}_demo"]:
                ok &= compare(f"{task}/{key}", old[f"{task}_demo/{key}"][:], new[f"{task}_demo/{key}"][:])
    print(f"[regen]  {'IDENTICAL' if ok else 'DIFFERENT'}: {old_file.name} vs {new_file.name}")
    return old_file


def check_labels(work, old_env, bank: pathlib.Path) -> None:
    old_copy, new_copy = work / "labels_old.hdf5", work / "labels_new.hdf5"
    shutil.copy(bank, old_copy)
    shutil.copy(bank, new_copy)
    run(f"""
        import runpy
        mod = runpy.run_path("{work}/old_src/mimiclabs/data_collection/sim/scripts/label_libero_goal_states.py")
        mod["process_hdf5_file"]("{old_copy}", "libero_goal_steerability", gripper_open_threshold=0.03)
    """, old_env, work)  # fmt: skip
    run(f"""
        from resteer.steergen import label_stages
        label_stages.label_bank(__import__("pathlib").Path("{new_copy}"), 0.03, legacy_dist_to_go=True)
    """, new_env(work), work)  # fmt: skip
    ok = True
    with h5py.File(old_copy, "r") as old, h5py.File(new_copy, "r") as new:
        for task in TASKS:
            for kind in ("stages", "dist_to_go"):
                for key in old[f"{task}_{kind}"]:
                    ok &= compare(
                        f"{task}_{kind}/{key}", old[f"{task}_{kind}/{key}"][:], new[f"{task}_{kind}/{key}"][:]
                    )
        stages = [new[f"{t}_stages/{k}"][:] for t in TASKS for k in new[f"{t}_stages"]]
    print(f"[labels] {'IDENTICAL' if ok else 'DIFFERENT'} (stage counts {np.bincount(np.concatenate(stages))})")


def check_v1(work, old_env, raw_small, steps) -> None:
    bank = work / "demo_states.hdf5"
    run(f"""
        from resteer import states
        states.build_from_raw("{raw_small}", "{bank}")
    """, new_env(work), work)  # fmt: skip
    demos = work / "old_src/mimiclabs/data_collection/sim/demos/parity_v1"
    shutil.rmtree(demos, ignore_errors=True)
    run(f"""
        import random, numpy as np, importlib.util, sys
        random.seed({SEED}); np.random.seed({SEED})
        path = "{work}/old_src/mimiclabs/data_collection/sim/scripts/collect_interpolation_data.py"
        spec = importlib.util.spec_from_file_location("old_v1", path)
        mod = importlib.util.module_from_spec(spec); sys.modules["old_v1"] = mod; spec.loader.exec_module(mod)
        mod.main(mod.InterpolationDataConfig(start_state={list(steps)}, start_task_id=[4, 9], end_task_id=[4, 9],
                                             state_file="{bank}", save_name="parity_v1"))
    """, old_env, work)  # fmt: skip
    new_file = work / "bridges_new.hdf5"
    new_file.unlink(missing_ok=True)
    run(f"""
        from pathlib import Path
        from resteer.steergen import generate
        generate.main(generate.Args(states=Path("{bank}"), out=Path("{new_file}"), seed={SEED}, steps={tuple(steps)},
                      source_tasks={TASKS}, target_tasks={TASKS}, prompt_style="title"))
    """, new_env(work), work)  # fmt: skip
    sys.path[:0] = [str(REPO), str(REPO / "third_party" / "libero")]
    from resteer.sim.libero import robot_state
    from resteer.steergen.controller import LIBERO_ACTION_SCALE

    ok, count = True, 0
    with h5py.File(new_file, "r") as new:
        for source, target in ((TASKS[0], TASKS[1]), (TASKS[1], TASKS[0])):
            names = sorted(n for n in new["episodes"] if n.startswith(f"{source}__{target}__"))
            old_files = sorted((demos / "libero_goal_steerability" / source / target).glob("demo_*.hdf5"),
                               key=lambda p: int(p.stem.split("_")[1]))  # fmt: skip
            ok &= len(names) == len(old_files)
            print(f"  {source} -> {target}: {len(old_files)} old / {len(names)} new bridges kept")
            for name, old_path in zip(names, old_files):
                with h5py.File(old_path, "r") as f:
                    demo = f["data/demo_0"]
                    obs = {k: demo[f"obs/{k}"][:] for k in demo["obs"]}
                    ep = new[f"episodes/{name}"]
                    ok &= compare(
                        f"{name}/actions",
                        ep["actions"][:],
                        (demo["actions"][:] * LIBERO_ACTION_SCALE).astype(np.float32),
                    )
                    ok &= compare(f"{name}/agentview", ep["images/agentview"][:], obs["agentview_image"])
                    ok &= compare(f"{name}/wrist", ep["images/wrist"][:], obs["robot0_eye_in_hand_image"])
                    ok &= compare(f"{name}/sim_states", ep["sim_states"][:], demo["states"][:])
                    state = np.stack(
                        [robot_state({k: v[t].copy() for k, v in obs.items()}) for t in range(len(demo["actions"]))]
                    )
                    ok &= compare(f"{name}/state", ep["state"][:], state.astype(np.float32))
                    count += 1
    print(f"[v1]     {'IDENTICAL' if ok else 'DIFFERENT'} ({count} bridges compared)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-repo", type=pathlib.Path, required=True)
    parser.add_argument("--mimiclabs", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--work", type=pathlib.Path, required=True)
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--only", choices=["regen", "labels", "v1"], nargs="*")
    args = parser.parse_args()
    args.old_repo, args.mimiclabs = args.old_repo.resolve(), args.mimiclabs.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    raw_small = work / "raw_small"
    truncate_raw(args.raw_dir, raw_small, args.num_demos)
    old_env = setup_old(args, work)
    only = set(args.only or ["regen", "labels", "v1"])
    bank = None
    if "regen" in only or "labels" in only:
        bank = check_regen(args, work, old_env, raw_small)
    if "labels" in only:
        check_labels(work, old_env, bank)
    if "v1" in only:
        check_v1(work, old_env, raw_small, args.steps)


if __name__ == "__main__":
    main()
