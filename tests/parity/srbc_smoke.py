"""End-to-end smoke test of SRBC data generation in the real simulator, without a GPU policy.

Runs ``resteer.srbc.collect`` (default mode) for both config sources against the deterministic fake policy
server. The task-success check is replaced by a seeded coin flip, so rollouts end in "successes" of realistic
length. It then mixes the two pools with ``resteer.srbc.build_dataset`` (index / sample / combine). With
``--lerobot-python`` (a Python with openpi's lerobot pins, e.g. third_party/openpi/.venv/bin/python) it also
converts the result to LeRobot and loads it back.

    uv run python tests/parity/srbc_smoke.py --out /tmp/srbc_smoke --lerobot-python PY
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import textwrap

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "parity"))
from srbc_parity import fake_server  # noqa: E402

FORCED_SUCCESS = """
import random, runpy, sys
import resteer.tasks as tasks
rng = random.Random(0)
tasks.is_success = lambda task, success_dict: rng.random() < 0.15
sys.argv = ["collect", *sys.argv[1:]]
runpy.run_module("resteer.srbc.collect", run_name="__main__")
"""

LOAD_BACK = """
import json, sys
import h5py, numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
episode_file, repo_id, root = sys.argv[1:4]
ds = LeRobotDataset(repo_id, root=root)
segments = []
with h5py.File(episode_file) as f:
    for name, g in f["episodes"].items():
        for s in json.loads(g.attrs["segments"]):
            segments.append((name, s))
    assert ds.meta.total_episodes == len(segments), (ds.meta.total_episodes, len(segments))
    assert ds.meta.total_frames == sum(s["end"] - s["start"] for _, s in segments)
    # Frame-exact check of every LeRobot episode against its segment of the episode file.
    for ep, (name, s) in enumerate(segments):
        start = int(ds.episode_data_index["from"][ep])
        for t in (0, s["end"] - s["start"] - 1):
            item = ds[start + t]
            g = f["episodes"][name]
            image = (item["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            assert np.array_equal(image, g["images/agentview"][s["start"] + t]), (ep, t)
            assert np.array_equal(item["state"].numpy(), g["state"][s["start"] + t]), (ep, t)
            assert np.array_equal(item["actions"].numpy(), g["actions"][s["start"] + t]), (ep, t)
            assert item["task"] == s["prompt"], (item["task"], s["prompt"])
print(f"LeRobot dataset OK: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
      f"{len(ds.meta.tasks)} tasks, features {sorted(ds.features)}")
"""


def write_inputs(out: pathlib.Path) -> None:
    results = out / "eval_results"
    cells = {  # (source task id, name) -> {switch step: {target: success rate}}
        (3, "put_the_bowl_on_the_plate"): {0: {"turn_on_the_stove": 0.5}, 20: {"open_the_middle_drawer_of_the_cabinet": 0.3}},
        (9, "turn_on_the_stove"): {10: {"put_the_bowl_on_the_stove": 0.8, "put_the_bowl_on_the_plate": 0.95}},
    }  # fmt: skip
    for (task_id, name), steps in cells.items():
        for k, rates in steps.items():
            path = results / f"task_{task_id}_{name}" / f"step_{k:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"source_task": name, "switch_step": k, "success_rate": rates}))
    tuples = [
        {"source_task": "put_the_bowl_on_the_plate", "target_task": "put_the_wine_bottle_on_the_rack",
         "policy_step": 15, "new_prompt_step": 25, "mutual_info": 0.01, "demo_name": "demo_0_states"},
        {"source_task": "turn_on_the_stove", "target_task": "push_the_plate_to_the_front_of_the_stove",
         "policy_step": 5, "new_prompt_step": 15, "mutual_info": 0.02, "demo_name": "demo_1_states"},
    ]  # fmt: skip
    (out / "cmi").mkdir(parents=True, exist_ok=True)
    (out / "cmi" / "low_cmi_tuples.json").write_text(json.dumps({"low_mutual_info_tuples": tuples}))


def run(cmd: list[str], log: pathlib.Path, env: dict | None = None) -> None:
    with open(log, "w") as f:
        if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode:
            raise SystemExit(f"failed: {' '.join(cmd)} (see {log})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lerobot-python", help="Python with lerobot (openpi's pins) to test the conversion")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_inputs(out)
    py = sys.executable
    collect_args = ["--max-steps", "60", "--successes-per-config", "1", "--max-attempts-per-config", "3"]
    collect_args += ["--port", str(args.port)]
    with fake_server(args.port, out / "requests.log"):
        for source, results in (("success_rate", out / "eval_results"), ("cmi", out / "cmi")):
            cmd = [py, "-c", FORCED_SUCCESS, "--source", source, "--results", str(results)]
            run(cmd + ["--out", str(out / f"srbc_{source}"), *collect_args], out / f"collect_{source}.log")

    from resteer.data import episodes
    from resteer.srbc import build_dataset

    for source in ("success_rate", "cmi"):
        for path in sorted((out / f"srbc_{source}").glob("*.hdf5")):
            for name, arrays, segments, _, meta in episodes.read_episodes(path):
                assert len(arrays["actions"]) == segments[-1].end and segments[0].start == 0
                assert meta["switch_step"] == (segments[1].start if len(segments) == 2 else 0)
                print(f"{source}: {path.name}/{name}: {len(arrays['actions'])} frames, "
                      f"{[(s.start, s.end, s.prompt) for s in segments]}")  # fmt: skip
        build_dataset.index(out / f"srbc_{source}")
    sampled = build_dataset.sample(
        out / "srbc_success_rate" / "database.json", out / "srbc_cmi" / "database.json", out / "mix" / "sampled.json"
    )
    build_dataset.combine(out / "mix" / "sampled.json", out / "mix" / "srbc.hdf5")
    total = sampled["statistics"]["total_demos"]
    assert total == len(list(episodes.read_episodes(out / "mix" / "srbc.hdf5"))) > 0

    if args.lerobot_python:
        env = {**os.environ, "PYTHONPATH": str(REPO / "policy"), "HF_LEROBOT_HOME": str(out / "lerobot")}
        convert = [args.lerobot_python, "-m", "resteer_policy.convert_to_lerobot", str(out / "mix" / "srbc.hdf5")]
        run(convert + ["--repo-id", "resteer/srbc_smoke", "--overwrite"], out / "convert.log", env)
        print(next(line for line in (out / "convert.log").read_text().splitlines() if line.startswith("Wrote")))
        root = out / "lerobot" / "resteer" / "srbc_smoke"
        load = [args.lerobot_python, "-c", textwrap.dedent(LOAD_BACK), str(out / "mix" / "srbc.hdf5")]
        run(load + ["resteer/srbc_smoke", str(root)], out / "load_back.log", env)
        print((out / "load_back.log").read_text().strip().splitlines()[-1])
    print(f"SMOKE OK ({total} combined episodes)")


if __name__ == "__main__":
    main()
