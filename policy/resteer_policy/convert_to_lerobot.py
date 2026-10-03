"""Converts ReSteer episode files (SteerGen bridges, SRBC rollouts) into a LeRobot dataset for openpi.

Runs in the policy/ environment so that the lerobot / datasets versions are the ones openpi trains
with (lerobot @ 0cf8648). It reads the layout written by ``resteer/data/episodes.py`` (version 1)
directly with h5py, so the two environments stay independent; keep both files in sync.

Every segment of an episode becomes one LeRobot episode whose task string is the segment prompt:
a switching rollout yields "source instruction" frames [0, k) and "target instruction" frames [k, T).
Features follow openpi's LIBERO format (image / wrist_image 256x256x3, state (8,), actions (7,), 10 fps).

    scripts/policy.sh convert_to_lerobot data/srbc/srbc.hdf5 --repo-id resteer/libero_goal_srbc

The dataset is written to ``$HF_LEROBOT_HOME/<repo-id>`` (or ``--root``).
"""

import dataclasses
import json
import pathlib
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

FORMAT = "resteer-episodes"
SUPPORTED_VERSIONS = (1,)
IMAGE_SIZE = 256
FEATURES = {
    "image": {"dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


@dataclasses.dataclass
class Args:
    inputs: tyro.conf.Positional[list[pathlib.Path]]
    """Episode files, or directories searched recursively for *.hdf5."""
    repo_id: str
    root: pathlib.Path | None = None
    """Output directory; default $HF_LEROBOT_HOME/<repo_id>."""
    overwrite: bool = False
    drop_zero_switch: bool = False
    """Paper-era behavior: skip SRBC episodes whose prompt switched at step 0."""
    max_episodes: int | None = None
    """Stop after this many LeRobot episodes (for testing)."""


def _resize(images: np.ndarray) -> np.ndarray:
    if images.shape[1:3] == (IMAGE_SIZE, IMAGE_SIZE):
        return images
    return np.stack([cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE)) for im in images])


def episode_files(inputs: list[pathlib.Path]) -> list[pathlib.Path]:
    files = []
    for path in inputs:
        files.extend(sorted(path.rglob("*.hdf5")) if path.is_dir() else [path])
    return files


def iter_segments(path: pathlib.Path, drop_zero_switch: bool = False):
    """Yields ``(episode_name, segment, frames)``; loads one episode at a time."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            print(f"skip {path}: not a {FORMAT} file")
            return
        if int(f.attrs["version"]) not in SUPPORTED_VERSIONS:
            raise ValueError(f"{path}: unsupported episode format version {f.attrs['version']}")
        for name, group in f["episodes"].items():
            meta = json.loads(group.attrs["meta"])
            if drop_zero_switch and group.attrs["source"] == "srbc" and meta.get("switch_step") == 0:
                continue
            frames = {
                "image": _resize(group["images/agentview"][:]),
                "wrist_image": _resize(group["images/wrist"][:]),
                "state": group["state"][:].astype(np.float32),
                "actions": group["actions"][:].astype(np.float32),
            }
            for segment in json.loads(group.attrs["segments"]):
                if segment["end"] > segment["start"]:
                    yield name, segment, {k: v[segment["start"] : segment["end"]] for k, v in frames.items()}


def main(args: Args) -> None:
    root = args.root or HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="panda",
        fps=10,
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    num_episodes, num_frames, tasks = 0, 0, {}
    for path in episode_files(args.inputs):
        for _, segment, frames in iter_segments(path, args.drop_zero_switch):
            for t in range(len(frames["actions"])):
                dataset.add_frame({**{k: v[t] for k, v in frames.items()}, "task": segment["prompt"]})
            dataset.save_episode()
            num_episodes += 1
            num_frames += len(frames["actions"])
            tasks[segment["prompt"]] = tasks.get(segment["prompt"], 0) + 1
            if args.max_episodes is not None and num_episodes >= args.max_episodes:
                break
        if args.max_episodes is not None and num_episodes >= args.max_episodes:
            break
    dataset.stop_image_writer()
    print(f"Wrote {num_episodes} episodes / {num_frames} frames to {root}")
    for task, count in sorted(tasks.items()):
        print(f"  {count:5d}  {task}")


if __name__ == "__main__":
    main(tyro.cli(Args))
