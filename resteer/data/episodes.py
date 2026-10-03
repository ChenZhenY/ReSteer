"""Episode files: the hand-off format between data generation (SteerGen, SRBC) and training.

One HDF5 file holds any number of episodes::

    attrs: format="resteer-episodes", version=1
    episodes/<name>/
        actions        [T, 7]       f32  action executed after observing frame t (LIBERO delta EEF + gripper)
        images/agentview [T, H, W, 3] u8 upright (180°-rotated) agent-view camera
        images/wrist   [T, H, W, 3] u8   upright wrist camera
        state          [T, 8]       f32  eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2)
        sim_states     [T, D]       f64  optional flattened MuJoCo states
        attrs:
          segments  JSON [{"start", "end", "task", "prompt"}]: frames [start, end) were produced under `prompt`
          source    "steergen" | "srbc"
          meta      JSON provenance (source/target task, switch step, success rate or CMI, seed, ...)

``policy/resteer_policy/convert_to_lerobot.py`` turns every segment into one LeRobot episode whose
task string is the segment prompt. It reads this layout directly (no import of this package), so
changes here must be mirrored there; bump ``VERSION`` when the layout changes.
"""

import dataclasses
import json
import pathlib

import cv2
import h5py
import numpy as np

FORMAT = "resteer-episodes"
VERSION = 1
IMAGE_SIZE = 256  # resolution stored for training (LeRobot LIBERO datasets use 256x256)


@dataclasses.dataclass
class Segment:
    start: int
    end: int
    task: str  # task name, see resteer.tasks
    prompt: str  # exact instruction string used as the training label


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Bilinear resize to the stored training resolution (the paper's converter used cv2 defaults)."""
    if image.shape[0] == size and image.shape[1] == size:
        return image
    return cv2.resize(image, (size, size))


class EpisodeBuilder:
    """Accumulates the frames of one episode."""

    def __init__(self):
        self.actions, self.agentview, self.wrist, self.state, self.sim_states = [], [], [], [], []

    def add(self, *, action, agentview, wrist, state, sim_state=None) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.agentview.append(resize_image(np.asarray(agentview, dtype=np.uint8)))
        self.wrist.append(resize_image(np.asarray(wrist, dtype=np.uint8)))
        self.state.append(np.asarray(state, dtype=np.float32))
        if sim_state is not None:
            self.sim_states.append(np.asarray(sim_state, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.actions)


class EpisodeWriter:
    """Appends episodes to one file (``with EpisodeWriter(path) as w: w.write(...)``)."""

    def __init__(self, path: "str | pathlib.Path", mode: str = "a"):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, mode)
        self._file.attrs["format"] = FORMAT
        self._file.attrs["version"] = VERSION
        self._root = self._file.require_group("episodes")
        # "meta" is written last: an episode without it was cut short by a crash of a previous run.
        for name in [name for name, group in self._root.items() if "meta" not in group.attrs]:
            del self._root[name]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self._root)

    def __contains__(self, name: str) -> bool:
        return name in self._root

    def write(self, name: str, episode: EpisodeBuilder, segments: list[Segment], source: str, meta: dict) -> None:
        if name in self._root:
            raise ValueError(f"Episode {name} already exists in {self.path}")
        if not segments or segments[0].start != 0 or segments[-1].end != len(episode):
            raise ValueError(f"Segments {segments} do not cover the {len(episode)} frames of {name}")
        group = self._root.create_group(name)
        group.create_dataset("actions", data=np.stack(episode.actions))
        group.create_dataset("images/agentview", data=np.stack(episode.agentview), compression="gzip")
        group.create_dataset("images/wrist", data=np.stack(episode.wrist), compression="gzip")
        group.create_dataset("state", data=np.stack(episode.state))
        if episode.sim_states:
            group.create_dataset("sim_states", data=np.stack(episode.sim_states), compression="gzip")
        group.attrs["segments"] = json.dumps([dataclasses.asdict(s) for s in segments])
        group.attrs["source"] = source
        group.attrs["meta"] = json.dumps(meta)
        self._file.flush()


def read_episodes(path: "str | pathlib.Path"):
    """Yields ``(name, arrays, segments, source, meta)`` for every episode in a file."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        for name, group in f["episodes"].items():
            arrays = {
                "actions": group["actions"][:],
                "agentview": group["images/agentview"][:],
                "wrist": group["images/wrist"][:],
                "state": group["state"][:],
            }
            segments = [Segment(**s) for s in json.loads(group.attrs["segments"])]
            yield name, arrays, segments, group.attrs["source"], json.loads(group.attrs["meta"])
