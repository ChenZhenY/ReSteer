"""LIBERO environment and observation handling shared by every ReSteer pipeline.

The constants and transforms here define what the policy sees; they match the environment
used for the paper's evaluations and must not change silently:

- a single scene, ``libero_goal_steerability.bddl``, rendered at 512x512;
- after ``reset`` the robot settles for 10 steps of the all-zero action;
- policy images are the agent-view and wrist cameras rotated by 180 degrees, then padded and
  resized to 224x224 (PIL bilinear); the proprioceptive state is
  ``[eef_pos (3), eef_axis_angle (3), gripper_qpos (2)]``.
"""

import math
import os
import pathlib
import platform

import numpy as np

# Must run before libero / robosuite are imported.
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.join(os.path.expanduser("~"), ".cache", "resteer", "libero"))
if platform.system() == "Linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
elif platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import libero.libero  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from resteer.client import image_tools  # noqa: E402

LIBERO_PACKAGE_DIR = pathlib.Path(libero.libero.__file__).resolve().parent
STEERABILITY_BDDL = LIBERO_PACKAGE_DIR / "bddl_files" / "libero_steerability" / "libero_goal_steerability.bddl"

RENDER_RESOLUTION = 512
POLICY_RESOLUTION = 224
NUM_WARMUP_STEPS = 10
ACTION_DIM = 7


def make_env(resolution: int = RENDER_RESOLUTION, seed: int | None = None) -> OffScreenRenderEnv:
    """Creates the steerability scene. ``env.seed`` reseeds NumPy's *global* RNG, which drives resets."""
    env = OffScreenRenderEnv(bddl_file_name=str(STEERABILITY_BDDL), camera_heights=resolution, camera_widths=resolution)
    if seed is not None:
        env.seed(seed)
    return env


CAMERA_OBSERVABLES = ("agentview_image", "robot0_eye_in_hand_image")


def set_camera_rendering(env, enabled: bool) -> None:
    """Turns rendering of the two cameras on or off for subsequent steps and observations.

    Rendering has no effect on the simulation (nor on the RNG), so it only needs to be on when
    images are used; ``env.reset()`` turns it back on.
    """
    for name in CAMERA_OBSERVABLES:
        env.env.modify_observable(name, "enabled", enabled)


def reset(env, num_warmup_steps: int = NUM_WARMUP_STEPS, render_cameras: bool = True) -> None:
    """Resets the scene (random object layout) and lets the robot settle with zero actions."""
    env.reset()
    if not render_cameras:
        set_camera_rendering(env, False)
    for _ in range(num_warmup_steps):
        env.step(np.zeros(ACTION_DIM))


def get_observation(env) -> dict:
    return env.env._get_observations(force_update=True)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) to axis-angle. Clips ``quat[3]`` in place, as the paper's code did."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def robot_state(obs: dict) -> np.ndarray:
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))


def camera_images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Agent-view and wrist images rotated by 180 degrees (upright), at render resolution."""
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return agentview, wrist


def policy_input(obs: dict, prompt: str, resolution: int = POLICY_RESOLUTION) -> dict:
    """The observation dict sent to the policy server (openpi LIBERO format)."""
    agentview, wrist = camera_images(obs)
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(agentview, resolution, resolution)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, resolution, resolution)
        ),
        "observation/state": robot_state(obs),
        "prompt": prompt,
    }


def step(env, action) -> tuple[dict, dict]:
    """Steps the env; returns ``(obs, success_dict)`` with one entry per task goal predicate."""
    obs, _, success_dict, _ = env.step(np.asarray(action).tolist(), return_success_dict=True)
    return obs, success_dict
