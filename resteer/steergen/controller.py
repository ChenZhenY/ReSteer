"""Operational-space (OSC_POSE) controller configurations used by SteerGen.

Both configs are robosuite's default ``osc_pose`` controller with delta control and a different
output range:

- ``INTERPOLATION_DELTA``: identity scaling, so an action is a raw end-effector displacement in
  metres / radians. Step-matched bridges are executed with it (as in the paper) and rescaled to
  LIBERO's action space when stored.
- ``REPLAY_DELTA``: LIBERO's action space (±1 maps to ±5 cm and ±0.5 rad per step), used to
  replay the original LIBERO-Goal demonstrations and to execute stage-matched bridges.
"""

import json
import os

import numpy as np

INTERPOLATION_DELTA = {
    "input_max": 1,
    "input_min": -1,
    "output_max": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "output_min": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
}
REPLAY_DELTA = {
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
}
CONTROLLERS = {"interpolation_delta": INTERPOLATION_DELTA, "replay_delta": REPLAY_DELTA}

# Converts a raw (metres, radians, gripper) action into LIBERO's normalized action space.
LIBERO_ACTION_SCALE = np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0, 1.0])


def osc_pose_config(name: str) -> dict:
    """robosuite's default OSC_POSE config with the named output range and delta control."""
    import robosuite

    with open(os.path.join(robosuite.__path__[0], "controllers", "config", "osc_pose.json")) as f:
        config = json.load(f)
    config.update(CONTROLLERS[name])
    config["control_delta"] = True
    return config
