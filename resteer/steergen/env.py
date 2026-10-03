"""Pose-controlled steerability scene for SteerGen.

A thin layer over LIBERO's ``CustomControlEnv`` (a LIBERO env with an explicit controller config)
on the ``libero_goal_steerability`` scene. SteerGen drives the robot with end-effector pose
deltas and restores arbitrary simulator states:

- ``reset_to(state)`` reloads the model XML captured at construction and sets a flattened MuJoCo
  state ``[time, qpos, qvel]``. It draws no random numbers, but the fixture placement baked into
  the XML comes from NumPy's global RNG at construction time.
- ``observe()`` returns robosuite observations with images rotated upright (180 degrees).
"""

import copy

import numpy as np
import transforms3d as t3d

from resteer.sim import libero as sim  # noqa: F401  (sets up LIBERO paths/rendering before import)
from resteer.steergen import controller

from libero.libero.envs.env_wrapper import CustomControlEnv  # noqa: E402  isort: skip

CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]


class SteerGenEnv(CustomControlEnv):
    def __init__(self, controller_name: str, camera_size: int = 256):
        super().__init__(
            bddl_file_name=str(sim.STEERABILITY_BDDL),
            robots=["Panda"],
            controller_configs=[controller.osc_pose_config(controller_name)],
            camera_names=CAMERA_NAMES,
            camera_heights=camera_size,
            camera_widths=camera_size,
        )
        self.xml = self.env.sim.model.get_xml()

    @property
    def robot(self):
        return self.env.robots[0]

    def get_state(self) -> np.ndarray:
        return np.array(self.env.sim.get_state().flatten())

    def reset_to(self, state: np.ndarray, reset_controller: bool = False) -> dict:
        """Reloads the construction-time XML and sets a flattened simulator state."""
        self.reset_from_xml_string(self.xml)
        self.sim.reset()
        self.sim.set_state_from_flattened(state)
        self.sim.forward()
        obs = self.observe()
        if reset_controller:
            # Re-sync the OSC controller with the new state (goal, nullspace posture, gripper command).
            robot = self.robot
            robot.controller.update(force=True)
            robot.controller.reset_goal()
            robot.controller.update_initial_joints(robot.controller.joint_pos)
            if hasattr(robot, "gripper"):
                robot.gripper.current_action = np.zeros(robot.gripper.dof)
        return obs

    def observe(self, force_update: bool = False) -> dict:
        obs = self.env._get_observations(force_update=force_update)
        for key, value in obs.items():
            if "image" in key:
                obs[key] = np.rot90(value, k=2)
        return copy.deepcopy(obs)

    def eef_pose(self) -> np.ndarray:
        """4x4 pose of the gripper's grip site."""
        site = self.sim.model.site_name2id(self.robot.controller.eef_name)
        mat = np.eye(4)
        mat[:3, :3] = np.array(self.sim.data.site_xmat[site].reshape([3, 3]))
        mat[:3, -1] = np.array(self.sim.data.site_xpos[site])
        return mat

    def step_delta(self, delta_pos, delta_quat, gripper: float) -> np.ndarray:
        """Steps with a pose delta (position, wxyz rotation) and a gripper command; returns the action."""
        vec, theta = t3d.quaternions.quat2axangle(delta_quat)
        action = list(delta_pos) + list(theta * vec) + [gripper]
        self.env.step(action)
        return np.array(action)

    def gripper_qpos_indices(self) -> list[int]:
        """Indices of the gripper finger joints in ``qpos``."""
        indices = set()
        for joint in self.robot.gripper.joints:
            addr = self.sim.model.get_joint_qpos_addr(joint)
            if isinstance(addr, (tuple, list, np.ndarray)):
                indices.update(range(addr[0], addr[1]) if len(addr) == 2 else addr)
            else:
                indices.add(int(addr))
        return sorted(indices)
