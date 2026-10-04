import os
import numpy as np
import robosuite as suite

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

MAX_RESET_RETRIES = 100


class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        controller_configs = suite.load_controller_config(default_controller=controller)
        self.controller_configs = controller_configs.copy() # for access only

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so
      
        # print(f"controller_configs: {controller_configs}")

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
     
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action, return_success_dict=False):
        return self.env.step(action, return_success_dict=return_success_dict)

    def reset(self):
        # Upstream used `finally: continue`, which silently swallows *every* exception
        # (e.g. EGL/rendering errors) and loops forever. Only retry object-placement
        # failures, and only a bounded number of times.
        for _ in range(MAX_RESET_RETRIES):
            try:
                return self.env.reset()
            except RandomizationError:
                continue
        raise RuntimeError(f"env.reset() hit RandomizationError {MAX_RESET_RETRIES} times in a row")

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        super().__init__(**kwargs)


class CustomControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller_configs=None,
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        self.controller_configs = controller_configs.copy() # for access only

        
        # Check if we're using a multi-armed environment and use env_configuration argument if so
      
        # print(f"controller_configs: {controller_configs}")

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
     
        self.env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action, return_success_dict=False):
        return self.env.step(action, return_success_dict=return_success_dict)

    def reset(self):
        # Upstream used `finally: continue`, which silently swallows *every* exception
        # (e.g. EGL/rendering errors) and loops forever. Only retry object-placement
        # failures, and only a bounded number of times.
        for _ in range(MAX_RESET_RETRIES):
            try:
                return self.env.reset()
            except RandomizationError:
                continue
        raise RuntimeError(f"env.reset() hit RandomizationError {MAX_RESET_RETRIES} times in a row")

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        self.env.seed(seed)

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env
