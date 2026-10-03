"""Labels the states of a state bank with manipulation stages and distance-to-go.

Stages come from the gripper: 0 = pre-grasp (until the gripper first closes), 1 = transport (until
it first re-opens), 2 = place (afterwards). The gripper counts as open while its first finger joint
is above ``--gripper-open-threshold`` (metres; a Panda finger spans 0-0.04). ``dist_to_go`` is the
end-effector path length remaining until the end of the demo. (The paper-era labeller stored only the
distance to the *next* state because its accumulation ran in the wrong direction;
``--legacy-dist-to-go`` reproduces those values.)

Labels are written into the bank as ``<task>_stages/demo_<i>`` and ``<task>_dist_to_go/demo_<i>``
(re-running replaces them).

    python -m resteer.steergen.label_stages data/states/libero_goal_regen.hdf5
"""

import dataclasses
import pathlib

import h5py
import numpy as np
import tyro

PREGRASP, TRANSPORT, PLACE = 0, 1, 2


def label_stages(aperture: np.ndarray, open_threshold: float = 0.03) -> np.ndarray:
    """Stage per state from the gripper aperture sequence."""
    is_open = aperture > open_threshold
    stages = np.zeros(len(aperture), dtype=int)
    closed = next((i for i in range(1, len(aperture)) if is_open[i - 1] and not is_open[i]), None)
    if closed is None:
        return stages
    reopened = next((i for i in range(closed + 1, len(aperture)) if is_open[i] and not is_open[i - 1]), None)
    stages[closed:] = TRANSPORT
    if reopened is not None:
        stages[reopened:] = PLACE
    return stages


def dist_to_go(positions: np.ndarray, legacy: bool = False) -> np.ndarray:
    """Remaining path length from every position to the last one (legacy: distance to the next one)."""
    steps = [np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    remaining = np.zeros(len(positions))
    for i in reversed(range(len(steps))):
        remaining[i] = steps[i] if legacy else remaining[i + 1] + steps[i]
    return remaining


def label_bank(path: pathlib.Path, open_threshold: float = 0.03, legacy_dist_to_go: bool = False) -> None:
    from resteer import states as _states
    from resteer.steergen.env import SteerGenEnv

    bank = _states.StateBank(path)
    for task_name in bank.task_names:
        env = SteerGenEnv("replay_delta")  # only used for forward kinematics
        nq = env.sim.model.nq
        finger = env.gripper_qpos_indices()[0]
        labels = {}
        for key in bank.demo_keys(task_name):
            states = bank.load(task_name, key)
            aperture = states[:, 1 : 1 + nq][:, finger]  # flattened state = [time, qpos, qvel]
            positions = []
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                positions.append(env.eef_pose()[:3, 3])
            demo = key[: -len("_states")]
            labels[demo] = (label_stages(aperture, open_threshold), dist_to_go(np.array(positions), legacy_dist_to_go))
        env.close()
        with h5py.File(path, "a") as f:
            for kind, index in (("stages", 0), ("dist_to_go", 1)):
                name = f"{task_name}_{kind}"
                if name in f:
                    del f[name]
                group = f.create_group(name)
                for demo, values in labels.items():
                    group.create_dataset(demo, data=values[index])
        print(f"{task_name}: labelled {len(labels)} demos")


@dataclasses.dataclass
class Args:
    states: tyro.conf.Positional[pathlib.Path]
    gripper_open_threshold: float = 0.03
    legacy_dist_to_go: bool = False
    """Store the paper-era values (distance to the next state instead of the remaining path length)."""


def main(args: Args) -> None:
    label_bank(args.states, args.gripper_open_threshold, args.legacy_dist_to_go)


if __name__ == "__main__":
    main(tyro.cli(Args))
