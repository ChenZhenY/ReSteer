"""End-effector pose interpolation used to build SteerGen bridges.

Poses are 7-vectors ``[x, y, z, qw, qx, qy, qz]`` (transforms3d quaternion order). A bridge
from pose A to pose B linearly interpolates the position and SLERPs the orientation with enough
waypoints that consecutive waypoints differ by at most ``max_xyz_delta`` metres in position and
``max_rot_delta`` in quaternion-component distance.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import transforms3d as t3d


def pose_from_matrix(mat: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> ``[pos (3), quat wxyz (4)]``."""
    return np.concatenate([mat[:3, 3], t3d.quaternions.mat2quat(mat[:3, :3])])


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two wxyz quaternions (scipy, shortest path)."""
    rotations = Rotation.from_quat([np.array([q1[1], q1[2], q1[3], q1[0]]), np.array([q2[1], q2[2], q2[3], q2[0]])])
    xyzw = Slerp([0, 1], rotations)([t]).as_quat()[0]
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def interpolate_poses(
    start: np.ndarray, end: np.ndarray, max_xyz_delta: float = 0.005, max_rot_delta: float = 0.05
) -> list[np.ndarray]:
    """Waypoints from ``start`` to ``end`` (both included)."""
    start_pos, start_quat = np.asarray(start[:3]), np.asarray(start[3:])
    end_pos, end_quat = np.asarray(end[:3]), np.asarray(end[3:])
    num = max(
        int(np.linalg.norm(end_pos - start_pos) / max_xyz_delta) + 1,
        int(np.linalg.norm(end_quat - start_quat) / max_rot_delta) + 1,
    )
    start_quat = start_quat / np.linalg.norm(start_quat)
    end_quat = end_quat / np.linalg.norm(end_quat)
    waypoints = []
    for i in range(num):
        alpha = i / max(1, num - 1)
        waypoints.append(
            np.concatenate([(1 - alpha) * start_pos + alpha * end_pos, slerp(start_quat, end_quat, alpha)])
        )
    return waypoints


def delta_to(current: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position delta and rotation delta (wxyz quaternion, applied on the left) from current to target."""
    delta_rot = t3d.quaternions.quat2mat(target[3:7]) @ np.linalg.inv(t3d.quaternions.quat2mat(current[3:7]))
    return target[:3] - current[:3], t3d.quaternions.mat2quat(delta_rot)


def scale_rotation(delta_quat: np.ndarray, factor: float) -> np.ndarray:
    """Multiplies the rotation angle of a wxyz quaternion by ``factor`` (same axis)."""
    angle = 2 * np.arccos(np.clip(delta_quat[0], -1.0, 1.0))
    axis = delta_quat[1:]
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    scaled = angle * factor
    return np.concatenate([[np.cos(scaled / 2.0)], axis / norm * np.sin(scaled / 2.0)])


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle vector -> wxyz quaternion (identity for angles below 1e-6)."""
    angle = np.linalg.norm(axis_angle)
    rot = t3d.axangles.axangle2mat(axis_angle / angle, angle) if angle > 1e-6 else np.eye(3)
    return t3d.quaternions.mat2quat(rot)


def is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    """A LIBERO action that neither moves the arm nor changes the gripper command."""
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]
