import numpy as np
import pytest
import transforms3d as t3d

from resteer.steergen import interp


def _pose(pos, axis, angle):
    return np.concatenate([pos, t3d.quaternions.axangle2quat(axis, angle)])


def test_waypoints_include_endpoints_and_respect_spacing():
    start = _pose([0.0, 0.0, 1.0], [0, 0, 1], 0.0)
    end = _pose([0.1, 0.0, 1.0], [0, 0, 1], 0.2)
    waypoints = interp.interpolate_poses(start, end, max_xyz_delta=0.005, max_rot_delta=0.05)
    np.testing.assert_allclose(waypoints[0], start, atol=1e-12)
    np.testing.assert_allclose(waypoints[-1][:3], end[:3], atol=1e-12)
    assert abs(abs(np.dot(waypoints[-1][3:], end[3:])) - 1) < 1e-12
    # Position spacing 0.1 / 0.005 -> 21 waypoints (rotation needs fewer).
    assert len(waypoints) == int(0.1 / 0.005) + 1
    steps = np.linalg.norm(np.diff([w[:3] for w in waypoints], axis=0), axis=1)
    assert steps.max() <= 0.005 + 1e-12


def test_slerp_halfway_is_half_the_angle():
    q1 = t3d.quaternions.axangle2quat([0, 0, 1], 0.0)
    q2 = t3d.quaternions.axangle2quat([0, 0, 1], 1.0)
    _, angle = t3d.quaternions.quat2axangle(interp.slerp(q1, q2, 0.5))
    assert angle == pytest.approx(0.5)


def test_delta_to_reaches_target():
    current = _pose([0.1, 0.2, 0.3], [1, 0, 0], 0.3)
    target = _pose([0.15, 0.2, 0.25], [0, 1, 0], -0.2)
    delta_pos, delta_quat = interp.delta_to(current, target)
    np.testing.assert_allclose(current[:3] + delta_pos, target[:3])
    reached = t3d.quaternions.quat2mat(delta_quat) @ t3d.quaternions.quat2mat(current[3:])
    np.testing.assert_allclose(reached, t3d.quaternions.quat2mat(target[3:]), atol=1e-12)
    zero_pos, identity = interp.delta_to(current, current)
    np.testing.assert_allclose(zero_pos, 0)
    assert abs(identity[0]) == pytest.approx(1.0)


def test_scale_rotation_multiplies_the_angle():
    q = t3d.quaternions.axangle2quat([0, 0.6, 0.8], 0.1)
    axis, angle = t3d.quaternions.quat2axangle(interp.scale_rotation(q, 2.0))
    assert angle == pytest.approx(0.2)
    np.testing.assert_allclose(axis, [0, 0.6, 0.8])
    np.testing.assert_allclose(interp.scale_rotation(np.array([1.0, 0, 0, 0]), 2.0), [1, 0, 0, 0])


def test_axis_angle_to_quat_roundtrip():
    axis_angle = np.array([0.01, -0.02, 0.03])
    axis, angle = t3d.quaternions.quat2axangle(interp.axis_angle_to_quat(axis_angle))
    np.testing.assert_allclose(axis * angle, axis_angle, atol=1e-12)
    np.testing.assert_allclose(interp.axis_angle_to_quat(np.zeros(3)), [1, 0, 0, 0])


def test_is_noop():
    still = np.array([0, 0, 0, 0, 0, 0, -1.0])
    assert interp.is_noop(still)
    assert interp.is_noop(still, prev_action=still)
    assert not interp.is_noop(still, prev_action=np.array([0, 0, 0, 0, 0, 0, 1.0]))  # gripper command changed
    assert not interp.is_noop(np.array([0.01, 0, 0, 0, 0, 0, -1.0]), prev_action=still)
