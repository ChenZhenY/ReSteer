import numpy as np

from resteer.steergen import generate
from resteer.steergen.label_stages import PLACE
from resteer.steergen.label_stages import PREGRASP
from resteer.steergen.label_stages import TRANSPORT
from resteer.steergen.label_stages import dist_to_go
from resteer.steergen.label_stages import label_stages

OPEN, CLOSED = 0.04, 0.0


def test_full_pick_and_place_cycle():
    aperture = np.array([OPEN, OPEN, CLOSED, CLOSED, OPEN, OPEN])
    np.testing.assert_array_equal(label_stages(aperture), [PREGRASP, PREGRASP, TRANSPORT, TRANSPORT, PLACE, PLACE])


def test_gripper_never_reopens():
    np.testing.assert_array_equal(label_stages(np.array([OPEN, CLOSED, CLOSED])), [PREGRASP, TRANSPORT, TRANSPORT])


def test_gripper_never_closes_or_starts_closed():
    np.testing.assert_array_equal(label_stages(np.full(4, OPEN)), [PREGRASP] * 4)
    # A gripper that starts closed has no open->closed transition: everything is pre-grasp.
    np.testing.assert_array_equal(label_stages(np.array([CLOSED, CLOSED, OPEN])), [PREGRASP] * 3)


def test_threshold_is_strict():
    np.testing.assert_array_equal(label_stages(np.array([0.031, 0.03]), open_threshold=0.03), [PREGRASP, TRANSPORT])


def test_dist_to_go():
    positions = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.1, 0.2, 0], [0.1, 0.2, 0]])
    np.testing.assert_allclose(dist_to_go(positions), [0.3, 0.2, 0.0, 0.0])
    # Paper-era labels: distance to the next state only.
    np.testing.assert_allclose(dist_to_go(positions, legacy=True), [0.1, 0.2, 0.0, 0.0])


def test_transport_groups():
    assert generate.transportable("put_the_bowl_on_the_plate", "put_the_bowl_on_the_stove")
    assert generate.transportable("put_the_wine_bottle_on_the_rack", "put_the_wine_bottle_on_top_of_the_cabinet")
    assert not generate.transportable("put_the_bowl_on_the_plate", "put_the_bowl_on_the_plate")
    assert not generate.transportable("put_the_bowl_on_the_plate", "put_the_wine_bottle_on_the_rack")
    assert not generate.transportable("turn_on_the_stove", "put_the_bowl_on_the_stove")
