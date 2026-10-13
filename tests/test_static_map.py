import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from mpc_controller.src.static_map import (StaticMap, yaw_from_compass, N_SLOTS,  # noqa: E402
                                           PER_SLOT, FAR_FWD, FEATURE_DIM)


def one_box(cx, cy, ex=0.5, ey=0.5, yaw=0.0):
    return StaticMap([cx], [cy], [ex], [ey], [yaw], ['Poles'])


def slots(f):
    return f.reshape(N_SLOTS, PER_SLOT)


def test_empty_map_window():
    f = one_box(500.0, 500.0).features(0.0, 0.0, 0.0)
    assert f.shape == (FEATURE_DIM,)
    assert np.all(slots(f)[:, 0] == 0) and np.all(slots(f)[:, 1] == FAR_FWD)


def test_object_ahead_and_lateral_sign():
    # Ego at origin facing +x; pole centred at (10, 3), 0.5 m half-size.
    s = slots(one_box(10.0, 3.0).features(0.0, 0.0, 0.0))
    assert s[0, 0] == 1.0
    assert np.isclose(s[0, 1], 9.5) and np.isclose(s[0, 2], 2.5)
    # Same scene, ego turned to face +y: the pole is now to the side/behind
    # in fwd (3 - 0.5) and lateral is -(10 - 0.5).
    s = slots(one_box(10.0, 3.0).features(0.0, 0.0, np.pi / 2))
    assert np.isclose(s[0, 1], 2.5) and np.isclose(s[0, 2], -9.5)


def test_object_behind_excluded():
    s = slots(one_box(-10.0, 0.0).features(0.0, 0.0, 0.0))
    assert s[0, 0] == 0.0


def test_nearest_point_of_long_rotated_box():
    # A 40 m wall along x at y = 4: nearest point is straight beside the ego,
    # not the far-away centre.
    sm = StaticMap([20.0], [4.0], [20.0], [0.2], [0.0], ['Walls'])
    s = slots(sm.features(5.0, 0.0, 0.0))
    assert s[0, 0] == 1.0 and np.isclose(s[0, 1], 0.0) and np.isclose(s[0, 2], 3.8)
    # Rotating the box by 180 deg is the same footprint.
    sm = StaticMap([20.0], [4.0], [20.0], [0.2], [np.pi], ['Walls'])
    assert np.isclose(slots(sm.features(5.0, 0.0, 0.0))[0, 2], 3.8)


def test_slots_sorted_and_capped():
    xs = np.arange(1, 13) * 2.0 + 2.0       # 12 poles ahead
    sm = StaticMap(xs, np.zeros(12), np.full(12, 0.1), np.full(12, 0.1),
                   np.zeros(12), ['Poles'] * 12)
    s = slots(sm.features(0.0, 0.0, 0.0))
    assert np.all(s[:, 0] == 1.0)
    assert np.all(np.diff(s[:, 1]) > 0)
    assert np.isclose(s[0, 1], xs[0] - 0.1)


def test_compass_convention():
    # CARLA: compass 0 faces north = -y (yaw -90 deg); facing east (+x) = pi/2.
    assert np.isclose(yaw_from_compass(np.pi / 2), 0.0)
    assert np.isclose(yaw_from_compass(0.0), -np.pi / 2)


def test_roundtrip_save_load(tmp_path):
    sm = one_box(1.0, 2.0)
    p = str(tmp_path / 'T.npz')
    sm.save(p)
    sm2 = StaticMap.load(p)
    assert np.allclose(sm.features(0, 0, 0), sm2.features(0, 0, 0))


def test_drop_boxes_covering_lane_points():
    # A 3 km "map-sized" box and a pole beside the lane; lane centre along y=0.
    sm = StaticMap([0.0, 10.0], [0.0, 3.0], [1500.0, 0.2], [1500.0, 0.2], [0.0, 0.0],
                   ['Vegetation', 'Poles'])
    lane = np.stack([np.arange(0, 50, 1.0), np.zeros(50)], axis=1)
    kept = sm.without_boxes_covering(lane)
    assert list(kept.label) == ['Poles']
    assert kept.dropped == {'Vegetation': 1}
    s = slots(kept.features(0.0, 0.0, 0.0))
    assert s[0, 0] == 1.0 and np.isclose(s[0, 2], 2.8)
