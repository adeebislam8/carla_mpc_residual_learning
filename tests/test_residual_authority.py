"""
Checks for the CBF-derived adaptive residual authority.

Runs without CARLA or acados -- the authority module is plain NumPy -- so this
is the one part of the control stack that can be exercised on any machine:

    python tests/test_residual_authority.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from mpc_controller.src.residual_authority import ResidualAuthority  # noqa: E402


# Geometry from bicycle_model_mpcc_cbf.py
LF, LR = 1.169, 1.801
MODEL = dict(kappa_fn=lambda s: 0.0, dt=0.05, C1=LR / (LR + LF),
             C2=1.0 / (LR + LF), delta_max=45 * np.pi / 180)

NO_OBS = np.full((6, 2), -100.0)

# Obstacle 8 m ahead; ego offset to its right (n = -2.5, obstacle at n = 0).
# The OCP negates the steering state (`delta = -delta` in
# bicycle_model_mpcc_cbf.py), so a NEGATIVE steering command raises n and
# closes the gap, and a positive one opens it.  This test used to assert the
# opposite, matching a gate that had dropped the negation -- see the module
# docstring of residual_authority.py.  With ego and obstacle both at n = 0
# the elliptical barrier is symmetric and the directions are indistinguishable.
BLOCKED = np.full((6, 2), -100.0)
BLOCKED[0] = [18.0, 0.0]
STATE = dict(s=10.0, n=-2.5, heading=0.0, v=12.0)


def test_full_authority_when_unobstructed():
    auth = ResidualAuthority(**MODEL)
    alpha, info = auth.compute(s=10.0, n=0.0, heading=0.0, v=10.0,
                               u_nom=(0.5, 0.0), du=np.array([0.1, 0.1]),
                               obstacles=NO_OBS)
    assert alpha == 1.0 and info['saturated']


def test_actuator_bound_alone_caps_alpha():
    auth = ResidualAuthority(**MODEL)
    # 0.95 + alpha * 0.5 <= throttle_max (1.0)  =>  alpha <= 0.1
    alpha, _ = auth.compute(s=10.0, n=0.0, heading=0.0, v=10.0,
                            u_nom=(0.95, 0.0), du=np.array([0.5, 0.0]),
                            obstacles=NO_OBS)
    assert 0.05 <= alpha <= 0.15


def test_residual_toward_obstacle_is_suppressed():
    auth = ResidualAuthority(**MODEL)
    away, _ = auth.compute(**STATE, u_nom=(0.5, 0.0),
                           du=np.array([0.0, +0.9]), obstacles=BLOCKED)
    into, _ = auth.compute(**STATE, u_nom=(0.5, 0.0),
                           du=np.array([0.0, -0.9]), obstacles=BLOCKED)
    assert into < away


def test_strict_mode_matches_spec_section_7():
    """With require_nominal_feasible, an infeasible nominal zeroes the residual."""
    strict = ResidualAuthority(require_nominal_feasible=True, **MODEL)
    alpha, info = strict.compute(**STATE, u_nom=(0.5, 0.0),
                                 du=np.array([0.0, -0.9]), obstacles=BLOCKED)
    assert info['nominal_infeasible'] and alpha == 0.0


def _random_case(seed):
    rng = np.random.default_rng(seed)
    obs = np.full((6, 2), -100.0)
    for k in range(rng.integers(0, 5)):
        obs[k] = [rng.uniform(2, 45), rng.uniform(-5, 5)]
    state = dict(s=float(rng.uniform(5, 60)), n=float(rng.uniform(-4, 4)),
                 heading=float(rng.uniform(-0.4, 0.4)),
                 v=float(rng.uniform(0.5, 28)))
    u_nom = (float(rng.uniform(-0.5, 1.0)), float(rng.uniform(-1, 1)))
    du = rng.uniform(-0.6, 0.6, size=2)
    return state, u_nom, du, obs


def test_non_degradation_over_random_states():
    """margin(alpha) >= min(0, margin(0)) -- the property the IROS review asked for."""
    auth = ResidualAuthority(**MODEL)
    for seed in range(2000):
        state, u_nom, du, obs = _random_case(seed)
        alpha, info = auth.compute(**state, u_nom=u_nom, du=du, obstacles=obs)
        assert 0.0 <= alpha <= 1.0
        assert info['margin_at_alpha'] >= min(0.0, info['nominal_margin']) - 1e-6


def test_returned_interval_is_a_prefix():
    """Every alpha' <= alpha_safe must also be admissible (spec Section 7)."""
    auth = ResidualAuthority(**MODEL)
    for seed in range(500):
        state, u_nom, du, obs = _random_case(10_000 + seed)
        alpha, info = auth.compute(**state, u_nom=u_nom, du=du, obstacles=obs)
        admissible = min(0.0, info['nominal_margin'])
        for frac in np.linspace(0.0, 1.0, 6):
            ok, _ = auth._feasible(
                info['alpha_safe'] * frac, state['s'], state['n'],
                state['heading'], state['v'], u_nom, du, obs, admissible)
            assert ok, (seed, frac)


def test_support_gate_only_reduces():
    auth = ResidualAuthority(**MODEL)
    alpha, info = auth.compute(s=10.0, n=0.0, heading=0.0, v=10.0,
                               u_nom=(0.5, 0.0), du=np.array([0.1, 0.1]),
                               obstacles=NO_OBS, support_gate=0.3)
    assert info['alpha_safe'] == 1.0 and abs(alpha - 0.3) < 1e-9


def test_steering_sign_matches_ocp_model():
    """bicycle_model_mpcc_cbf.py uses ndot = v*sin(alpha + C1*(-delta)):
    a positive steering command must DECREASE n."""
    auth = ResidualAuthority(horizon_steps=5, **MODEL)
    traj = auth.rollout(s=0.0, n=0.0, heading=0.0, v=10.0,
                        throttle=0.0, steering=+0.5)
    assert traj[-1, 1] < 0.0
    traj = auth.rollout(s=0.0, n=0.0, heading=0.0, v=10.0,
                        throttle=0.0, steering=-0.5)
    assert traj[-1, 1] > 0.0


def test_throttle_moves_speed_in_rollout():
    auth = ResidualAuthority(horizon_steps=10, **MODEL)
    fast = auth.rollout(0.0, 0.0, 0.0, 10.0, throttle=1.0, steering=0.0)
    slow = auth.rollout(0.0, 0.0, 0.0, 10.0, throttle=-0.5, steering=0.0)
    assert fast[-1, 3] > 10.0 > slow[-1, 3]
    assert fast[-1, 0] > slow[-1, 0]


# Obstacle dead ahead in our lane, closing range.
AHEAD = np.full((6, 2), -100.0)
AHEAD[0] = [16.0, 0.0]
AHEAD_STATE = dict(s=10.0, n=0.0, heading=0.0, v=12.0)


def test_one_step_check_is_blind_to_throttle():
    """The original gate: one Euler step with v held, so throttle cannot
    change the predicted barrier and accelerating into a car passes."""
    auth = ResidualAuthority(horizon_steps=1, **MODEL)
    _, base = auth.compute(**AHEAD_STATE, u_nom=(-0.5, 0.0),
                           du=np.array([0.0, 0.0]), obstacles=AHEAD)
    _, gas = auth.compute(**AHEAD_STATE, u_nom=(-0.5, 0.0),
                          du=np.array([1.5, 0.0]), obstacles=AHEAD)
    assert abs(base['nominal_margin'] - gas['nominal_margin']) < 1e-12
    alpha, _ = auth.compute(**AHEAD_STATE, u_nom=(-0.5, 0.0),
                            du=np.array([1.5, 0.0]), obstacles=AHEAD)
    assert alpha == 1.0


def test_rollout_check_suppresses_acceleration_into_obstacle():
    auth = ResidualAuthority(horizon_steps=10, **MODEL)
    gas, _ = auth.compute(**AHEAD_STATE, u_nom=(-0.5, 0.0),
                          du=np.array([1.5, 0.0]), obstacles=AHEAD)
    brake, _ = auth.compute(**AHEAD_STATE, u_nom=(0.5, 0.0),
                            du=np.array([-1.0, 0.0]), obstacles=AHEAD)
    assert gas < 1.0
    assert brake > gas


def test_corridor_blocks_steering_off_road():
    """Ego 0.3 m inside the left edge; a negative steering command raises n
    toward that edge.  Unchecked without a corridor, capped with one."""
    auth = ResidualAuthority(horizon_steps=10, **MODEL)
    kw = dict(s=10.0, n=1.7, heading=0.0, v=10.0, u_nom=(0.3, 0.0),
              du=np.array([0.0, -0.8]), obstacles=NO_OBS)
    free, _ = auth.compute(**kw)
    capped, info = auth.compute(**kw, corridor=(-2.0, 2.0))
    assert free == 1.0
    assert capped < 1.0
    assert info['corridor_margin_at_alpha'] >= -1e-9


def test_non_degradation_with_rollout_and_corridor():
    auth = ResidualAuthority(horizon_steps=10, **MODEL)
    for seed in range(300):
        state, u_nom, du, obs = _random_case(20_000 + seed)
        corridor = (-4.5, 4.5)
        alpha, info = auth.compute(**state, u_nom=u_nom, du=du, obstacles=obs,
                                   corridor=corridor)
        assert 0.0 <= alpha <= 1.0
        assert info['margin_at_alpha'] >= min(0.0, info['nominal_margin']) - 1e-6
        assert (info['corridor_margin_at_alpha']
                >= min(0.0, info['nominal_corridor_margin']) - 1e-6)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'  ok  {name}')
    print('\nall checks passed')
