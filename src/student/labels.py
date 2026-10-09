"""
Teacher (ORION) output -> the residual target the student regresses.

Kept free of torch/CARLA so the dataset builder, the comparison tool and the
tests all share exactly one definition.

Conventions (see tools/orion/record_orion_episodes.py):
  * Model steering > 0 steers LEFT; the env sends control.steer = -steering.
    ORION's PID outputs CARLA steer (> 0 = right): model steering = -ORION steer.
  * Model throttle is signed (brake < 0).  ORION's PID brake is bang-bang, so
    the longitudinal label instead comes from ORION's planned speed, as a
    RELATIVE correction to the MPCC's throttle (teacher_throttle_relative) --
    the model's absolute throttle is mis-calibrated against CARLA.
  * The env applies du = action * residual_max, so the student's target is
    the normalised action a* = clip(du* / residual_max, -1, 1).
"""

import numpy as np

# Longitudinal model of bicycle_model_mpcc_cbf.py (same constants as
# residual_authority._MODEL_LONG).
_M, _CM1, _CM2 = 2065.03, 9.36424211e+03, 4.08690122e+01
_CR0, _CR2, _CR3 = 5.84856121e+02, 2.04799356e+00, 1.13995833e+01
D_MIN, D_MAX = -0.5, 1.0          # the authority gate's actuator bounds


def throttle_from_speed(v_target, v, tau=1.0):
    """
    Throttle D that reaches v_target in tau seconds under the MPCC's own
    longitudinal model, inverting Fxd = (Cm1 - Cm2 v) D - Cr2 v^2
    - Cr0 tanh(Cr3 v), a = Fxd / m.  Vectorised.

    tau = 1 s: ORION's desired speed (its PID's formula) is already the speed
    ~0.5-1 s down the plan; at 0.5 s a 2 m/s gap saturates the throttle and
    recreates the bang-bang label.  On the first recorded episode this cut the
    share of clipped throttle targets from 56% (PID) to 25%.
    """
    v_target = np.asarray(v_target, dtype=float)
    v = np.asarray(v, dtype=float)
    a = (v_target - v) / tau
    D = (_M * a + _CR2 * v * v + _CR0 * np.tanh(_CR3 * v)) / (_CM1 - _CM2 * v)
    return np.clip(D, D_MIN, D_MAX)


def plan_speed(speeds, dt=0.05):
    """
    ORION's 'desired speed' formula (its PID: 0.75 * |wp0| * 2 + 0.25 *
    |wp1 - wp0| * 2, waypoints 0.5 s apart) applied to a REALISED speed
    trace: 0.75 * mean speed over the next 0.5 s + 0.25 * mean over 0.5-1.0 s.
    Frames near the end of the episode use what remains.  speeds: (T,).
    """
    v = np.asarray(speeds, dtype=float)
    T, h = len(v), int(round(0.5 / dt))
    out = np.empty(T)
    for i in range(T):
        a = v[i + 1:i + 1 + h]
        b = v[i + 1 + h:i + 1 + 2 * h]
        a = a if len(a) else v[i:i + 1]
        b = b if len(b) else a
        out[i] = 0.75 * a.mean() + 0.25 * b.mean()
    return out


def teacher_throttle_relative(v_orion, v_realised, v, u_nom_throttle, tau=1.0):
    """
    u_teacher = u_nom + D(v_orion) - D(v_realised): the MPCC's own throttle,
    shifted by the model's estimate of the throttle difference between
    reaching ORION's planned speed and reaching the speed the MPCC actually
    reached.

    Why relative: the bicycle model's longitudinal constants are not CARLA's.
    Measured on the first real episodes, the model holds speed at D ~ 0.17
    where the car needs ~ 0.42, so the ABSOLUTE label asked for 0.25 less
    throttle than the MPCC even on frames where ORION planned exactly the
    speed the MPCC reached -- a uniform 'slow down' bias.  The offset cancels
    in the difference; only the model's gain remains.
    """
    # Unclipped linear form of D(v_orion) - D(v_realised): the resistance
    # terms cancel exactly, and clipping each term first would read a
    # saturated pair as no difference.
    a = (np.asarray(v_orion, float) - np.asarray(v_realised, float)) / tau
    delta = _M * a / (_CM1 - _CM2 * np.asarray(v, float))
    return np.clip(np.asarray(u_nom_throttle, float) + delta, D_MIN, D_MAX)


def teacher_action(orion_ctrl, desired_speed, ego_speed, tau=1.0):
    """
    ORION output -> (throttle, steering) in the MPC model's convention.

    orion_ctrl: (..., 3) steer, throttle, brake from ORION's PID
    desired_speed, ego_speed: (...,) m/s
    """
    orion_ctrl = np.asarray(orion_ctrl, dtype=float)
    thr = throttle_from_speed(desired_speed, ego_speed, tau)
    steer = -orion_ctrl[..., 0]
    return np.stack([thr, steer], axis=-1)


def residual_target(u_teacher, u_nom, residual_max):
    """Normalised residual action the student should output."""
    du = np.asarray(u_teacher, dtype=float) - np.asarray(u_nom, dtype=float)
    return np.clip(du / residual_max, -1.0, 1.0)
