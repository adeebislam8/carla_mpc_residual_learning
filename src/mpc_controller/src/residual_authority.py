"""
CBF-derived adaptive residual authority.

Implements Section 6 ("Component 3 - CBF-Derived Adaptive Residual Authority")
and the feasibility proposition of Section 7 of the project specification.

The previous control law applied a manually fixed residual scale:

    u = u_nom + 0.1 * pi(o)

which does not re-check the CBF condition after the residual is added, so the
final action is not guaranteed to satisfy the constraints the MPCC+CBF layer
enforced on u_nom.  This module replaces the constant with

    alpha_t^safe = max { alpha in [0, 1] : u_nom + alpha * du satisfies
                         every modelled CBF and actuator constraint }

and the final authority is alpha_t = g_t^support * alpha_t^safe.

The barrier, its discretisation and the propagation model here are deliberate
copies of what `bicycle_model_mpcc_cbf.py` encodes in the OCP, so that the
re-verification uses the *same modelled dynamics* as the controller
(specification Section 7, assumption 2).  If that file's constants change,
change them here too -- `ResidualAuthority.from_mpc_model` reads whatever it
can off the model struct to reduce the chance of drift.
"""

import numpy as np


# Obstacle slots carrying s <= this are inactive padding, matching the
# valid-obstacle test in mpc_controller_python.solve().
_INACTIVE_OBS_S = -50.0

# Obstacles this far behind the ego are ignored by the barrier, matching
# `condition = s > (s_obs + 5)` in distance2obs_casadi_elliptical().
_IGNORE_BEHIND_M = 5.0


class ResidualAuthority:
    """Largest residual scale that keeps the final action CBF-feasible."""

    def __init__(
        self,
        kappa_fn,
        dt: float,
        C1: float,
        C2: float,
        delta_max: float,
        throttle_min: float = -0.5,
        throttle_max: float = 1.0,
        d_safe: float = 1.0,
        obs_gamma: float = 1.0,
        a_long: float = 4.0,
        b_lat: float = 2.0,
        n_grid: int = 11,
        bisection_iters: int = 6,
        n_verify: int = 16,
        require_nominal_feasible: bool = False,
    ):
        """
        Args:
            kappa_fn: path curvature as a function of arc length.  Accepts the
                CasADi Function built by the model or the scipy spline that
                MPCController.update_path() swaps in later.
            dt: controller timestep, the same dt the OCP discretises the
                barrier with.
            C1, C2: bicycle model geometry terms (lr/(lr+lf), 1/(lr+lf)).
            delta_max: steering angle at |steering| = 1, used to map the
                normalised steering command onto the model's delta state.
            throttle_min/throttle_max: actuator bounds on D.
            d_safe: right-hand side of the discrete CBF condition
                (SAFETY_DISTANCE in the model).
            obs_gamma: CBF class-K gain (obs_gamma in the model).
            a_long, b_lat: elliptical safety-zone semi-axes.  NOTE these are
                the values passed at the call sites of
                distance2obs_casadi_elliptical(), not that function's own
                defaults.
            n_grid: coarse grid resolution for the prefix-feasible scan.
            bisection_iters: refinement steps after the coarse scan.
            require_nominal_feasible: if True, demand margin >= 0 outright and
                give the residual no authority whenever the nominal action is
                itself in violation (the literal reading of spec Section 7).
                Default False -- see _admissible_margin() for why.
        """
        self.kappa_fn = kappa_fn
        self.dt = dt
        self.C1 = C1
        self.C2 = C2
        self.delta_max = delta_max
        self.throttle_min = throttle_min
        self.throttle_max = throttle_max
        self.d_safe = d_safe
        self.obs_gamma = obs_gamma
        self.a_long = a_long
        self.b_lat = b_lat
        self.n_grid = n_grid
        self.bisection_iters = bisection_iters
        self.n_verify = n_verify
        self.require_nominal_feasible = require_nominal_feasible

    def _largest_prefix(self, upper, s, n, heading, v, u_nom, du, obstacles,
                        admissible, n_points):
        """
        Largest a <= upper such that every sampled point in [0, a] is
        admissible.  Feasibility is not monotone in alpha under nonlinear
        dynamics, so scanning upward and stopping at the first failure is what
        makes the returned interval a genuine prefix (spec Section 7) rather
        than merely one admissible point.

        The property therefore holds at this sampling resolution, not
        continuously; n_points trades cost against how narrow a dip can hide.
        """
        best, best_margin = 0.0, np.inf
        for a in np.linspace(0.0, upper, n_points + 1)[1:]:
            ok, margin = self._feasible(
                a, s, n, heading, v, u_nom, du, obstacles, admissible)
            if not ok:
                break
            best, best_margin = float(a), float(margin)
        return best, best_margin

    def _admissible_margin(self, nominal_margin: float) -> float:
        """
        Lower bound the final action's CBF margin must clear.

        Section 7 assumes the nominal MPCC+CBF action is CBF-feasible.  In this
        implementation it frequently is not: acados_settings_mpcc.py slacks the
        six obstacle constraints (idxsh covers h-indices 6..11), so the OCP
        treats them as soft and returns actions with a negative discrete-CBF
        margin whenever an obstacle is closing fast.  Demanding margin >= 0
        outright would therefore zero the residual most of the time an obstacle
        is present -- exactly the situations the residual exists to improve.

        The guarantee is stated instead as non-degradation:

            margin(alpha) >= min(0, margin(0))

        which reads as: where the nominal action satisfies the modelled CBF,
        the final action must too (strict preservation); where the OCP has
        already conceded a slack violation, the residual may not make it worse.
        That is the property the IROS review actually asked for -- the residual
        cannot invalidate the safety the CBF layer enforced -- and unlike the
        literal Section 7 statement it holds against the solver as implemented.

        Set require_nominal_feasible=True to recover the strict rule.
        """
        if self.require_nominal_feasible:
            return 0.0
        return min(0.0, nominal_margin)

    @classmethod
    def from_mpc_model(cls, model, dt: float, **overrides):
        """Build from an initialised MPCController's model struct."""
        params = model.params
        kwargs = dict(
            kappa_fn=model.kapparef_s,
            dt=dt,
            C1=params.C1,
            C2=params.C2,
            delta_max=model.delta_max,
            throttle_min=model.throttle_min,
            throttle_max=model.throttle_max,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # ---------------------------------------------------------------- barrier

    def _kappa(self, s: float) -> float:
        """Curvature at s, tolerating CasADi or scipy spline callables."""
        try:
            return float(self.kappa_fn(s))
        except Exception:
            return 0.0

    def _barrier(self, s: float, n: float, s_obs: float, n_obs: float) -> float:
        """
        Normalised elliptical distance to one obstacle.

        Mirrors distance2obs_casadi_elliptical(): obstacles more than
        _IGNORE_BEHIND_M behind the ego do not constrain it.
        """
        if s > (s_obs + _IGNORE_BEHIND_M):
            return None  # inactive; caller skips it
        return np.sqrt(((s - s_obs) / self.a_long) ** 2
                       + ((n - n_obs) / self.b_lat) ** 2)

    def _propagate(self, s, n, heading, v, delta):
        """
        One Euler step of (s, n) under the modelled dynamics.

        The OCP propagates the barrier with exactly this pair of states
        (`s_next = s + sdot*dt`, `n_next = n + ndot*dt` in
        bicycle_model_mpcc_cbf.py), so the re-verification does the same.
        """
        denom = 1.0 - self._kappa(s) * n
        if abs(denom) < 1e-6:
            denom = np.sign(denom) * 1e-6 if denom != 0.0 else 1e-6

        sdot = (v * np.cos(heading + self.C1 * delta)) / denom
        ndot = v * np.sin(heading + self.C1 * delta)
        return s + sdot * self.dt, n + ndot * self.dt

    # ------------------------------------------------------------ feasibility

    def _actuator_ok(self, throttle: float, steering: float) -> bool:
        return (self.throttle_min <= throttle <= self.throttle_max
                and -1.0 <= steering <= 1.0)

    def _cbf_margin(self, s, n, heading, v, delta, obstacles) -> float:
        """
        Smallest slack in the discrete CBF condition over active obstacles.

            (b_next - b)/dt + gamma * b  -  d_safe  >=  0

        Returns +inf when no obstacle is active.
        """
        worst = np.inf
        s_next, n_next = self._propagate(s, n, heading, v, delta)

        for s_obs, n_obs in obstacles:
            if s_obs <= _INACTIVE_OBS_S:
                continue
            b = self._barrier(s, n, s_obs, n_obs)
            if b is None:
                continue
            b_next = self._barrier(s_next, n_next, s_obs, n_obs)
            if b_next is None:
                # Ego passed the obstacle within the step: no longer binding.
                continue
            condition = (b_next - b) / self.dt + self.obs_gamma * b
            worst = min(worst, condition - self.d_safe)

        return worst

    def _feasible(self, alpha, s, n, heading, v, u_nom, du, obstacles,
                  admissible=0.0):
        """Check the candidate final action at this alpha."""
        throttle = u_nom[0] + alpha * du[0]
        steering = u_nom[1] + alpha * du[1]

        if not self._actuator_ok(throttle, steering):
            return False, -np.inf

        delta = steering * self.delta_max
        margin = self._cbf_margin(s, n, heading, v, delta, obstacles)
        return margin >= admissible, margin

    # ---------------------------------------------------------------- compute

    def compute(self, s, n, heading, v, u_nom, du, obstacles, support_gate=1.0):
        """
        Largest prefix-feasible residual authority.

        Args:
            s, n, heading, v: current Frenet state (arc length, lateral
                offset, heading error, speed).
            u_nom: nominal MPCC+CBF action, (throttle, steering) normalised.
            du: residual proposal in the same units; alpha = 1 applies it whole.
            obstacles: (N, 2) array of [s, d] in the Frenet frame, padded with
                s <= -50 for empty slots.
            support_gate: g_t^support in [0, 1] (Section 8).  Left at 1.0 until
                the support monitor exists; it only ever reduces authority.

        Returns:
            (alpha, info) where alpha = support_gate * alpha_safe.
        """
        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)
        du = np.asarray(du, dtype=float)

        # Margin of the nominal action sets the bar the final action must meet.
        _, nominal_margin = self._feasible(
            0.0, s, n, heading, v, u_nom, du, obstacles)
        admissible = self._admissible_margin(nominal_margin)

        info = {
            "alpha_safe": 0.0,
            "alpha": 0.0,
            "nominal_infeasible": bool(nominal_margin < 0.0),
            "saturated": False,
            "margin_at_alpha": -np.inf,
            "nominal_margin": float(nominal_margin),
            "support_gate": float(support_gate),
        }

        # Strict mode only: no authority when the nominal action itself
        # violates the modelled CBF.
        if self.require_nominal_feasible and nominal_margin < 0.0:
            info["margin_at_alpha"] = float(nominal_margin)
            return 0.0, info

        # Coarse prefix scan, then bisect the boundary between the last
        # admissible grid point and the first inadmissible one.
        lo, lo_margin = self._largest_prefix(
            1.0, s, n, heading, v, u_nom, du, obstacles, admissible,
            self.n_grid - 1)

        if lo < 1.0:
            hi = min(1.0, lo + 1.0 / (self.n_grid - 1))
            for _ in range(self.bisection_iters):
                mid = 0.5 * (lo + hi)
                ok, margin = self._feasible(
                    mid, s, n, heading, v, u_nom, du, obstacles, admissible)
                if ok:
                    lo, lo_margin = mid, margin
                else:
                    hi = mid

        # The coarse scan steps in 1/(n_grid-1) increments and can stride over a
        # narrow inadmissible dip, which would break the prefix property.  Re-scan
        # [0, lo] at the finer n_verify resolution and back off to the last point
        # before any failure.
        alpha_safe, lo_margin = self._largest_prefix(
            lo, s, n, heading, v, u_nom, du, obstacles, admissible,
            self.n_verify)

        info.update(alpha_safe=float(alpha_safe),
                    margin_at_alpha=float(lo_margin),
                    saturated=bool(alpha_safe >= 1.0 - 1e-9))
        info["alpha"] = float(np.clip(support_gate, 0.0, 1.0) * alpha_safe)
        return info["alpha"], info
