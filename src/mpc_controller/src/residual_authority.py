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
                         every modelled CBF, corridor and actuator constraint }

and the final authority is alpha_t = g_t^support * alpha_t^safe.

The barrier, its discretisation and the propagation model here are deliberate
copies of what `bicycle_model_mpcc_cbf.py` encodes in the OCP, so that the
re-verification uses the *same modelled dynamics* as the controller
(specification Section 7, assumption 2).  If that file's constants change,
change them here too -- `ResidualAuthority.from_mpc_model` reads whatever it
can off the model struct to reduce the chance of drift.

Changes 2026-10-08 (all three are needed before the residual may be large):

* STEERING SIGN.  The OCP negates the steering state before using it
  (`delta = -delta` in bicycle_model_mpcc_cbf.py, and again in
  MPCController's own propagator).  This module did not, so from 2026-09-09
  until this fix the gate predicted lateral motion MIRRORED: steering toward
  an obstacle looked safe and steering away looked unsafe.  Every adaptive
  (b5) result before this date was gated on that mirrored model.
* THROTTLE.  The old check was one explicit-Euler step of (s, n) with v held
  constant, so throttle could not move the predicted barrier at all -- the
  gate was blind to unsafe acceleration.  `horizon_steps` > 1 now rolls the
  held final action forward with the model's longitudinal dynamics (Fxd/m)
  and heading dynamics, and applies the discrete CBF condition at every step.
  horizon_steps = 1 reproduces the one-step check (with the sign fixed).
* ROAD CORRIDOR.  The OCP bounds n to the drivable corridor; the gate did not
  check it, so a large steering residual could leave the road unchecked.
  `corridor=(n_min, n_max)` adds the bound over the same rollout, under the
  same non-degradation rule as the barrier.
"""

import numpy as np


# Obstacle slots carrying s <= this are inactive padding, matching the
# valid-obstacle test in mpc_controller_python.solve().
_INACTIVE_OBS_S = -50.0

# Obstacles this far behind the ego are ignored by the barrier, matching
# `condition = s > (s_obs + 5)` in distance2obs_casadi_elliptical().
_IGNORE_BEHIND_M = 5.0

# Longitudinal model constants from bicycle_model_mpcc_cbf.py.  The model
# struct exposes Cm1, Cm2, Cr0, Cr2 on model.params but not m or Cr3.
_MODEL_LONG = dict(m=2065.03, Cm1=9.36424211e+03, Cm2=4.08690122e+01,
                   Cr0=5.84856121e+02, Cr2=2.04799356e+00, Cr3=1.13995833e+01)


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
        horizon_steps: int = 1,
        m: float = _MODEL_LONG['m'],
        Cm1: float = _MODEL_LONG['Cm1'],
        Cm2: float = _MODEL_LONG['Cm2'],
        Cr0: float = _MODEL_LONG['Cr0'],
        Cr2: float = _MODEL_LONG['Cr2'],
        Cr3: float = _MODEL_LONG['Cr3'],
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
            horizon_steps: how many dt steps the held final action is rolled
                forward.  1 = the original one-step check, which cannot see
                throttle.  10 (0.5 s) is what a large residual needs.
            m, Cm1, Cm2, Cr0, Cr2, Cr3: longitudinal model, Fxd/m.
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
        self.horizon_steps = max(1, int(horizon_steps))
        self.m, self.Cm1, self.Cm2 = m, Cm1, Cm2
        self.Cr0, self.Cr2, self.Cr3 = Cr0, Cr2, Cr3

    def _largest_prefix(self, upper, state, u_nom, du, obstacles, admissible,
                        corridor, n_points):
        """
        Largest a <= upper such that every sampled point in [0, a] is
        admissible.  Feasibility is not monotone in alpha under nonlinear
        dynamics, so scanning upward and stopping at the first failure is what
        makes the returned interval a genuine prefix (spec Section 7) rather
        than merely one admissible point.

        The property therefore holds at this sampling resolution, not
        continuously; n_points trades cost against how narrow a dip can hide.
        """
        best, best_margins = 0.0, (np.inf, np.inf)
        for a in np.linspace(0.0, upper, n_points + 1)[1:]:
            ok, margins = self._feasible(
                a, *state, u_nom, du, obstacles, admissible, corridor)
            if not ok:
                break
            best, best_margins = float(a), margins
        return best, best_margins

    def _admissible_margin(self, nominal_margin: float) -> float:
        """
        Lower bound the final action's margin must clear.

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

        The same rule is applied to the corridor margin.

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
        for name in ('Cm1', 'Cm2', 'Cr0', 'Cr2'):
            if hasattr(params, name):
                kwargs[name] = getattr(params, name)
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

    def rollout(self, s, n, heading, v, throttle, steering, steps=None):
        """
        Hold (throttle, steering) for `steps` explicit-Euler steps of the
        OCP's model.  Returns an array of shape (steps + 1, 4) holding
        (s, n, heading, v), row 0 being the current state.

        Matches bicycle_model_mpcc_cbf.py, including its `delta = -delta`:
        the normalised steering command maps to the model's delta state as
        steering * delta_max, and the dynamics use the NEGATED state.

        Curvature is evaluated once at the current s and held over the
        rollout.  At 0.5 s and ~10 m/s that is ~5 m of road, and one CasADi
        call per Euler step would dominate the cost of the whole gate.
        """
        steps = self.horizon_steps if steps is None else max(1, int(steps))
        delta = -float(steering) * self.delta_max
        D = float(throttle)
        kappa = self._kappa(s)
        cd, sd = np.cos(self.C1 * delta), self.C1 * delta

        traj = np.empty((steps + 1, 4))
        traj[0] = (s, n, heading, v)
        for k in range(steps):
            denom = 1.0 - kappa * n
            if abs(denom) < 1e-6:
                denom = 1e-6 if denom >= 0.0 else -1e-6
            sdot = v * np.cos(heading + sd) / denom
            ndot = v * np.sin(heading + sd)
            hdot = v * self.C2 * delta - kappa * sdot
            fxd = ((self.Cm1 - self.Cm2 * v) * D - self.Cr2 * v * v
                   - self.Cr0 * np.tanh(self.Cr3 * v))
            vdot = fxd / self.m * cd
            s, n = s + sdot * self.dt, n + ndot * self.dt
            heading = heading + hdot * self.dt
            v = max(0.0, v + vdot * self.dt)
            traj[k + 1] = (s, n, heading, v)
        return traj

    # ------------------------------------------------------------ feasibility

    def _actuator_ok(self, throttle: float, steering: float) -> bool:
        return (self.throttle_min <= throttle <= self.throttle_max
                and -1.0 <= steering <= 1.0)

    def cbf_margin(self, traj, obstacles) -> float:
        """
        Smallest slack in the discrete CBF condition over active obstacles and
        every step of the rollout:

            (b_{k+1} - b_k)/dt + gamma * b_k  -  d_safe  >=  0

        Returns +inf when no obstacle is active.
        """
        worst = np.inf
        for s_obs, n_obs in obstacles:
            if s_obs <= _INACTIVE_OBS_S:
                continue
            for k in range(len(traj) - 1):
                b = self._barrier(traj[k, 0], traj[k, 1], s_obs, n_obs)
                if b is None:
                    break      # ego has passed this obstacle
                b_next = self._barrier(traj[k + 1, 0], traj[k + 1, 1],
                                       s_obs, n_obs)
                if b_next is None:
                    break
                condition = (b_next - b) / self.dt + self.obs_gamma * b
                worst = min(worst, condition - self.d_safe)
        return worst

    @staticmethod
    def corridor_margin(traj, corridor) -> float:
        """Smallest distance of the rolled-out n to either corridor edge
        (negative = outside).  +inf when no corridor is given."""
        if corridor is None:
            return np.inf
        n_min, n_max = corridor
        n = traj[1:, 1]
        return float(min(np.min(n - n_min), np.min(n_max - n)))

    def _feasible(self, alpha, s, n, heading, v, u_nom, du, obstacles,
                  admissible=0.0, corridor=None):
        """
        Check the candidate final action at this alpha.

        admissible: (cbf_bound, corridor_bound), or a single float for the
        barrier alone.  Returns (ok, (cbf_margin, corridor_margin)).
        """
        if np.isscalar(admissible):
            admissible = (float(admissible), -np.inf)
        throttle = u_nom[0] + alpha * du[0]
        steering = u_nom[1] + alpha * du[1]

        if not self._actuator_ok(throttle, steering):
            return False, (-np.inf, -np.inf)

        traj = self.rollout(s, n, heading, v, throttle, steering)
        cbf = self.cbf_margin(traj, obstacles)
        cor = self.corridor_margin(traj, corridor)
        return (cbf >= admissible[0] and cor >= admissible[1]), (cbf, cor)

    # ---------------------------------------------------------------- compute

    def compute(self, s, n, heading, v, u_nom, du, obstacles, support_gate=1.0,
                corridor=None):
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
            corridor: (n_min, n_max) drivable bounds at the current s, or None
                to skip the corridor check.

        Returns:
            (alpha, info) where alpha = support_gate * alpha_safe.
        """
        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)
        du = np.asarray(du, dtype=float)
        state = (s, n, heading, v)

        # Margins of the nominal action set the bar the final action must meet.
        _, (nom_cbf, nom_cor) = self._feasible(
            0.0, *state, u_nom, du, obstacles, (-np.inf, -np.inf), corridor)
        admissible = (self._admissible_margin(nom_cbf),
                      self._admissible_margin(nom_cor))

        info = {
            "alpha_safe": 0.0,
            "alpha": 0.0,
            "nominal_infeasible": bool(nom_cbf < 0.0),
            "nominal_off_corridor": bool(nom_cor < 0.0),
            "saturated": False,
            "margin_at_alpha": -np.inf,
            "corridor_margin_at_alpha": -np.inf,
            "nominal_margin": float(nom_cbf),
            "nominal_corridor_margin": float(nom_cor),
            "support_gate": float(support_gate),
        }

        # Strict mode only: no authority when the nominal action itself
        # violates the modelled CBF or leaves the corridor.
        if self.require_nominal_feasible and (nom_cbf < 0.0 or nom_cor < 0.0):
            info["margin_at_alpha"] = float(nom_cbf)
            info["corridor_margin_at_alpha"] = float(nom_cor)
            return 0.0, info

        # Coarse prefix scan, then bisect the boundary between the last
        # admissible grid point and the first inadmissible one.
        lo, _ = self._largest_prefix(
            1.0, state, u_nom, du, obstacles, admissible, corridor,
            self.n_grid - 1)

        if lo < 1.0:
            hi = min(1.0, lo + 1.0 / (self.n_grid - 1))
            for _ in range(self.bisection_iters):
                mid = 0.5 * (lo + hi)
                ok, _ = self._feasible(
                    mid, *state, u_nom, du, obstacles, admissible, corridor)
                if ok:
                    lo = mid
                else:
                    hi = mid

        # The coarse scan steps in 1/(n_grid-1) increments and can stride over a
        # narrow inadmissible dip, which would break the prefix property.  Re-scan
        # [0, lo] at the finer n_verify resolution and back off to the last point
        # before any failure.
        alpha_safe, margins = self._largest_prefix(
            lo, state, u_nom, du, obstacles, admissible, corridor,
            self.n_verify)
        if alpha_safe == 0.0:
            margins = (nom_cbf, nom_cor)

        info.update(alpha_safe=float(alpha_safe),
                    margin_at_alpha=float(margins[0]),
                    corridor_margin_at_alpha=float(margins[1]),
                    saturated=bool(alpha_safe >= 1.0 - 1e-9))
        info["alpha"] = float(np.clip(support_gate, 0.0, 1.0) * alpha_safe)
        return info["alpha"], info
