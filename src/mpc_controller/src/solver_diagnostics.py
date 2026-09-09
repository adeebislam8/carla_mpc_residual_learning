"""
Capture what the OCP was actually fed when it failed.

HPIPM status 3 is NAN_SOL -- a NaN reached the QP -- so the useful question is
not "which weight should I change" but "which input was already bad".  This
records the solver inputs on every call together with the handful of derived
quantities that can actually produce a NaN in this formulation, so failures can
be attributed instead of guessed at.

The candidate NaN sources, all of which this records:

  1. 1 - kappa(s)*n  ->  0.  Appears as a denominator in sdot in both
     bicycle_model_mpcc_cbf.py and MPCController._dynamics_of_car.  Division by
     ~0 makes sdot enormous, which poisons the propagated state below.
  2. NaN / Inf in propagated_x.  solve() pins stage 0 with lbx = ubx =
     propagated_x, so a bad delay propagation is fed straight into the QP as an
     equality constraint with no slack to absorb it.
  3. s outside the curvature spline's knot range.  kapparef_s is a CasADi
     bspline; evaluated past its knots it extrapolates, and a cubic
     extrapolation diverges fast.
  4. Barrier b -> 0.  b = sqrt((ds/a)^2 + (dn/b)^2) has an unbounded gradient at
     the origin, i.e. when the ego is on top of an obstacle.
  5. lh > uh.  Inverted bounds make the stage QP infeasible.
  6. v outside [v_min, v_max].

Usage -- set CARLA_MPC_DIAG=1 (or pass enabled=True) and run as normal:

    CARLA_MPC_DIAG=1 ./run_mpc_python.sh

then

    python tools/analyze_solver_failures.py diagnostics/
"""

import os
import time

import numpy as np


class SolverDiagnostics:
    """Per-solve recorder. Cheap enough to leave on (pure numpy, no I/O)."""

    _announced = False

    def __init__(self, enabled=None, out_dir="diagnostics"):
        if enabled is None:
            enabled = os.environ.get("CARLA_MPC_DIAG", "0") not in ("0", "", "false")
        self.enabled = bool(enabled)
        self.out_dir = out_dir
        self.records = []
        self._warned = False

        # Say so once per process.  Silently recording nothing because an env
        # var was missing is the failure mode worth ruling out immediately.
        if not SolverDiagnostics._announced:
            SolverDiagnostics._announced = True
            if self.enabled:
                print(f"[diag] recording solver inputs -> "
                      f"{os.path.abspath(self.out_dir)}/")
            else:
                print("[diag] disabled (set CARLA_MPC_DIAG=1 to record "
                      "solver inputs)")

    # ------------------------------------------------------------------ record

    def record(self, status, s, n, alpha, v, D, delta, propagated_x, obstacles,
               kappa_fn, path_length, n_min, n_max, lh=None, uh=None,
               v_min=None, v_max=None, a_long=4.0, b_lat=2.0,
               horizon_s=None, solver=None):
        """
        Store one solve. `status` is the acados return code (0 = success).

        Everything is coerced to plain floats/arrays so the record survives
        being written to npz even when the solver hands back CasADi types.
        """
        if not self.enabled:
            return

        rec = {
            "t": time.time(),
            "status": int(status),
            "s": float(s), "n": float(n), "alpha": float(alpha),
            "v": float(v), "D": float(D), "delta": float(delta),
            "path_length": float(path_length),
            "n_min": float(n_min), "n_max": float(n_max),
        }

        prop = np.asarray(propagated_x, dtype=float).ravel()
        rec["propagated_x"] = prop
        rec["prop_nonfinite"] = int(not np.all(np.isfinite(prop)))
        # A delay propagation of ~0.03 s should move s by well under a metre.
        rec["prop_s_jump"] = float(abs(prop[0] - s)) if prop.size > 0 else np.nan

        obs = np.asarray(obstacles, dtype=float).reshape(-1, 2)
        rec["obstacles"] = obs
        rec["obs_nonfinite"] = int(not np.all(np.isfinite(obs)))

        # --- 1 & 3: curvature and the sdot denominator ----------------------
        def kappa(at_s):
            try:
                return float(kappa_fn(at_s))
            except Exception:
                return np.nan

        k_now = kappa(s)
        rec["kappa"] = k_now
        rec["denom"] = float(1.0 - k_now * n) if np.isfinite(k_now) else np.nan

        # Same denominator swept over the arc lengths the horizon will visit.
        if horizon_s is None:
            horizon_s = [s]
        denoms, kappas = [], []
        for hs in horizon_s:
            kh = kappa(hs)
            kappas.append(kh)
            denoms.append(1.0 - kh * n if np.isfinite(kh) else np.nan)
        denoms = np.asarray(denoms, dtype=float)
        rec["denom_min_abs"] = float(np.nanmin(np.abs(denoms))) if denoms.size else np.nan
        rec["kappa_max_abs"] = float(np.nanmax(np.abs(kappas))) if kappas else np.nan
        rec["s_beyond_path"] = int(max(horizon_s) > path_length)
        rec["s_horizon_max"] = float(max(horizon_s))

        # --- 4: closest barrier value ---------------------------------------
        b_min = np.inf
        for s_obs, n_obs in obs:
            if s_obs <= -50.0:
                continue
            if s > (s_obs + 5.0):      # matches the model's ignore-behind test
                continue
            b_min = min(b_min, float(np.sqrt(((s - s_obs) / a_long) ** 2
                                             + ((n - n_obs) / b_lat) ** 2)))
        rec["barrier_min"] = float(b_min)
        rec["n_active_obs"] = int(np.sum(obs[:, 0] > -50.0))

        # --- 5: inverted bounds ---------------------------------------------
        if lh is not None and uh is not None:
            lh = np.asarray(lh, dtype=float)
            uh = np.asarray(uh, dtype=float)
            rec["bounds_inverted"] = int(np.any(lh > uh))
            rec["bounds_inverted_idx"] = int(np.argmax(lh > uh)) if np.any(lh > uh) else -1
        else:
            rec["bounds_inverted"] = 0
            rec["bounds_inverted_idx"] = -1
        rec["n_bounds_inverted"] = int(n_min > n_max)

        # --- 6: speed outside the model's own bounds -------------------------
        if v_min is not None and v_max is not None:
            rec["v_out_of_bounds"] = int(not (v_min <= v <= v_max))
        else:
            rec["v_out_of_bounds"] = 0

        # --- solver internals -------------------------------------------------
        rec.update(self._solver_stats(solver))

        self.records.append(rec)

        if status != 0 and not self._warned:
            self._warned = True
            print("\n[diag] first solver failure -- likely cause: "
                  f"{self.explain(rec)}")
            print("[diag] recording continues; run tools/analyze_solver_failures.py "
                  "on the saved file for the full picture\n")

    @staticmethod
    def _solver_stats(solver):
        out = {"qp_stat": -1, "sqp_iter": -1, "qp_iter": -1, "solve_time": np.nan}
        if solver is None:
            return out
        for key, field in (("qp_stat", "qp_stat"), ("sqp_iter", "sqp_iter"),
                           ("qp_iter", "qp_iter"), ("solve_time", "time_tot")):
            try:
                val = solver.get_stats(field)
                out[key] = float(np.max(np.asarray(val, dtype=float)))
            except Exception:
                pass
        return out

    # ----------------------------------------------------------------- explain

    @staticmethod
    def explain(rec):
        """One-line attribution for a single record, most specific first."""
        if rec["prop_nonfinite"]:
            return ("propagated_x is NaN/Inf -- it is pinned as the stage-0 "
                    "equality, so the QP starts poisoned")
        if np.isfinite(rec["denom_min_abs"]) and rec["denom_min_abs"] < 0.05:
            return (f"1 - kappa*n = {rec['denom_min_abs']:.4f} over the horizon "
                    f"(kappa_max={rec['kappa_max_abs']:.4f}, n={rec['n']:.2f}) "
                    "-- sdot denominator near zero")
        if rec["prop_s_jump"] > 5.0:
            return (f"delay propagation moved s by {rec['prop_s_jump']:.1f} m "
                    "in one step -- sdot blew up")
        if rec["s_beyond_path"]:
            return (f"horizon reaches s={rec['s_horizon_max']:.1f} past "
                    f"path_length={rec['path_length']:.1f} -- curvature spline "
                    "is extrapolating")
        if rec["barrier_min"] < 0.15:
            return (f"barrier b={rec['barrier_min']:.3f} -- sqrt gradient is "
                    "unbounded this close to an obstacle")
        if rec["bounds_inverted"] or rec["n_bounds_inverted"]:
            return f"inverted constraint bounds (h index {rec['bounds_inverted_idx']})"
        if rec["obs_nonfinite"]:
            return "NaN/Inf in the obstacle parameters"
        if rec["v_out_of_bounds"]:
            return f"v={rec['v']:.2f} outside the model's speed bounds"
        return ("no input-side cause found -- points at cost/Hessian "
                "conditioning rather than a bad input")

    # -------------------------------------------------------------------- save

    def save(self, tag="episode"):
        """Write the buffer to <out_dir>/<tag>_<n>.npz and clear it."""
        if not self.enabled or not self.records:
            return None

        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{tag}_{int(time.time())}.npz")

        keys = [k for k in self.records[0] if k not in ("propagated_x", "obstacles")]
        cols = {k: np.array([r[k] for r in self.records]) for k in keys}
        cols["propagated_x"] = np.stack([r["propagated_x"] for r in self.records])
        cols["obstacles"] = np.stack([r["obstacles"] for r in self.records])

        np.savez_compressed(path, **cols)
        n_fail = int(np.sum(cols["status"] != 0))
        print(f"[diag] {len(self.records)} solves, {n_fail} failures -> {path}")

        self.records = []
        self._warned = False
        return path
