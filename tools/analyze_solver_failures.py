#!/usr/bin/env python
"""
Attribute acados/HPIPM solver failures to their inputs.

    CARLA_MPC_DIAG=1 ./run_mpc_python.sh      # record
    python tools/analyze_solver_failures.py diagnostics/

Prints, in order:
  1. failure rate and the acados status codes seen
  2. how often each candidate NaN source was present, at failures vs successes
     -- the lift column is what matters, a cause that fires equally often on
     successful solves is not the cause
  3. how the state differed at failure
  4. the worst individual failures, each with a one-line attribution

Status codes: 0 ok, 1 NAN_DETECTED, 2 MAXITER, 3 MINSTEP, 4 QP_FAIL.
HPIPM qp_stat: 0 ok, 1 MAX_ITER, 2 MIN_STEP, 3 NAN_SOL, 4 INCONS_EQ.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'mpc_controller', 'src'))
from solver_diagnostics import SolverDiagnostics  # noqa: E402


# name -> (predicate over the loaded columns, human description)
CHECKS = {
    "propagated_x NaN/Inf": (
        lambda d: d["prop_nonfinite"] > 0,
        "stage-0 equality poisoned by a bad delay propagation"),
    "1-kappa*n < 0.05": (
        lambda d: np.abs(d["denom_min_abs"]) < 0.05,
        "sdot denominator near zero over the horizon"),
    "1-kappa*n < 0.20": (
        lambda d: np.abs(d["denom_min_abs"]) < 0.20,
        "sdot denominator getting small"),
    "prop s jump > 5 m": (
        lambda d: d["prop_s_jump"] > 5.0,
        "delay propagation blew up"),
    "s beyond path end": (
        lambda d: d["s_beyond_path"] > 0,
        "curvature spline extrapolating past its knots"),
    "barrier < 0.15": (
        lambda d: d["barrier_min"] < 0.15,
        "ego essentially on top of an obstacle; sqrt gradient unbounded"),
    "barrier < 0.50": (
        lambda d: d["barrier_min"] < 0.50,
        "close to an obstacle"),
    "inverted lh > uh": (
        lambda d: d["bounds_inverted"] > 0,
        "empty constraint set for a stage"),
    "inverted n bounds": (
        lambda d: d["n_bounds_inverted"] > 0,
        "n_min_adaptive > n_max_adaptive"),
    "obstacles NaN/Inf": (
        lambda d: d["obs_nonfinite"] > 0,
        "bad obstacle parameters"),
    "v out of bounds": (
        lambda d: d["v_out_of_bounds"] > 0,
        "speed outside the model's own limits"),
    # The lateral bound carries Zl[2] = 5e5 against cost terms of order 1, so a
    # metre of slack puts ~1e5 into the QP Hessian.  Saturating or exceeding it
    # is the condition to watch, not the barrier.
    "n outside [n_min,n_max]": (
        lambda d: (d["n"] < d["n_min"]) | (d["n"] > d["n_max"]),
        "lateral bound violated -- its 5e5 slack weight dominates the QP"),
    "n within 0.5m of bound": (
        lambda d: (d["n"] < d["n_min"] + 0.5) | (d["n"] > d["n_max"] - 0.5),
        "lateral bound nearly active"),
    "|n| > 2 m": (
        lambda d: np.abs(d["n"]) > 2.0,
        "large lateral excursion"),
    "obstacle present": (
        lambda d: d["n_active_obs"] > 0,
        "at least one obstacle in the parameter vector"),
    "obstacle + near bound": (
        lambda d: (d["n_active_obs"] > 0)
        & ((d["n"] < d["n_min"] + 0.5) | (d["n"] > d["n_max"] - 0.5)),
        "avoiding an obstacle while pressed against the road edge"),
    "|kappa| > 0.05": (
        lambda d: np.abs(d["kappa"]) > 0.05,
        "high path curvature"),
}

STATE_KEYS = ["s", "n", "alpha", "v", "D", "delta", "kappa", "denom",
              "denom_min_abs", "barrier_min", "prop_s_jump", "n_active_obs",
              "n_min", "n_max", "n_slack"]


def load(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.npz")))
        else:
            files.append(p)
    if not files:
        sys.exit(f"no .npz files found in {paths}")

    cols = {}
    for f in files:
        with np.load(f) as z:
            for k in z.files:
                cols.setdefault(k, []).append(z[k])
    merged = {k: np.concatenate(v, axis=0) for k, v in cols.items()}

    # How far n is outside its bounds (0 when inside).  This is the quantity the
    # 5e5 slack weight multiplies, so it is worth seeing directly.
    if {"n", "n_min", "n_max"} <= merged.keys():
        merged["n_slack"] = np.maximum(
            0.0, np.maximum(merged["n_min"] - merged["n"],
                            merged["n"] - merged["n_max"]))

    print(f"loaded {len(files)} file(s), {len(merged['status'])} solves\n")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="npz files or a directory")
    ap.add_argument("--top", type=int, default=8,
                    help="how many individual failures to print")
    args = ap.parse_args()

    d = load(args.paths)
    status = d["status"]
    fail = status != 0
    n_fail, n_ok = int(fail.sum()), int((~fail).sum())

    print("=" * 72)
    print(f"FAILURES: {n_fail}/{len(status)} ({100*n_fail/max(1,len(status)):.1f}%)")
    for code in np.unique(status):
        print(f"   acados status {int(code)}: {int((status == code).sum())}")
    if "qp_stat" in d:
        for code in np.unique(d["qp_stat"][fail]) if n_fail else []:
            print(f"   qp_stat {int(code)} among failures: "
                  f"{int((d['qp_stat'][fail] == code).sum())}")
    print("=" * 72)

    if n_fail == 0:
        print("\nno failures recorded -- nothing to attribute")
        return

    print("\nCANDIDATE CAUSES            at failure   at success       lift")
    print("-" * 72)
    rows = []
    for name, (pred, _) in CHECKS.items():
        try:
            hit = pred(d)
        except KeyError:
            continue
        p_fail = hit[fail].mean() if n_fail else 0.0
        p_ok = hit[~fail].mean() if n_ok else 0.0
        lift = p_fail / p_ok if p_ok > 1e-9 else (np.inf if p_fail > 0 else 0.0)
        rows.append((lift, p_fail, p_ok, name))
    for lift, p_fail, p_ok, name in sorted(rows, reverse=True):
        flag = "  <<<" if (p_fail > 0.2 and lift > 3) else ""
        lift_s = "inf" if np.isinf(lift) else f"{lift:.1f}x"
        print(f"  {name:<26} {100*p_fail:6.1f}%     {100*p_ok:6.1f}%   {lift_s:>8}{flag}")
    print("-" * 72)
    print("  lift = how much more often this holds at failures than successes.")
    print("  A cause with high 'at failure' AND high lift is the one to chase.")

    print("\nSTATE AT FAILURE vs SUCCESS")
    print("-" * 72)
    print(f"  {'':<16}{'fail mean':>12}{'ok mean':>12}{'fail min':>12}{'fail max':>12}")
    for k in STATE_KEYS:
        if k not in d:
            continue
        v = d[k].astype(float)
        with np.errstate(invalid='ignore'):
            print(f"  {k:<16}{np.nanmean(v[fail]):12.4f}{np.nanmean(v[~fail]):12.4f}"
                  f"{np.nanmin(v[fail]):12.4f}{np.nanmax(v[fail]):12.4f}")

    print(f"\nWORST {args.top} FAILURES")
    print("-" * 72)
    idx = np.where(fail)[0]
    # most interesting first: smallest denominator, then smallest barrier
    order = sorted(idx, key=lambda i: (float(d["denom_min_abs"][i]),
                                       float(d["barrier_min"][i])))
    for i in order[:args.top]:
        rec = {k: (d[k][i] if d[k].ndim == 1 else d[k][i]) for k in d}
        print(f"  #{i}  status={int(rec['status'])} "
              f"s={float(rec['s']):.1f} n={float(rec['n']):+.2f} "
              f"v={float(rec['v']):.1f} kappa={float(rec['kappa']):+.4f} "
              f"denom_min={float(rec['denom_min_abs']):.4f} "
              f"barrier={float(rec['barrier_min']):.3f}")
        print(f"      -> {SolverDiagnostics.explain(rec)}")

    print("\nVERDICT")
    print("-" * 72)
    top = sorted(rows, reverse=True)
    if top and top[0][1] > 0.2 and top[0][0] > 3:
        print(f"  '{top[0][3]}' fires on {100*top[0][1]:.0f}% of failures "
              f"({'inf' if np.isinf(top[0][0]) else f'{top[0][0]:.1f}x'} lift).")
        print(f"  {CHECKS[top[0][3]][1]}")
        print("  Fix that input before touching any cost weight.")
    else:
        print("  No input-side cause separates failures from successes.")
        print("  That points at cost/Hessian conditioning, not a bad input:")
        print("    - EXTERNAL cost is differentiated exactly and contains")
        print("      if_else/sign/fmax/fmin/fabs, so the Hessian can be indefinite")
        print("    - slack spread Zl[2]=5e5 vs Zl[0]=1e-3 is ~5e8")
        print("  Try regularize_method='CONVEXIFY' + levenberg_marquardt, and")
        print("  smooth the switches, one at a time, re-running this each time.")


if __name__ == "__main__":
    main()
