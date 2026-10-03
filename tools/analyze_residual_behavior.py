#!/usr/bin/env python
"""
What does the residual actually DO in the steps leading up to a collision?

Reads the per-step misbehaviour traces benchmark_mpcc.py writes to
results/traces/<label>_seed<seed>_ep<ep>.csv -- one file per episode that
ended in collision (carlaEnv.py only attaches a trace to info on that
outcome). Each row is one control step: residual action, alpha, nominal vs
final control, nearest obstacle distance, and the full reward-term breakdown
for that step.

This is a DIFFERENT question from the aggregate mean_reward_*/mean_alpha
numbers in the main report: those answer "on average, how much does each
term contribute". This answers "what was the residual doing right before
THIS failure", across every collision in the run.

Usage:
    python tools/analyze_residual_behavior.py --label sac_b5_Town01
    python tools/analyze_residual_behavior.py --label sac_b5_Town01 --window 10
"""
import argparse
import csv
import glob
import os

import numpy as np


def _load_trace(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    def col(name, default=np.nan):
        out = []
        for r in rows:
            v = r.get(name, '')
            try:
                out.append(float(v) if v != '' else default)
            except ValueError:
                out.append(default)
        return np.array(out)

    return {
        'residual_throttle': col('residual_throttle'),
        'residual_steer': col('residual_steer'),
        'alpha': col('alpha'),
        'nearest_obstacle_ds': col('nearest_obstacle_ds'),
        'n_steps': len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True,
                     help='same --label used for the benchmark_mpcc.py run')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--window', type=int, default=10,
                     help='how many steps before the collision count as '
                          '"near-collision" (default 10, i.e. 0.5s at 20Hz)')
    args = ap.parse_args()

    pattern = os.path.join(args.out_dir, 'traces', f"{args.label}_seed*_ep*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"no traces found matching {pattern}")
        print("(traces only exist for episodes that ended in collision -- "
              "re-run the benchmark if this label predates that instrumentation)")
        return

    near_steer, rest_steer = [], []
    near_throttle, rest_throttle = [], []
    near_alpha, rest_alpha = [], []
    near_ds, rest_ds = [], []
    obstacle_present_steer, obstacle_absent_steer = [], []

    for p in paths:
        t = _load_trace(p)
        n = t['n_steps']
        if n < 2:
            continue
        w = min(args.window, n)

        near_steer.append(np.nanmean(np.abs(t['residual_steer'][-w:])))
        rest_steer.append(np.nanmean(np.abs(t['residual_steer'][:-w])) if n > w else np.nan)
        near_throttle.append(np.nanmean(np.abs(t['residual_throttle'][-w:])))
        rest_throttle.append(np.nanmean(np.abs(t['residual_throttle'][:-w])) if n > w else np.nan)
        near_alpha.append(np.nanmean(t['alpha'][-w:]))
        rest_alpha.append(np.nanmean(t['alpha'][:-w]) if n > w else np.nan)
        near_ds.append(np.nanmean(t['nearest_obstacle_ds'][-w:]))
        rest_ds.append(np.nanmean(t['nearest_obstacle_ds'][:-w]) if n > w else np.nan)

        has_obs = ~np.isnan(t['nearest_obstacle_ds'])
        if has_obs.any():
            obstacle_present_steer.extend(np.abs(t['residual_steer'][has_obs]).tolist())
        if (~has_obs).any():
            obstacle_absent_steer.extend(np.abs(t['residual_steer'][~has_obs]).tolist())

    def fmt(a):
        a = np.array(a, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return "  n/a"
        return f"{np.mean(a):+.4f}  (n={len(a)}, sd={np.std(a):.4f})"

    print("=" * 72)
    print(f"RESIDUAL MISBEHAVIOUR TRACE ANALYSIS  --  {args.label}")
    print(f"  {len(paths)} collision episodes, last {args.window} steps = "
          f"'near-collision window'")
    print("=" * 72)
    print()
    print("mean |residual steer|,   near-collision window:", fmt(near_steer))
    print("mean |residual steer|,   rest of episode:       ", fmt(rest_steer))
    print("mean |residual throttle|,near-collision window:", fmt(near_throttle))
    print("mean |residual throttle|,rest of episode:       ", fmt(rest_throttle))
    print()
    print("mean alpha,              near-collision window:", fmt(near_alpha))
    print("mean alpha,              rest of episode:       ", fmt(rest_alpha))
    print("  -> if near-collision alpha is NOT lower than the rest-of-episode")
    print("     value, the authority gate is not actually tightening up when")
    print("     it matters most, even in episodes that still end badly.")
    print()
    print("mean nearest-obstacle ds,near-collision window:", fmt(near_ds))
    print("mean nearest-obstacle ds,rest of episode:       ", fmt(rest_ds))
    print()
    print("mean |residual steer| WITH an obstacle in range:", fmt(obstacle_present_steer))
    print("mean |residual steer| with NO obstacle in range:", fmt(obstacle_absent_steer))
    print("  -> the overtake attractor (n_overtake) is obstacle-blind by")
    print("     construction (mpcc-overtake-attractor). If the residual's own")
    print("     steer magnitude does NOT differ with/without an obstacle")
    print("     present, it has not learned to compensate for that blindness")
    print("     either -- it is reacting to the same information the MPCC is.")


if __name__ == '__main__':
    main()
