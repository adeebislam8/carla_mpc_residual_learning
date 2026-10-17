#!/usr/bin/env python
"""
How good is the sensor-only perception?  Reads the diagnostic columns the
agent logs in privileged:false runs (perception vs ground truth, per tick).

    python tools/b2d/check_perception.py results/b2d/ssmoke_smoke

Per route and overall:
  pose error       GNSS/IMU estimate vs true position (median / p95, m)
  in-lane object   nearest object ahead in our lane (0-40 m):
                     detected   GT has one and perception has one within 3 m
                     missed     GT has one, perception none (or > 3 m off)
                     phantom    perception has one, GT none
                     range err  median |perceived ds - GT ds| when detected
  lights           how often each light state was reported
"""

import collections
import csv
import glob
import os
import sys

import numpy as np


def f(x):
    return None if x in ('', 'None', None) else float(x)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'results/b2d/ssmoke_smoke'
    files = sorted(glob.glob(os.path.join(run, 'traces', '*.csv')))
    if not files:
        sys.exit(f'no traces in {run}/traces')
    tot = collections.Counter()
    pose_all, err_all, lights_all = [], [], collections.Counter()
    print(f"{'route':48s} {'pose med/p95':>13s} {'det':>5s} {'miss':>5s} {'phan':>5s} {'rngerr':>7s}")
    for p in files:
        rows = list(csv.DictReader(open(p)))
        if not rows or 'gt_ds' not in rows[0]:
            continue
        c, pose, err = collections.Counter(), [], []
        for r in rows:
            pe, g, q = f(r.get('pose_err')), f(r.get('gt_ds')), f(r.get('per_ds'))
            if pe is not None:
                pose.append(pe)
            if g is not None and q is not None and abs(g - q) < 3.0:
                c['det'] += 1
                err.append(abs(g - q))
            elif g is not None:
                c['miss'] += 1
            elif q is not None:
                c['phan'] += 1
            lights_all[r.get('light') or 'none'] += 1
        n = max(sum(c.values()), 1)
        print(f"{os.path.basename(p)[:48]:48s} {np.median(pose) if pose else float('nan'):6.2f}/"
              f"{np.percentile(pose, 95) if pose else float('nan'):5.2f} "
              f"{100 * c['det'] / n:4.0f}% {100 * c['miss'] / n:4.0f}% {100 * c['phan'] / n:4.0f}% "
              f"{np.median(err) if err else float('nan'):6.2f}")
        tot.update(c)
        pose_all += pose
        err_all += err
    n = max(sum(tot.values()), 1)
    print(f"\nALL: pose median {np.median(pose_all):.2f} m (p95 {np.percentile(pose_all, 95):.2f}); "
          f"in-lane object detected {100 * tot['det'] / n:.0f}%, missed {100 * tot['miss'] / n:.0f}%, "
          f"phantom {100 * tot['phan'] / n:.0f}%, range error median {np.median(err_all) if err_all else float('nan'):.2f} m")
    print('light states reported (ticks):', dict(lights_all))


if __name__ == '__main__':
    main()
