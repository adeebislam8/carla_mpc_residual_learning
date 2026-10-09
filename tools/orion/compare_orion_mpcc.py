#!/usr/bin/env python
"""
Go/no-go check for ORION as the residual's teacher: on recorded episodes,
where does ORION's control disagree with what the MPCC did, and does the
disagreement point the right way before a collision?

    python tools/orion/compare_orion_mpcc.py \\
        --rec ~/Documents/nett/orion_data/test --labels ~/Documents/nett/orion_data/labels_test

Needs only numpy and matplotlib, so it runs in either conda env.

For every frame it builds the distillation target the student would get,

    du* = u_teacher - u_nom

in the MPC model's convention (see record_orion_episodes.py):
    u_teacher throttle = ORION throttle - ORION brake
    u_teacher steering = -ORION steer        (model steering > 0 = LEFT)
    u_nom              = the NEXT record's u_nom_prev_step, i.e. the MPCC's
                         action computed from this frame's state

and reports:
  * speed: ORION's planned speed vs the ego's actual speed
  * steering: correlation and sign agreement between ORION and the MPCC
  * du*: size distribution, and how much of it --residual-max would clip
  * pre-collision window: what ORION planned in the last N frames before a
    collision, versus what the MPCC did -- the case the paper cares about
and writes <labels>/<episode>_compare.png with the time series.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(rec_dir, label_path):
    recs = [json.loads(l) for l in open(rec_dir / 'meta.jsonl') if l.strip()]
    by_step = {r['step']: i for i, r in enumerate(recs)}
    lab = np.load(label_path)
    summary = json.load(open(rec_dir / 'summary.json'))

    rows = []
    for j, step in enumerate(lab['steps']):
        i = by_step.get(int(step))
        if i is None or i + 1 >= len(recs) or lab['empty'][j]:
            continue
        u_nom = recs[i + 1]['u_nom_prev_step']
        if u_nom[0] is None or u_nom[1] is None:
            continue
        steer, thr, brake = lab['ctrl'][j]
        rows.append(dict(
            step=int(step),
            speed=recs[i]['speed'],
            v_orion=float(lab['desired_speed'][j]),
            lat_orion=float(lab['plan_fl'][j, 2, 1]),   # left offset at 1.5 s
            command=recs[i]['command'],
            nom_thr=float(u_nom[0]), nom_steer=float(u_nom[1]),
            teach_thr=float(thr - brake), teach_steer=float(-steer),
        ))
    return rows, summary.get('outcome'), lab


def report(name, rows, outcome, residual_max, window):
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    du_thr = a['teach_thr'] - a['nom_thr']
    du_steer = a['teach_steer'] - a['nom_steer']

    print(f'\n== {name}  outcome={outcome}  frames compared={len(rows)}')
    print(f'   speed: ego median {np.median(a["speed"]):.2f} m/s | ORION planned '
          f'median {np.median(a["v_orion"]):.2f} m/s | ORION slower than ego on '
          f'{np.mean(a["v_orion"] < a["speed"] - 0.5) * 100:.0f}% of frames')
    if np.std(a['teach_steer']) > 1e-6 and np.std(a['nom_steer']) > 1e-6:
        corr = np.corrcoef(a['teach_steer'], a['nom_steer'])[0, 1]
    else:
        corr = float('nan')
    big = (np.abs(a['nom_steer']) > 0.05) | (np.abs(a['teach_steer']) > 0.05)
    agree = (np.mean(np.sign(a['teach_steer'][big]) == np.sign(a['nom_steer'][big])) * 100
             if big.any() else float('nan'))
    print(f'   steering: corr(ORION, MPCC) {corr:+.2f} | same direction on '
          f'{agree:.0f}% of frames where either steers > 0.05')
    for label, du in (('throttle', du_thr), ('steering', du_steer)):
        print(f'   du* {label:8s}: median |du| {np.median(np.abs(du)):.3f}, '
              f'p90 {np.percentile(np.abs(du), 90):.3f}, '
              f'clipped by residual_max={residual_max}: '
              f'{np.mean(np.abs(du) > residual_max) * 100:.0f}%')

    if outcome == 'collision':
        w = slice(max(0, len(rows) - window), len(rows))
        print(f'   last {window} frames before the collision:')
        print(f'     ego speed {a["speed"][w].mean():.2f} m/s vs ORION planned '
              f'{a["v_orion"][w].mean():.2f} m/s '
              f'({"ORION would be slower" if a["v_orion"][w].mean() < a["speed"][w].mean() - 0.5 else "no slowdown from ORION"})')
        print(f'     throttle: MPCC {a["nom_thr"][w].mean():+.2f} vs ORION '
              f'{a["teach_thr"][w].mean():+.2f} | steering: MPCC '
              f'{a["nom_steer"][w].mean():+.2f} vs ORION {a["teach_steer"][w].mean():+.2f}')
    return a


def plot(a, out_png, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = a['step']
    ax[0].plot(t, a['speed'], label='ego speed (MPCC)')
    ax[0].plot(t, a['v_orion'], label='ORION planned speed')
    ax[0].set_ylabel('m/s')
    ax[1].plot(t, a['nom_steer'], label='MPCC steering')
    ax[1].plot(t, a['teach_steer'], label='ORION steering (model sign)')
    ax[1].set_ylabel('steer (+ = left)')
    ax[2].plot(t, a['nom_thr'], label='MPCC throttle')
    ax[2].plot(t, a['teach_thr'], label='ORION throttle - brake')
    ax[2].set_ylabel('throttle')
    ax[2].set_xlabel('simulator step (0.05 s)')
    junction = a['command'] != 4
    for axis in ax:
        axis.fill_between(t, *axis.get_ylim(), where=junction, color='0.9',
                          step='mid', label='_nolegend_')
        axis.legend(loc='upper left', fontsize=8)
        axis.grid(alpha=0.3)
    ax[0].set_title(f'{title}   (grey = junction command)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(f'   plot -> {out_png}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', required=True, help='dir holding recorded episodes')
    ap.add_argument('--labels', required=True, help='dir holding orion_infer .npz')
    ap.add_argument('--residual-max', type=float, default=0.5)
    ap.add_argument('--window', type=int, default=40,
                    help='frames before a collision to summarise (40 = 2 s)')
    args = ap.parse_args()

    rec_root, lab_root = Path(args.rec).expanduser(), Path(args.labels).expanduser()
    for label_path in sorted(lab_root.glob('*.npz')):
        rec_dir = rec_root / label_path.stem
        if not rec_dir.is_dir():
            print(f'skip {label_path.name}: no recording at {rec_dir}')
            continue
        rows, outcome, _ = load(rec_dir, label_path)
        if not rows:
            print(f'skip {label_path.name}: nothing to compare')
            continue
        a = report(label_path.stem, rows, outcome, args.residual_max, args.window)
        plot(a, lab_root / f'{label_path.stem}_compare.png',
             f'{label_path.stem} ({outcome})')


if __name__ == '__main__':
    main()
