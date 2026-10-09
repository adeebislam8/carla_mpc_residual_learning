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

# Longitudinal model of bicycle_model_mpcc_cbf.py (same constants as
# residual_authority._MODEL_LONG).
_M, _CM1, _CM2 = 2065.03, 9.36424211e+03, 4.08690122e+01
_CR0, _CR2, _CR3 = 5.84856121e+02, 2.04799356e+00, 1.13995833e+01
D_MIN, D_MAX = -0.5, 1.0          # the authority gate's actuator bounds


def throttle_from_speed(v_target, v, tau=1.0):
    """
    Throttle D that reaches v_target in tau seconds under the MPCC's own
    longitudinal model, by inverting Fxd = (Cm1 - Cm2 v) D - Cr2 v^2
    - Cr0 tanh(Cr3 v) with a = Fxd / m.

    ORION's PID turns its plan into a bang-bang brake (0 or 1), which makes a
    useless regression target: 56% of du* was clipped on the first episode.
    Its planned SPEED is smooth, so the longitudinal label comes from that,
    in the same units the MPCC commands.

    tau = 1 s: ORION's desired speed (from its PID's formula) is already the
    speed ~0.5-1 s down the plan.  At tau = 0.5 s a 2 m/s gap demands 4 m/s^2
    and saturates the throttle, recreating the bang-bang label.
    """
    a = (v_target - v) / tau
    D = (_M * a + _CR2 * v * v + _CR0 * np.tanh(_CR3 * v)) / (_CM1 - _CM2 * v)
    return float(np.clip(D, D_MIN, D_MAX))


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
            teach_thr_v=throttle_from_speed(float(lab['desired_speed'][j]),
                                            recs[i]['speed']),
        ))
    return rows, summary.get('outcome'), lab


def report(name, rows, outcome, residual_max, window):
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    du_thr = a['teach_thr'] - a['nom_thr']
    du_thr_v = a['teach_thr_v'] - a['nom_thr']
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
    for label, du in (('throttle (ORION PID)', du_thr),
                      ('throttle (from planned speed)', du_thr_v),
                      ('steering', du_steer)):
        print(f'   du* {label:29s}: median |du| {np.median(np.abs(du)):.3f}, '
              f'p90 {np.percentile(np.abs(du), 90):.3f}, '
              f'clipped by residual_max={residual_max}: '
              f'{np.mean(np.abs(du) > residual_max) * 100:.0f}%')

    if outcome == 'collision':
        w = slice(max(0, len(rows) - window), len(rows))
        print(f'   last {window} frames before the collision:')
        print(f'     ego speed {a["speed"][w].mean():.2f} m/s vs ORION planned '
              f'{a["v_orion"][w].mean():.2f} m/s '
              f'({"ORION would be slower" if a["v_orion"][w].mean() < a["speed"][w].mean() - 0.5 else "no slowdown from ORION"})')
        print(f'     throttle: MPCC {a["nom_thr"][w].mean():+.2f} vs ORION PID '
              f'{a["teach_thr"][w].mean():+.2f} / from speed '
              f'{a["teach_thr_v"][w].mean():+.2f} | steering: MPCC '
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
    ax[2].plot(t, a['teach_thr'], label='ORION PID throttle - brake', alpha=0.4)
    ax[2].plot(t, a['teach_thr_v'], label='ORION from planned speed (label)')
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
