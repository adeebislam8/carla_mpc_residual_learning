#!/usr/bin/env python
"""
Recorded episodes + ORION labels -> one training set for the residual student.

    python tools/student/build_dataset.py \\
        --rec ~/Documents/nett/orion_data/rec_town01 \\
        --labels ~/Documents/nett/orion_data/labels_town01 \\
        --out ~/Documents/nett/orion_data/ds_round0.npz

DAgger: build one file per round and pass them all to train_student.py; the
rounds are aggregated there, which is what makes it DAgger.

Per recorded frame i the target is

    a*_i = clip((u_teacher_i - u_nom_i) / residual_max, -1, 1)

  u_teacher_i  ORION's label for frame i: steering = -ORION steer; throttle
               = u_nom + D(v_orion) - D(v_realised), see
               student.labels.teacher_throttle_relative
  u_nom_i      the MPCC's action for frame i's state -- recorded on the NEXT
               tick as u_nom_prev_step, hence record i+1
  x_i          obs.npy row i, the v2 observation after the tick that
               produced frame i: the same vector the policy sees when it
               chooses the action for that state

and the chunk target for speculative execution is Y_i = (a*_i, ..., a*_{i+K-1})
with a mask where frames are missing, the label is ORION's empty fallback, or
the episode ends.

Stored per frame, for the trainer to filter or weight:
  active   an obstacle 0 < ds < window ahead (where the residual acts)
  precoll  within the last --precoll-frames frames of a collision episode
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from student.labels import (plan_speed, residual_target, teacher_action,  # noqa: E402
                            teacher_throttle_relative)

# v2 observation layout (carlaEnv._get_observation_v2): obstacles start after
# ego 7 + corridor 2 + nominal 4 + speed error 1 + curvature 30, 4 per slot.
OBS_V2_DIM = 68
OBS_V2_OBSTACLE0 = 44
OBS_V2_PER_OBS = 4


def window_active(obs, window):
    """(N,) bool: any present obstacle slot with 0 < ds < window."""
    slots = obs[:, OBS_V2_OBSTACLE0:].reshape(len(obs), -1, OBS_V2_PER_OBS)
    present, ds = slots[..., 0] > 0.5, slots[..., 1]
    return np.any(present & (ds > 0.0) & (ds < window), axis=1)


def episode_samples(rec_dir, label_path, args):
    recs = [json.loads(l) for l in open(rec_dir / 'meta.jsonl') if l.strip()]
    obs = np.load(rec_dir / 'obs.npy')
    summary = json.load(open(rec_dir / 'summary.json'))
    if obs.shape != (len(recs), OBS_V2_DIM):
        raise ValueError(f'{rec_dir.name}: obs {obs.shape} for {len(recs)} records; '
                         f'need obs_version v2 ({OBS_V2_DIM} dims)')
    lab = np.load(label_path)
    by_step = {r['step']: i for i, r in enumerate(recs)}

    n = len(recs)
    speeds = np.array([r['speed'] for r in recs])
    v_realised = plan_speed(speeds)     # ORION's speed formula on what happened
    a_star = np.full((n, 2), np.nan)
    for j, step in enumerate(lab['steps']):
        i = by_step.get(int(step))
        if i is None or i + 1 >= n or lab['empty'][j]:
            continue
        u_nom = recs[i + 1]['u_nom_prev_step']
        if u_nom[0] is None or u_nom[1] is None:
            continue
        u_t = teacher_action(lab['ctrl'][j], lab['desired_speed'][j],
                             speeds[i], tau=args.tau)
        if args.throttle_label == 'relative':
            u_t[0] = teacher_throttle_relative(lab['desired_speed'][j], v_realised[i],
                                               speeds[i], u_nom[0], tau=args.tau)
        a_star[i] = residual_target(u_t, u_nom, args.residual_max)

    valid = ~np.isnan(a_star[:, 0])
    K = args.chunk
    Y = np.zeros((n, K, 2), dtype=np.float32)
    M = np.zeros((n, K), dtype=bool)
    for k in range(K):
        idx = np.arange(n) + k
        ok = idx < n
        ok[ok] &= valid[idx[ok]]
        Y[ok, k] = a_star[idx[ok]]
        M[ok, k] = True

    keep = valid                       # first chunk step must be labelled
    outcome = summary.get('outcome')
    precoll = np.zeros(n, dtype=bool)
    if outcome == 'collision':
        precoll[max(0, n - args.precoll_frames):] = True
    return dict(X=obs[keep].astype(np.float32), Y=Y[keep], M=M[keep],
                active=window_active(obs, args.window)[keep],
                precoll=precoll[keep],
                step=np.array([r['step'] for r in recs])[keep]), outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', nargs='+', required=True, help='recording root dir(s)')
    ap.add_argument('--labels', nargs='+', required=True, help='label root dir(s)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=10,
                    help='K, steps per speculative chunk (10 = 0.5 s)')
    ap.add_argument('--residual-max', type=float, default=0.5,
                    help='must equal the env flag the student will run under')
    ap.add_argument('--window', type=float, default=25.0,
                    help='residual window, m (--residual-window)')
    ap.add_argument('--tau', type=float, default=1.0,
                    help='time constant of the speed -> throttle label, s')
    ap.add_argument('--precoll-frames', type=int, default=40)
    ap.add_argument('--throttle-label', default='relative', choices=['relative', 'absolute'],
                    help="'relative' (default): MPCC throttle + D(v_orion) - "
                         "D(v_realised), immune to the model's calibration offset; "
                         "'absolute': D(v_orion) alone (biased ~-0.25 measured)")
    args = ap.parse_args()

    label_files = {}
    for root in args.labels:
        for p in Path(root).expanduser().glob('*.npz'):
            label_files[p.stem] = p
    parts, names, outcomes, missing = [], [], [], 0
    for root in args.rec:
        for ep in sorted(Path(root).expanduser().iterdir()):
            if not (ep / 'meta.jsonl').exists():
                continue
            if ep.name not in label_files:
                missing += 1
                continue
            d, outcome = episode_samples(ep, label_files[ep.name], args)
            if len(d['X']) == 0:
                continue
            d['ep'] = np.full(len(d['X']), len(names), dtype=np.int32)
            parts.append(d)
            names.append(ep.name)
            outcomes.append(outcome)

    if not parts:
        sys.exit('no labelled episodes found')
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    config = dict(chunk=args.chunk, residual_max=args.residual_max,
                  window=args.window, tau=args.tau, obs_version='v2',
                  throttle_label=args.throttle_label,
                  precoll_frames=args.precoll_frames)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data, episodes=np.array(names),
                        outcomes=np.array(outcomes), config=json.dumps(config))

    a0 = data['Y'][:, 0]
    act = data['active']
    print(f'{len(names)} episodes ({missing} recorded but unlabelled, skipped), '
          f'{len(a0)} frames -> {out}')
    print(f'  outcomes: ' + ', '.join(f'{o} {outcomes.count(o)}' for o in sorted(set(outcomes))))
    print(f'  residual window open on {act.mean() * 100:.0f}% of frames '
          f'({act.sum()} frames -- what the student trains on by default)')
    print(f'  pre-collision frames: {data["precoll"].sum()} '
          f'({(data["precoll"] & act).sum()} with the window open)')
    for d, name in enumerate(('throttle', 'steering')):
        x = a0[act, d] if act.any() else a0[:, d]
        print(f'  a* {name:8s} (window open): mean |a| {np.abs(x).mean():.3f}, '
              f'at the +-1 clip {np.mean(np.abs(x) >= 0.999) * 100:.0f}%')


if __name__ == '__main__':
    main()
