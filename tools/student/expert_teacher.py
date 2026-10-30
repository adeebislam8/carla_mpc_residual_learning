#!/usr/bin/env python
"""
Rule experts as TEACHERS for the tactical student: direct go / yield / dv
labels for every recorded frame, in the npz layout build_tactical.py reads
next to ORION's and SimLingo's (teacher ablation, E3a).

    python tools/student/expert_teacher.py --mode privileged \\
        --rec results/b2d/rec_train_clips/rec --out results/b2d/rec_train_clips/labels_privileged
    python tools/student/expert_teacher.py --mode rules --rec ... --out .../labels_rules

  privileged  "Learning by Cheating" (Chen et al., CoRL 2019) analogue: the
              decision rules below on SIMULATOR TRUTH -- the recording's 'gt'
              block (true actors, true lane frame, true light state, true road
              curvature).  PDM-Lite-style: current state + constant-velocity
              forecasts, no learning.  Needs recordings made after 2026-10-18
              (mpcc_b2d_agent.py _gt_scene).
  rules       the SAME rules on the PERCEIVED scene (LiDAR tracks, estimated
              pose, camera light) -- the hand-engineered teacher, i.e. what a
              student learns when the labels carry no knowledge beyond our own
              stack.

Rules (one function each, shared by both modes):
  go     at a go decision frame (go_label's: in-lane blocker 8-40 m, ego in
         lane): pass iff the blocker is a PARKED obstacle (never seen moving,
         or stopped >= 15 s -- queued traffic moved recently) and the frame's
         own gap check passes (build_tactical.go_unsafe: oncoming meets us
         before the pass is done, fast car behind, slow car ahead in the
         passing lane)
  yield  at a yield decision frame: wait iff a vehicle or walker, forecast at
         constant Frenet velocity, is within 3 m of our path inside the
         junction window [entry - 2, entry + 22] m while we would be there
         (t_in - 1 s .. t_out + 1 s at max(v, 3) m/s)
  dv     desired speed min(15, sqrt(3 / kappa), lead time-gap speed, 4 m/s
         near a walker by the path) minus ours, clipped to [-8, 3]
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, os.path.join(ROOT, 'src')]
from build_tactical import go_unsafe  # noqa: E402
from student.tactical import (LANE_D, blocker_info, frame_features,  # noqa: E402
                              yield_decision_frame)

V_MAX = 15.0
A_LAT = 3.0
PARKED_S = 15.0


def pseudo_scene(rec, mode):
    """The scene the rules look at: the recorded (perceived) one, or for
    'privileged' the same fields from the truth block.  None if unusable."""
    sc = rec['scene']
    if mode == 'rules':
        return dict(sc, kappa=route_kappa(sc.get('route_fl') or []))
    gt = rec.get('gt')
    if not gt:
        return None
    stop = None
    if gt.get('light') in ('red', 'yellow') and gt.get('light_ds') is not None:
        stop = gt['light_ds']
    lead = [a for a in gt['actors'] if abs(a['d'] - gt['d']) < LANE_D and a['ds'] > 0
            and a['vs'] > 0.5 and not a['type'].startswith('walker')]
    lead = min(lead, key=lambda a: a['ds']) if lead else None
    return dict(v=sc['v'], d=gt['d'], actors=gt['actors'], route_fl=sc.get('route_fl'),
                entry_ds=sc.get('entry_ds'), in_junction=sc.get('in_junction'),
                junction_ds=sc.get('junction_ds'), stop_ds=stop,
                lead_ds=None if lead is None else lead['ds'],
                lead_v=None if lead is None else lead['vs'], kappa=gt.get('kappa', 0.0))


def route_kappa(route_fl):
    """Max |curvature| of the perceived route ahead (3-point circles)."""
    p = np.asarray(route_fl, float)
    k = 0.0
    for a, b, c in zip(p, p[1:], p[2:]):
        ab, bc, ac = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(c - a)
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if ab * bc * ac > 1e-6:
            k = max(k, 2.0 * area2 / (ab * bc * ac))
    return k


def go_rule(sc):
    b = blocker_info(sc)
    if b is None or abs(sc['d']) > 1.0:
        return -1
    v = float(sc['v'])                       # same reach gate as tactical.go_label (v2)
    if b['ds'] < 3.0 or not (b['ds'] < 15.0 or b['ds'] < 3.5 * max(v, 1.0)):
        return -1
    # blocker_info clips stopped_for at 10 s; read the raw one
    inl = [a for a in sc['actors'] if abs(a['d']) < LANE_D and abs(a['ds'] - b['ds']) < 0.5]
    stopped = max((a.get('stopped_for', 0.0) for a in inl), default=0.0)
    parked = abs(b['vs']) < 0.5 and stopped >= PARKED_S
    if not parked:
        return 0
    return 0 if go_unsafe(frame_features(sc)) else 1


def yield_rule(sc, v2=False):
    """v2 (gap acceptance, 2026-10-30): the window is timed at the speed we
    would CROSS at, max(v, 6) m/s -- at the approach speed (>= 3) a slow
    approach widened the window and the expert waited on 62 % of entries --
    and only MOVING actors conflict (stopped ones are the lead / obstacle
    layer's, e.g. a queue in our own lane)."""
    if not yield_decision_frame(sc):
        return -1
    e, v = float(sc['entry_ds']), max(float(sc['v']), 6.0 if v2 else 3.0)
    t_in, t_out = e / v - 1.0, (e + 20.0) / v + 1.0
    for a in sc['actors']:
        if a['type'].startswith('static'):
            continue
        same_dir = a['vs'] > 0.5 and abs(a['vd']) < 1.0
        if same_dir:                                   # traffic going our way: not a conflict
            continue
        if v2 and math.hypot(a['vs'], a['vd']) < 0.5:
            continue
        for t in np.arange(max(t_in, 0.0), t_out + 1e-6, 0.25):
            s_t, d_t = a['ds'] + a['vs'] * t, a['d'] + a['vd'] * t
            if e - 2.0 <= s_t <= e + 22.0 and abs(d_t - sc['d']) < 3.0:
                return 1
    return 0


def dv_rule(sc):
    v = float(sc['v'])
    v_des = V_MAX
    k = float(sc.get('kappa') or 0.0)
    if k > 1e-3:
        v_des = min(v_des, math.sqrt(A_LAT / k))
    if sc.get('lead_ds') is not None:
        gap = float(sc['lead_ds'])
        v_des = min(v_des, max(float(sc.get('lead_v') or 0.0) + (gap - 8.0) / 2.0, 0.0))
    for a in sc['actors']:
        if a['type'].startswith('walker') and 0.0 < a['ds'] < 25.0 and abs(a['d'] - sc['d']) < 4.0:
            v_des = min(v_des, 4.0)
    return float(np.clip(v_des - v, -8.0, 3.0))


def label_episode(ep, mode, stride=1, yield_v2=False):
    recs = [json.loads(l) for l in open(os.path.join(ep, 'meta.jsonl')) if l.strip()][::stride]
    steps, empty, go, yl, dv = [], [], [], [], []
    for r in recs:
        steps.append(int(r['step']))
        sc = pseudo_scene(r, mode) if 'scene' in r else None
        if sc is None:
            empty.append(True)
            go.append(-1), yl.append(-1), dv.append(np.nan)
            continue
        empty.append(False)
        go.append(go_rule(sc)), yl.append(yield_rule(sc, yield_v2)), dv.append(dv_rule(sc))
    return dict(steps=np.array(steps), empty=np.array(empty), go=np.array(go, np.int8),
                yield_=np.array(yl, np.int8), dv=np.array(dv, np.float32),
                teacher=np.array(mode), kind=np.array('direct'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=('privileged', 'rules'), required=True)
    ap.add_argument('--rec', nargs='+', required=True, help='recording roots (contain episode dirs)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--yield-v2', action='store_true',
                    help='gap-acceptance yield rule (crossing speed >= 6 m/s, moving actors only)')
    ap.add_argument('--stride', type=int, default=1,
                    help='label every N-th frame -- use the image teachers\' stride, so every\n'
                         'teacher in the ablation trains on the same frames')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n_ep = n_fr = n_nogt = 0
    tot = dict(go1=0, go0=0, y1=0, y0=0)
    for root in args.rec:
        for ep in sorted(glob.glob(os.path.join(os.path.expanduser(root), '*'))):
            if not os.path.exists(os.path.join(ep, 'meta.jsonl')):
                continue
            out = label_episode(ep, args.mode, args.stride, args.yield_v2)
            if out['empty'].all():
                n_nogt += 1
                print(f'  {os.path.basename(ep)}: no usable frames (no gt block?) -- skipped')
                continue
            np.savez_compressed(os.path.join(args.out, os.path.basename(ep) + '.npz'), **out)
            n_ep += 1
            n_fr += len(out['steps'])
            tot['go1'] += int((out['go'] == 1).sum()); tot['go0'] += int((out['go'] == 0).sum())
            tot['y1'] += int((out['yield_'] == 1).sum()); tot['y0'] += int((out['yield_'] == 0).sum())
    print(f'{args.mode} expert: {n_ep} episodes, {n_fr} frames -> {args.out}  '
          f'(go {tot["go1"]}/{tot["go0"]} pass/wait, yield {tot["y1"]}/{tot["y0"]} wait/enter'
          + (f'; {n_nogt} episodes without truth' if n_nogt else '') + ')')


if __name__ == '__main__':
    main()
